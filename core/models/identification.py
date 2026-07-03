"""
Card identification model -- the ArcFace integration point.

Two interchangeable heads share one ResNet-50 backbone and one inference API; the
active one is chosen by ``config.IDENTIFICATION_METHOD`` (overridable per call):

* ``"triplet"``  -- metric learning with ``TripletMarginLoss``; matching by
  nearest Euclidean anchor.
* ``"arcface"`` -- angular-margin softmax (``ArcFaceLayer``) during training;
  matching by highest cosine similarity at inference.

Optimisations over the original code (same libraries, behaviour-preserving):

* ``identify_cards`` for the triplet method is fully vectorised -- one
  ``torch.cdist`` instead of a Python loop over every anchor (the additive
  TripletMarginLoss margin does not change the arg-min, so the selection is
  identical).
* ``identify_cards_with_color_correction`` caches each anchor's per-channel BGR
  mean/std once instead of reloading the anchor image from disk on every frame.
"""

import os
import json
import math
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision.models import resnet50

from core.config import *  # noqa: F401,F403
from core.transforms import (get_valid_transform, get_inference_transform,
                             get_identification_train_transform)
from core.image_ops import transfer_rgb_distribution


def _resolve_method(method):
    return (method or IDENTIFICATION_METHOD).lower()


# --------------------------------------------------------------------------- #
# Architectures
# --------------------------------------------------------------------------- #
class ArcFaceLayer(nn.Module):
    """Angular-margin softmax head (used only during ArcFace training)."""

    def __init__(self, in_features, out_features, s=ARCFACE_SCALE, m=ARCFACE_MARGIN):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features  # number of classes (cards)
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input_embedding, label):
        cosine = F.linear(F.normalize(input_embedding), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros(cosine.size(), device=input_embedding.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        return output * self.s


def build_triplet_model(device):
    model = resnet50(weights='DEFAULT')
    num_filters = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_filters, 512),
        nn.ReLU(),
        nn.Linear(512, IDENTIFICATION_OUT_FEATURES),
    )
    for params in model.parameters():
        params.requires_grad = True
    return model.to(device)


def build_arcface_backbone(device, embedding_size=ARCFACE_EMBEDDING_SIZE):
    model = resnet50(weights='DEFAULT')
    num_filters = model.fc.in_features
    model.fc = nn.Sequential(
        nn.BatchNorm1d(num_filters),
        nn.Dropout(0.5),
        nn.Linear(num_filters, embedding_size),
        nn.BatchNorm1d(embedding_size),
    )
    for params in model.parameters():
        params.requires_grad = True
    return model.to(device)


def build_model(method=None, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    method = _resolve_method(method)
    if method == "arcface":
        return build_arcface_backbone(device)
    return build_triplet_model(device)


def load_model(method=None, fine_tuned=True, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
               eval_mode=True):
    """Load identification weights for the requested method."""
    method = _resolve_method(method)
    model = build_model(method, device)
    path = model_save_path(method, fine_tuned)
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except FileNotFoundError:
        print(f"identification model weight file does not exist: {path}", file=__import__('sys').stderr)
        raise
    state = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    model.load_state_dict(state)
    if eval_mode:
        model.eval()
    else:
        for params in model.fc.parameters():
            params.requires_grad = True
    return model


# --------------------------------------------------------------------------- #
# Negative sampling + datasets (triplet training)
# --------------------------------------------------------------------------- #
def select_targeted_negative_sample_strategy(anchor_item, anchor_list, pokemon_id_list, pokemon_pokedex_list,
                                            pokemon_type_list, pokemon_list, trainer_list, energy_list):
    anchor_card = pokemon_id_list[anchor_item]
    if anchor_card[0] == "Pokémon":
        random_number = random.random()
        if not anchor_card[1]:
            random_number += 0.5
        if random_number < 0.5:
            id_list = []
            for pokemon_number in anchor_card[1]:
                for pokemon_type in anchor_card[2]:
                    for pokemon_subtype in anchor_card[3]:
                        id_list.append(pokemon_pokedex_list[str(pokemon_number)][pokemon_type][pokemon_subtype])
            id_list = list(dict.fromkeys(sum(id_list, [])))
            if len(id_list) > 1:
                negative_item = random.choice(id_list)
                while anchor_item == negative_item:
                    negative_item = random.choice(id_list)
                return negative_item
        elif random_number < 0.7:
            id_list = []
            for pokemon_type in anchor_card[2]:
                for pokemon_subtype in anchor_card[3]:
                    id_list.append(pokemon_type_list[pokemon_type][pokemon_subtype])
            id_list = list(dict.fromkeys(sum(id_list, [])))
            if len(id_list) > 1:
                negative_item = random.choice(id_list)
                while anchor_item == negative_item:
                    negative_item = random.choice(id_list)
                return negative_item
        elif random_number < 0.8:
            if len(pokemon_list) > 1:
                negative_item = random.choice(pokemon_list)
                while anchor_item == negative_item:
                    negative_item = random.choice(pokemon_list)
                return negative_item
    elif anchor_card[0] == "Trainer":
        if random.random() < 0.8 and len(trainer_list) > 1:
            negative_item = random.choice(trainer_list)
            while anchor_item == negative_item:
                negative_item = random.choice(trainer_list)
            return negative_item
    elif anchor_card[0] == "Energy":
        if random.random() < 0.8 and len(energy_list) > 1:
            negative_item = random.choice(energy_list)
            while anchor_item == negative_item:
                negative_item = random.choice(energy_list)
            return negative_item

    negative_item = random.choice(anchor_list)
    while anchor_item == negative_item:
        negative_item = random.choice(anchor_list)
    return negative_item.replace(".jpg", "")


def select_random_negative_sample_strategy(anchor_item, anchor_list):
    negative_item = random.choice(anchor_list)
    while anchor_item == negative_item:
        negative_item = random.choice(anchor_list)
    return negative_item.replace(".jpg", "")


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_card_id_list():
    """The card-id metadata dict ``{card_id: [supertype, pokedex, types, subtypes]}``
    with the card back injected as a real card (it has no API metadata)."""
    d = _load_json(IDENTIFICATION_POKEMON_CARD_ID_DATABASE_FILE)
    d.setdefault(BACK_CARD_ID, ["Back", None, None, None])
    return d


def load_wrong_scans():
    """Set of card ids whose stored image is actually a card back (wrong scans),
    as written by ``installation/find_wrong_scans.py``.  Empty if the list is absent."""
    if not os.path.exists(WRONG_SCAN_LIST_FILE):
        return set()
    data = _load_json(WRONG_SCAN_LIST_FILE)
    return set(data.keys() if isinstance(data, dict) else data)


class Negative_selector:
    def __init__(self, strategy, anchor_list, fine_tune):
        self.strategy = strategy
        self.anchor_list = anchor_list
        self.fine_tune = fine_tune
        if strategy == select_random_negative_sample_strategy:
            self.card_id_list = self.pokemon_pokedex_list = self.pokemon_type_list = None
            self.pokemon_list = self.trainer_list = self.energy_list = None
            return

        original_card_id_list = load_card_id_list()
        self.pokemon_pokedex_list = _load_json(IDENTIFICATION_POKEMON_CARD_NUMBER_DATABASE_FILE)
        self.pokemon_type_list = _load_json(IDENTIFICATION_POKEMON_CARD_TYPE_DATABASE_FILE)
        self.pokemon_list = _load_json(IDENTIFICATION_POKEMON_CARD_DATABASE_FILE)
        self.trainer_list = _load_json(IDENTIFICATION_TRAINER_CARD_DATABASE_FILE)
        self.energy_list = _load_json(IDENTIFICATION_ENERGY_CARD_DATABASE_FILE)

        if fine_tune:
            self.card_id_list = {anchor.replace(".jpg", ""): original_card_id_list[anchor.replace(".jpg", "")]
                                 for anchor in anchor_list}
            for k1 in self.pokemon_pokedex_list:
                for k2 in self.pokemon_pokedex_list[k1]:
                    for k3 in self.pokemon_pokedex_list[k1][k2]:
                        self.pokemon_pokedex_list[k1][k2][k3] = [v for v in self.pokemon_pokedex_list[k1][k2][k3] if (v + ".jpg") in anchor_list]
            for k1 in self.pokemon_type_list:
                for k2 in self.pokemon_type_list[k1]:
                    self.pokemon_type_list[k1][k2] = [v for v in self.pokemon_type_list[k1][k2] if (v + ".jpg") in anchor_list]
            self.pokemon_list = [v for v in self.pokemon_list if (v + ".jpg") in anchor_list]
            self.trainer_list = [v for v in self.trainer_list if (v + ".jpg") in anchor_list]
            self.energy_list = [v for v in self.energy_list if (v + ".jpg") in anchor_list]
        else:
            self.card_id_list = original_card_id_list

    def select_negative(self, anchor_item):
        if self.strategy == select_random_negative_sample_strategy:
            return self.strategy(anchor_item, self.anchor_list)
        return self.strategy(anchor_item, self.anchor_list, self.card_id_list, self.pokemon_pokedex_list,
                             self.pokemon_type_list, self.pokemon_list, self.trainer_list, self.energy_list)


class Pokemon_card_train_dataset_triplet:
    def __init__(self, anchor_path, anchor_list, data_path, negative_selector, anchor_transform, data_transform):
        self.anchor_transform = anchor_transform
        self.data_transform = data_transform
        self.anchor_path = anchor_path
        self.anchor_list = anchor_list
        self.data_path = data_path
        self.negative_selector = negative_selector

    def __len__(self):
        return len(self.anchor_list)

    def __getitem__(self, item):
        anchor_img_file_name = self.anchor_list[item]
        anchor_label = anchor_img_file_name.replace(".jpg", "")
        anchor_img = self.anchor_transform(Image.open(self.anchor_path + anchor_img_file_name))
        positive_img = self.data_transform(Image.open(self.data_path + anchor_label + "-" + str(random.randint(0, POSITIVE_DATA_NUMBER - 2)) + ".png"))
        negative_img_id = self.negative_selector.select_negative(anchor_label)
        negative_img = self.data_transform(Image.open(self.data_path + negative_img_id + "-" + str(random.randint(0, POSITIVE_DATA_NUMBER - 2)) + ".png"))
        return anchor_img, positive_img, negative_img, anchor_label


class Pokemon_card_test_dataset_triplet:
    def __init__(self, anchor_path, anchor_list, data_path, transform):
        self.transform = transform
        self.anchor_path = anchor_path
        self.anchor_list = anchor_list
        self.data_path = data_path

    def __len__(self):
        return len(self.anchor_list)

    def __getitem__(self, item):
        anchor_img_file_name = self.anchor_list[item]
        anchor_label = anchor_img_file_name.replace(".jpg", "")
        anchor_img = self.transform(Image.open(self.anchor_path + anchor_img_file_name))
        positive_img = self.transform(Image.open(self.data_path + anchor_label + "-" + str(POSITIVE_DATA_NUMBER - 1) + ".png"))
        return anchor_img, positive_img, anchor_label


class Pokemon_card_inference_dataset_triplet:
    def __init__(self, anchor_path, anchor_list, data_path, transform):
        self.transform = transform
        self.anchor_path = anchor_path
        self.anchor_list = anchor_list
        self.data_path = data_path

    def __len__(self):
        return len(self.anchor_list)

    def __getitem__(self, item):
        anchor_img_file_name = self.anchor_list[item]
        anchor_label = anchor_img_file_name.replace(".jpg", "")
        return self.transform(Image.open(self.anchor_path + anchor_img_file_name)), anchor_label


class Pokemon_card_train_dataset_arcface:
    """Single-label dataset for ArcFace training (returns image + integer class)."""

    def __init__(self, anchor_list, data_path, data_transform):
        self.data_transform = data_transform
        self.data_path = data_path
        self.anchor_list = anchor_list
        self.class_to_idx = {a.replace(".jpg", ""): i for i, a in enumerate(anchor_list)}

    def __len__(self):
        return len(self.anchor_list) * (POSITIVE_DATA_NUMBER - 1)

    def __getitem__(self, item):
        card_idx = item // (POSITIVE_DATA_NUMBER - 1)
        aug_idx = item % (POSITIVE_DATA_NUMBER - 1)
        label = self.anchor_list[card_idx].replace(".jpg", "")
        img = self.data_transform(Image.open(self.data_path + label + "-" + str(aug_idx) + ".png"))
        return img, self.class_to_idx[label]


def all_card_anchor_list():
    """Every usable card-image filename: cards present in the id database, PLUS the
    card back (``back1-1``, a real card despite having no API metadata), MINUS any
    card flagged as a wrong scan (image is actually a back)."""
    id_db = _load_json(IDENTIFICATION_POKEMON_CARD_ID_DATABASE_FILE)
    banned = load_wrong_scans()
    files = os.listdir(POKEMON_CARD_DATABASE_FOLDER_PATH)
    cards = [f for f in files if f.endswith(".jpg") and f[:-4] in id_db and f[:-4] not in banned]
    back = BACK_CARD_ID + ".jpg"
    if back in files and BACK_CARD_ID not in banned:
        cards.append(back)
    return sorted(cards)


def get_datasets(anchor_list=None):
    if anchor_list:
        negative_selector = Negative_selector(select_targeted_negative_sample_strategy, anchor_list, True)
    else:
        anchor_list = all_card_anchor_list()
        negative_selector = Negative_selector(select_targeted_negative_sample_strategy, anchor_list, False)

    train_dataset = Pokemon_card_train_dataset_triplet(
        POKEMON_CARD_DATABASE_FOLDER_PATH, anchor_list, IDENTIFICATION_DATA_FOLDER_PATH, negative_selector,
        get_valid_transform(IDENTIFICATION_IMAGE_SIZE), get_identification_train_transform(IDENTIFICATION_IMAGE_SIZE))
    valid_dataset = Pokemon_card_test_dataset_triplet(
        POKEMON_CARD_DATABASE_FOLDER_PATH, anchor_list, IDENTIFICATION_DATA_FOLDER_PATH, get_valid_transform(IDENTIFICATION_IMAGE_SIZE))

    valid_size = int(IDENTIFICATION_VALID_SPLIT * len(anchor_list))
    indices = torch.randperm(len(valid_dataset)).tolist()
    valid_dataset = Subset(valid_dataset, indices[-valid_size:])
    return train_dataset, valid_dataset


def get_data_loaders(train_dataset, valid_dataset):
    train_loader = DataLoader(train_dataset, batch_size=IDENTIFICATION_BATCH_SIZE, shuffle=True,
                              num_workers=TRAIN_IDENTIFICATION_NUM_WORKERS, persistent_workers=True)
    valid_loader = DataLoader(valid_dataset, batch_size=IDENTIFICATION_BATCH_SIZE, shuffle=False,
                              num_workers=VALID_IDENTIFICATION_NUM_WORKERS, persistent_workers=True)
    return train_loader, valid_loader


# --------------------------------------------------------------------------- #
# Anchors + inference
# --------------------------------------------------------------------------- #
def _read_deck_anchor_list():
    try:
        with open(IDENTIFICATION_FINE_TUNE_DECK_LIST_FILE, 'r') as f:
            cards_list = f.readlines()
    except FileNotFoundError:
        raise
    anchor_list = [card.removesuffix("\n") + ".jpg" for card in cards_list if card.strip()]
    if len(anchor_list) == 0:
        # No deck -> every usable card (includes the back card, excludes wrong scans).
        anchor_list = all_card_anchor_list()
    return anchor_list


def evaluate_anchors(model, device, anchor_list=None, num_workers=None,
                     progress_callback=None, model_path=None):
    """Build {card_id -> embedding} for the deck (or every card if the deck is empty).

    Pass ``num_workers=0`` when calling from a non-main thread (e.g. a QThread) to avoid
    multiprocessing-spawn issues on Windows.

    ``progress_callback`` (optional): callable(int) invoked with values 0–100 as
    batches complete.  Useful for wiring to a QProgressBar.

    ``model_path`` (optional): path to the .pth weights file loaded into ``model``.
    When provided the result is persisted to EMBEDDING_CACHE_PATH and loaded from
    there on subsequent calls with the same model + anchor list (cache key = first
    8 hex digits of SHA-256 of each).
    """
    import hashlib as _hashlib

    if anchor_list is None:
        anchor_list = _read_deck_anchor_list()

    # ── cache check ──────────────────────────────────────────────────────────
    cache_file = None
    if model_path is not None:
        os.makedirs(EMBEDDING_CACHE_PATH, exist_ok=True)
        try:
            with open(model_path, 'rb') as _f:
                model_hash = _hashlib.sha256(_f.read()).hexdigest()[:8]
        except OSError:
            model_hash = "nohash"
        list_hash = _hashlib.sha256(",".join(anchor_list).encode()).hexdigest()[:8]
        cache_file = EMBEDDING_CACHE_PATH + f"{model_hash}_{list_hash}.pt"
        if os.path.exists(cache_file):
            try:
                result = torch.load(cache_file, map_location='cpu', weights_only=False)
                if progress_callback:
                    progress_callback(100)
                return result
            except Exception:
                pass  # corrupt cache — fall through to recompute

    # ── compute ──────────────────────────────────────────────────────────────
    anchor_dataset = Pokemon_card_inference_dataset_triplet(
        POKEMON_CARD_DATABASE_FOLDER_PATH, anchor_list, IDENTIFICATION_DATA_FOLDER_PATH, get_valid_transform(IDENTIFICATION_IMAGE_SIZE))
    # Parallel image loading: building anchors over all ~20k cards is image-IO bound;
    # single-threaded it takes minutes (workers help a lot, e.g. tool startup / eval).
    # Caller can override with num_workers=0 when running inside a non-main thread.
    if num_workers is None:
        num_workers = TRAIN_IDENTIFICATION_NUM_WORKERS if len(anchor_list) > 200 else 0
    anchor_loader = DataLoader(anchor_dataset, batch_size=64, num_workers=num_workers)

    anchors, anchors_label = [], []
    n_batches = max(len(anchor_loader), 1)
    model.eval()
    with torch.no_grad():
        for batch_idx, (anchor_img, anchor_label) in enumerate(anchor_loader):
            anchors_label += list(anchor_label)
            anchors.append(model(anchor_img.to(device)).detach().to("cpu"))
            if progress_callback:
                progress_callback(int((batch_idx + 1) / n_batches * 100))

    result = dict(zip(anchors_label, torch.cat(anchors)))

    # ── persist to disk ──────────────────────────────────────────────────────
    if cache_file is not None:
        try:
            torch.save(result, cache_file)
        except Exception as _e:
            import sys as _sys
            print(f"Warning: could not save embedding cache: {_e}", file=_sys.stderr)

    return result


def _pokemons_for(cards_id, card_id_list, truncate=False):
    pokemons_id = []
    for card_id in cards_id:
        numbers = card_id_list[card_id][1]
        if numbers:
            pokemons_id.append(numbers[0:MAX_NUMBER_OF_POKEMON_PER_CARD] if truncate else numbers)
        else:
            pokemons_id.append([None])
    return pokemons_id


def identify_cards(card_images, anchors_dict, model, device, card_id_list, criterion=None, method=None):
    """Identify a batch of card crops against the anchor embeddings.

    ``card_images`` is a pre-transformed tensor (B, 3, H, W).  Returns
    ``(cards_id: np.ndarray[str], pokemons_id: list)``.
    """
    method = _resolve_method(method)
    model.eval()
    card_images = card_images.to(device)
    with torch.no_grad():
        query = model(card_images).detach().to("cpu")

    anchor_labels = list(anchors_dict.keys())
    anchor_matrix = torch.stack(list(anchors_dict.values()))  # (N, D)

    if method == "arcface":
        query = F.normalize(query, p=2, dim=1)
        anchor_matrix = F.normalize(anchor_matrix, p=2, dim=1)
        similarity = torch.mm(query, anchor_matrix.t())        # (B, N)
        best = torch.max(similarity, dim=1).indices.numpy()
    else:
        # Vectorised nearest-anchor: argmin of Euclidean distance (the additive
        # TripletMarginLoss margin does not affect the arg-min).
        distances = torch.cdist(query, anchor_matrix, p=2)     # (B, N)
        best = torch.min(distances, dim=1).indices.numpy()

    cards_id = np.array([anchor_labels[i] for i in best])
    return cards_id, _pokemons_for(cards_id, card_id_list)


# Cache of anchor card-image BGR (mean, std) so we read each card from disk only once.
_anchor_color_stats = {}


def _anchor_color_stat(card_id):
    stat = _anchor_color_stats.get(card_id)
    if stat is None:
        anchor = cv2.imread(POKEMON_CARD_DATABASE_FOLDER_PATH + card_id + ".jpg")
        b, g, r = cv2.split(anchor)
        stat = ((np.mean(b), np.mean(g), np.mean(r)), (np.std(b), np.std(g), np.std(r)))
        _anchor_color_stats[card_id] = stat
    return stat


def identify_cards_with_color_correction(card_images, anchors_dict, model_input_transform, model, device,
                                        card_id_list, criterion=None, method=None):
    """Colour-corrected variant: recolour each query to every anchor's colour
    distribution before matching.  ``card_images`` is a list of raw BGR crops."""
    method = _resolve_method(method)
    model.eval()
    nb = len(card_images)
    cards_id = np.full((nb,), next(iter(anchors_dict)))
    if method == "arcface":
        best_score = torch.full((nb,), -np.inf)
    else:
        best_score = torch.full((nb,), np.inf)

    for key in anchors_dict:
        means, stds = _anchor_color_stat(key)
        recolor = torch.tensor([])
        for card_image in card_images:
            rc = transfer_rgb_distribution(card_image, means, stds)
            rc = cv2.cvtColor(rc, cv2.COLOR_BGR2RGB)
            rc = torch.unsqueeze(model_input_transform(rc), 0)
            recolor = torch.cat((recolor, rc))
        with torch.no_grad():
            output = model(recolor.to(device)).detach().to("cpu")
        anchor = anchors_dict[key]

        if method == "arcface":
            score = torch.sum(F.normalize(output, dim=1) * F.normalize(anchor.unsqueeze(0), dim=1), dim=1)
            cards_id = np.where(best_score.numpy() > score.numpy(), cards_id, np.full((nb,), key))
            best_score = torch.maximum(best_score, score)
        else:
            anchors = torch.unsqueeze(anchor, 0).repeat(nb, 1)
            score = torch.linalg.norm(anchors - output, dim=1)
            cards_id = np.where(best_score.numpy() < score.numpy(), cards_id, np.full((nb,), key))
            best_score = torch.minimum(best_score, score)

    return cards_id, _pokemons_for(cards_id, card_id_list, truncate=True)


# --------------------------------------------------------------------------- #
# Top-k proposals + orientation pick (used by the real-set annotation tool)
# --------------------------------------------------------------------------- #
def identify_topk(card_images, anchors_dict, model, device, k=10, method=None):
    """Return, per query crop, the top-k ``[(card_id, score), ...]`` ranked best->worst
    (cosine similarity for arcface, Euclidean distance for triplet)."""
    method = _resolve_method(method)
    model.eval()
    with torch.no_grad():
        query = model(card_images.to(device)).detach()
    anchor_labels = list(anchors_dict.keys())
    anchor_matrix = torch.stack(list(anchors_dict.values())).to(device)
    k = min(k, len(anchor_labels))

    if method == "arcface":
        scores = torch.mm(F.normalize(query, p=2, dim=1), F.normalize(anchor_matrix, p=2, dim=1).t())
        top = scores.topk(k, dim=1, largest=True)
    else:
        dists = torch.cdist(query, anchor_matrix)
        top = dists.topk(k, dim=1, largest=False)
    vals = top.values.cpu().numpy()
    idx = top.indices.cpu().numpy()
    return [[(anchor_labels[idx[b][j]], float(vals[b][j])) for j in range(k)] for b in range(idx.shape[0])]


def best_orientation_topk(crop_bgr, anchors_dict, model, device, transform, k=10, method=None):
    """Score a BGR crop both upright and rotated 180 degrees; return
    ``(flip, proposals)`` where ``flip`` (0/1) is the orientation that matches best
    and ``proposals`` is its top-k ``[(card_id, score), ...]``."""
    import cv2
    method = _resolve_method(method)
    oriented = [crop_bgr, cv2.rotate(crop_bgr, cv2.ROTATE_180)]
    batch = torch.stack([transform(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for c in oriented])
    topk = identify_topk(batch, anchors_dict, model, device, k=k, method=method)
    s0, s1 = topk[0][0][1], topk[1][0][1]
    if method == "arcface":
        flip = 0 if s0 >= s1 else 1     # higher cosine is better
    else:
        flip = 0 if s0 <= s1 else 1     # lower distance is better
    return flip, topk[flip]

"""
Evaluate the identification model (top-1 accuracy) for either method.

Synthetic protocol: build the anchor embeddings from the (clean) card images,
then query with each card's held-out augmented sample
(``<card_id>-<POSITIVE_DATA_NUMBER-1>.png``) and check the nearest anchor is the
right card.  Because it reuses ``core.models.identification.identify_cards`` it
runs unchanged for both ``triplet`` and ``arcface`` -- so the two methods can be
compared directly.

Real protocol: point ``--real-dir`` at a folder of cropped real card images named
``<card_id>.png`` (or ``<card_id>-<n>.png``) to score on real data.
"""

import os
import argparse
from glob import glob

import cv2
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from core import config
from core.config import (IDENTIFICATION_DATA_FOLDER_PATH, IDENTIFICATION_IMAGE_SIZE, POSITIVE_DATA_NUMBER,
                         POKEMON_CARD_DATABASE_FOLDER_PATH, IDENTIFICATION_POKEMON_CARD_ID_DATABASE_FILE)
from core.models import identification as idm
from core.transforms import get_valid_transform, get_inference_transform


def _device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _load_card_id_list():
    return idm.load_card_id_list()  # includes the back card entry


def _batched(items, n):
    for i in range(0, len(items), n):
        yield items[i:i + n]


def evaluate_synthetic(method=None, max_cards=None, batch_size=40, fine_tuned=True, full=False, top_k=5):
    """Top-1 and top-k identification accuracy on the held-out synthetic samples.

    The anchor matrix is built once; each query batch is scored against it in a
    single op (cosine similarity for arcface, Euclidean distance for triplet)."""
    method = (method or config.IDENTIFICATION_METHOD).lower()
    device = _device()
    model = idm.load_model(method, fine_tuned=fine_tuned, device=device)
    card_id_list = _load_card_id_list()

    if full:
        # Every card image that has a metadata entry (drops the manually-added
        # back1-1 card back, which has no entry in the id database).
        all_imgs = os.listdir(POKEMON_CARD_DATABASE_FOLDER_PATH)
        anchor_list = sorted(f for f in all_imgs if f.endswith(".jpg") and f[:-4] in card_id_list)
    else:
        anchor_list = idm._read_deck_anchor_list()
    if max_cards:
        anchor_list = anchor_list[:max_cards]

    print(f"[{method}] {'base' if not fine_tuned else 'fine-tuned'} model, "
          f"building anchors for {len(anchor_list)} cards...")
    anchors_dict = idm.evaluate_anchors(model, device, anchor_list=anchor_list)
    anchor_labels = list(anchors_dict.keys())
    anchor_matrix = torch.stack(list(anchors_dict.values())).to(device)
    if method == "arcface":
        anchor_matrix = torch.nn.functional.normalize(anchor_matrix, p=2, dim=1)
    k = min(top_k, len(anchor_labels))

    transform = get_valid_transform(IDENTIFICATION_IMAGE_SIZE)
    correct1 = correctk = total = 0
    labels = [a.replace(".jpg", "") for a in anchor_list]

    model.eval()
    for batch in tqdm(list(_batched(labels, batch_size))):
        tensors, truths = [], []
        for card_id in batch:
            test_path = IDENTIFICATION_DATA_FOLDER_PATH + card_id + "-" + str(POSITIVE_DATA_NUMBER - 1) + ".png"
            if not os.path.exists(test_path):
                continue
            tensors.append(transform(Image.open(test_path)))
            truths.append(card_id)
        if not tensors:
            continue
        with torch.no_grad():
            query = model(torch.stack(tensors).to(device))
        if method == "arcface":
            query = torch.nn.functional.normalize(query, p=2, dim=1)
            scores = torch.mm(query, anchor_matrix.t())            # higher = better
            topk = scores.topk(k, dim=1, largest=True).indices
        else:
            distances = torch.cdist(query, anchor_matrix)          # lower = better
            topk = distances.topk(k, dim=1, largest=False).indices
        topk = topk.cpu().numpy()
        for i, truth in enumerate(truths):
            preds = [anchor_labels[j] for j in topk[i]]
            total += 1
            correct1 += (preds[0] == truth)
            correctk += (truth in preds)

    acc1 = 100.0 * correct1 / total if total else 0.0
    acck = 100.0 * correctk / total if total else 0.0
    print(f"[{method}] full={full} base={not fine_tuned}: "
          f"top-1 {acc1:.2f}%  top-{k} {acck:.2f}%  ({total} cards)")
    return acc1, acck


def evaluate_real(real_dir, method=None, batch_size=40):
    method = (method or config.IDENTIFICATION_METHOD).lower()
    device = _device()
    model = idm.load_model(method, fine_tuned=True, device=device)
    anchors_dict = idm.evaluate_anchors(model, device)
    card_id_list = _load_card_id_list()
    transform = get_inference_transform(IDENTIFICATION_IMAGE_SIZE)

    paths = sorted(glob(os.path.join(real_dir, "*.png")) + glob(os.path.join(real_dir, "*.jpg")))
    correct = 0
    total = 0
    for batch in tqdm(list(_batched(paths, batch_size))):
        tensors, truths = [], []
        for p in batch:
            truth = os.path.splitext(os.path.basename(p))[0].split("-")[0]
            img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
            tensors.append(transform(img))
            truths.append(truth)
        preds, _ = idm.identify_cards(torch.stack(tensors), anchors_dict, model, device, card_id_list, method=method)
        correct += int(np.sum(np.array(preds) == np.array(truths)))
        total += len(truths)

    acc = 100.0 * correct / total if total else 0.0
    print(f"[{method}] real identification top-1 accuracy: {acc:.2f}% ({correct}/{total})")
    return acc


def evaluate_real_set(method=None, fine_tuned=False, batch_size=40, top_k=5, deck_restricted=False):
    """Top-1 / top-k identification on the annotated real set.

    Uses the ground-truth-upright crops (orientation taken from the annotation, so
    identification is measured independently of the orientation model). By default the
    crops are matched against anchors over every card (open set, ``K``); with
    ``deck_restricted`` the candidate set is reduced to the cards actually present in
    the real data, mirroring the deck-list restriction used in deployment."""
    import core.real_data as rd
    method = (method or config.IDENTIFICATION_METHOD).lower()
    device = _device()
    model = idm.load_model(method, fine_tuned=fine_tuned, device=device)
    transform = get_inference_transform(IDENTIFICATION_IMAGE_SIZE)

    crops = [(crop, gt) for _i, _poly, crop, gt, _o in rd.iter_real_crops(upright=True)]

    if deck_restricted:
        # Candidate set = the cards present in the real data (their ground-truth ids).
        present = sorted({gt for _c, gt in crops})
        anchor_list = [cid + ".jpg" for cid in present]
        anchors_dict = idm.evaluate_anchors(model, device, anchor_list=anchor_list)
        print(f"[{method}] real set (deck-restricted to {len(present)} present cards): "
              f"{len(crops)} annotated cards")
    else:
        anchors_dict = idm.evaluate_anchors(model, device, anchor_list=idm.all_card_anchor_list())
        print(f"[{method}] real set (open set, all cards): {len(crops)} annotated cards (excluding 'unknown')")
    correct1 = correctk = total = 0
    for batch in tqdm(list(_batched(crops, batch_size))):
        tensors = [transform(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for c, _ in batch]
        truths = [gt for _, gt in batch]
        proposals = idm.identify_topk(torch.stack(tensors), anchors_dict, model, device, k=top_k, method=method)
        for i, truth in enumerate(truths):
            preds = [cid for cid, _score in proposals[i]]
            total += 1
            correct1 += (preds[0] == truth)
            correctk += (truth in preds)

    acc1 = 100.0 * correct1 / total if total else 0.0
    acck = 100.0 * correctk / total if total else 0.0
    print(f"[{method}] REAL set base={not fine_tuned}: top-1 {acc1:.2f}%  top-{top_k} {acck:.2f}%  ({total} cards)")
    return acc1, acck


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the identification model")
    parser.add_argument("--method", choices=["triplet", "arcface"], default=None)
    parser.add_argument("--dataset", choices=["synthetic", "real"], default="synthetic")
    parser.add_argument("--max-cards", type=int, default=None, help="limit number of cards (quick smoke run)")
    parser.add_argument("--full", action="store_true", help="evaluate over every card, not just the deck")
    parser.add_argument("--base", action="store_true", help="use the base model instead of the fine-tuned one")
    parser.add_argument("--real-dir", default=None, help="folder of labelled real crops <card_id>.png")
    parser.add_argument("--deck-restricted", action="store_true",
                        help="real set: restrict candidates to the cards present in the real data")
    args = parser.parse_args()
    if args.real_dir:
        evaluate_real(args.real_dir, args.method)
    elif args.dataset == "real":
        evaluate_real_set(args.method, fine_tuned=not args.base, deck_restricted=args.deck_restricted)
    else:
        evaluate_synthetic(args.method, args.max_cards, fine_tuned=not args.base, full=args.full)

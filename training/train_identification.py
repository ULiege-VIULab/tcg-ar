"""
Train / fine-tune the card identification model.

Supports BOTH identification methods, selected by ``config.IDENTIFICATION_METHOD``
(or ``--method``):

* ``triplet`` -- ResNet-50 metric learning with ``TripletMarginLoss``.
* ``arcface`` -- ResNet-50 embedding + angular-margin softmax head.

All model definitions, datasets and loops are reused from ``core`` -- nothing is
duplicated here.
"""

import os
import sys
import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader

from core import config
from core.config import (IDENTIFICATION_LR, IDENTIFICATION_EPOCHS, IDENTIFICATION_FINE_TUNE_LR,
                         IDENTIFICATION_FINE_TUNE_EPOCHS, IDENTIFICATION_FINE_TUNE_BATCH_SIZE,
                         IDENTIFICATION_BATCH_SIZE, IDENTIFICATION_DATA_FOLDER_PATH,
                         IDENTIFICATION_IMAGE_SIZE, IDENTIFICATION_FINE_TUNE_DECK_LIST_FILE,
                         POKEMON_CARD_DATABASE_FOLDER_PATH, ARCFACE_EMBEDDING_SIZE,
                         TRAIN_IDENTIFICATION_NUM_WORKERS, model_save_path)
from core.models import identification as idm
from core.transforms import get_identification_train_transform
from core.training_utils import (train_triplet_one_epoch, validate_triplet, save_model, save_triplet_plots)


def _device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def _read_deck_anchor_list():
    with open(IDENTIFICATION_FINE_TUNE_DECK_LIST_FILE, 'r') as f:
        return [card.removesuffix("\n") + ".jpg" for card in f.readlines()]


# --------------------------------------------------------------------------- #
# Triplet
# --------------------------------------------------------------------------- #
def train_triplet():
    train_ds, valid_ds = idm.get_datasets()
    train_loader, valid_loader = idm.get_data_loaders(train_ds, valid_ds)
    device = _device()
    model = idm.build_triplet_model(device)
    optimizer = optim.Adam(model.parameters(), lr=IDENTIFICATION_LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 4, 1 / 3)
    criterion = nn.TripletMarginLoss(1.0, 2.0, reduction="mean")
    valid_criterion = nn.TripletMarginLoss(1.0, 2.0, reduction='none')

    train_loss, train_distance, validation_loss = [], [], []
    for epoch in range(IDENTIFICATION_EPOCHS):
        print(f"[INFO]: Epoch {epoch+1} of {IDENTIFICATION_EPOCHS}")
        tl, td = train_triplet_one_epoch(model, train_loader, optimizer, scheduler, criterion, device)
        vl, _ = validate_triplet(model, valid_loader, valid_criterion, device, False)
        train_loss.append(tl); train_distance.append(td); validation_loss.append(vl)
        print(f"Training loss: {tl:.3f}, distance: {td:.3f} | Validation distance: {vl:.3f}")
        save_model(epoch + 1, model, optimizer, criterion, model_save_path("triplet", fine_tuned=False))
        time.sleep(5)

    save_model(IDENTIFICATION_EPOCHS, model, optimizer, criterion, model_save_path("triplet", fine_tuned=False))
    save_triplet_plots(train_distance, train_loss, validation_loss, "identification")
    vl, va = validate_triplet(model, valid_loader, valid_criterion, device, True)
    print(f"Final validation distance: {vl:.3f}, accuracy: {va:.3f}")


def fine_tune_triplet():
    anchor_list = _read_deck_anchor_list()
    train_ds, valid_ds = idm.get_datasets(anchor_list)
    device = _device()
    train_loader = DataLoader(train_ds, batch_size=IDENTIFICATION_FINE_TUNE_BATCH_SIZE, shuffle=True, num_workers=1, persistent_workers=True)
    valid_loader = DataLoader(valid_ds, batch_size=IDENTIFICATION_FINE_TUNE_BATCH_SIZE, shuffle=False, num_workers=1, persistent_workers=True)

    model = idm.load_model("triplet", fine_tuned=False, device=device, eval_mode=False)
    optimizer = optim.Adam(model.parameters(), lr=IDENTIFICATION_FINE_TUNE_LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 4, 1 / 3)
    criterion = nn.TripletMarginLoss(1.0, 2.0, reduction="mean")
    valid_criterion = nn.TripletMarginLoss(1.0, 2.0, reduction='none')

    train_loss, train_distance, validation_loss = [], [], []
    for epoch in range(IDENTIFICATION_FINE_TUNE_EPOCHS):
        print(f"[INFO]: Epoch {epoch+1} of {IDENTIFICATION_FINE_TUNE_EPOCHS}")
        tl, td = train_triplet_one_epoch(model, train_loader, optimizer, scheduler, criterion, device)
        vl, _ = validate_triplet(model, valid_loader, valid_criterion, device, False)
        train_loss.append(tl); train_distance.append(td); validation_loss.append(vl)
        save_model(epoch + 1, model, optimizer, criterion, model_save_path("triplet", fine_tuned=True))

    save_model(IDENTIFICATION_FINE_TUNE_EPOCHS, model, optimizer, criterion, model_save_path("triplet", fine_tuned=True))
    save_triplet_plots(train_distance, train_loss, validation_loss, "identification_fine_tuning")
    vl, va = validate_triplet(model, valid_loader, valid_criterion, device, True, True)
    print(f"Final validation distance: {vl:.3f}, accuracy: {va:.3f}")


def _fine_tune_triplet_only(progress=None):
    """Headless fine-tune used by the GUI: no validation, emits progress 0..100."""
    try:
        anchor_list = _read_deck_anchor_list()
    except FileNotFoundError:
        print("the deck file does not exist", file=sys.stderr)
        raise
    device = _device()
    model = idm.load_model("triplet", fine_tuned=False, device=device, eval_mode=False)

    # Empty deck -> just persist the base model under the fine-tune path.
    if len(anchor_list) == 0:
        torch.save({'model_state_dict': model.state_dict()}, model_save_path("triplet", fine_tuned=True))
        return

    train_ds, _ = idm.get_datasets(anchor_list)
    train_loader = DataLoader(train_ds, batch_size=IDENTIFICATION_FINE_TUNE_BATCH_SIZE, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=IDENTIFICATION_FINE_TUNE_LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 4, 1 / 3)
    criterion = nn.TripletMarginLoss(1.0, 2.0, reduction="mean")

    for i in range(IDENTIFICATION_FINE_TUNE_EPOCHS):
        train_triplet_one_epoch(model, train_loader, optimizer, scheduler, criterion, device, verbose=False)
        if progress:
            progress.emit(int(i / IDENTIFICATION_FINE_TUNE_EPOCHS * 100))
    save_model(IDENTIFICATION_FINE_TUNE_EPOCHS, model, optimizer, criterion, model_save_path("triplet", fine_tuned=True))


# --------------------------------------------------------------------------- #
# ArcFace
# --------------------------------------------------------------------------- #
def _train_arcface_one_epoch(backbone, head, loader, optimizer, scheduler, criterion, device):
    backbone.train(); head.train()
    running_loss = 0.0
    running_correct = 0
    counter = 0
    for image, label in tqdm(loader, total=len(loader)):
        counter += 1
        image = image.to(device); label = label.to(device)
        optimizer.zero_grad()
        embedding = backbone(image)
        logits = head(embedding, label)
        loss = criterion(logits, label)
        running_loss += loss.item()
        running_correct += (torch.max(logits.data, 1).indices == label).sum().item()
        loss.backward()
        optimizer.step()
    scheduler.step()
    return running_loss / counter, 100. * running_correct / len(loader.dataset)


def _arcface_train(anchor_list, epochs, lr, save_path, base_weights=None):
    device = _device()
    dataset = idm.Pokemon_card_train_dataset_arcface(anchor_list, IDENTIFICATION_DATA_FOLDER_PATH,
                                                     get_identification_train_transform(IDENTIFICATION_IMAGE_SIZE))
    loader = DataLoader(dataset, batch_size=IDENTIFICATION_BATCH_SIZE, shuffle=True,
                        num_workers=TRAIN_IDENTIFICATION_NUM_WORKERS, persistent_workers=True)

    backbone = idm.build_arcface_backbone(device)
    if base_weights is not None:
        backbone.load_state_dict(torch.load(base_weights, map_location=device, weights_only=False)['model_state_dict'])
    head = idm.ArcFaceLayer(ARCFACE_EMBEDDING_SIZE, out_features=len(anchor_list)).to(device)

    # Recipe matching the original working ArcFace model: SGD + momentum + weight
    # decay + cosine annealing.  (Adam + StepLR(/3 every 4 epochs) collapses the LR to
    # ~0 by epoch 12, so the 20k-class head never converges -> loss stuck ~16, acc 0.)
    optimizer = optim.SGD([
        {'params': backbone.parameters()},
        {'params': head.parameters()},
    ], lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        print(f"[INFO]: Epoch {epoch+1} of {epochs}")
        loss, acc = _train_arcface_one_epoch(backbone, head, loader, optimizer, scheduler, criterion, device)
        print(f"Training loss: {loss:.3f}, accuracy: {acc:.3f}")
        # Checkpoint every epoch: ArcFace converges fast, and this means an
        # interruption (e.g. session teardown) loses at most one epoch, not the run.
        save_model(epoch + 1, backbone, optimizer, criterion, save_path)
    print('ARCFACE TRAINING COMPLETE')


def train_arcface():
    # Only cards with a metadata entry (excludes the manually-added card back).
    anchor_list = idm.all_card_anchor_list()
    _arcface_train(anchor_list, IDENTIFICATION_EPOCHS, IDENTIFICATION_LR, model_save_path("arcface", fine_tuned=False))


def fine_tune_arcface():
    anchor_list = _read_deck_anchor_list() or idm.all_card_anchor_list()
    _arcface_train(anchor_list, IDENTIFICATION_FINE_TUNE_EPOCHS, IDENTIFICATION_FINE_TUNE_LR,
                   model_save_path("arcface", fine_tuned=True), base_weights=model_save_path("arcface", fine_tuned=False))


# --------------------------------------------------------------------------- #
# Method dispatch
# --------------------------------------------------------------------------- #
def train_identification(method=None):
    method = (method or config.IDENTIFICATION_METHOD).lower()
    (train_arcface if method == "arcface" else train_triplet)()


def fine_tune_identification(method=None):
    method = (method or config.IDENTIFICATION_METHOD).lower()
    (fine_tune_arcface if method == "arcface" else fine_tune_triplet)()


def fine_tune_model_only_train(progress=None, method=None):
    """Used by the GUI.  Triplet is headless (no validation); ArcFace reuses its fine-tune."""
    method = (method or config.IDENTIFICATION_METHOD).lower()
    if method == "arcface":
        fine_tune_arcface()
        if progress:
            progress.emit(100)
    else:
        _fine_tune_triplet_only(progress)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train / fine-tune the identification model")
    parser.add_argument("--method", choices=["triplet", "arcface"], default=None)
    parser.add_argument("--task", choices=["train", "fine_tune"], default="train")
    args = parser.parse_args()
    if args.task == "train":
        train_identification(args.method)
    else:
        fine_tune_identification(args.method)

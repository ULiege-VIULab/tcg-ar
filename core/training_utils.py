"""
Training / evaluation helpers shared by every model module.

Contains:
* checkpoint + plot saving,
* the generic single-label classifier loops (used by the orientation model),
* the metric-learning triplet loops (used by the identification model).
"""

import os

import torch
from tqdm import tqdm
from matplotlib import pyplot as plt

from core.config import OUTPUT_FOLDER_PATH


def save_model(epochs, model, optimizer, criterion, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': criterion,
    }, path)


def save_plots(train_acc, valid_acc, train_loss, valid_loss, folder_path, filename):
    """Accuracy + loss plots (classifier models)."""
    os.makedirs(folder_path, exist_ok=True)
    plt.figure(figsize=(10, 7))
    plt.plot(train_acc, color='green', linestyle='-', label='train accuracy')
    plt.plot(valid_acc, color='blue', linestyle='-', label='validation accuracy')
    plt.xlabel('Epochs'); plt.ylabel('Accuracy'); plt.legend()
    plt.savefig(os.path.join(folder_path, "accuracy_pretrained_" + filename + ".png"))

    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, color='orange', linestyle='-', label='train loss')
    plt.plot(valid_loss, color='red', linestyle='-', label='validation loss')
    plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.legend()
    plt.savefig(os.path.join(folder_path, "loss_pretrained_" + filename + ".png"))


def save_triplet_plots(train_distance, train_loss, validation_loss, name, folder_path=OUTPUT_FOLDER_PATH):
    """Distance + loss plots (metric-learning identification model)."""
    os.makedirs(folder_path, exist_ok=True)
    plt.figure(figsize=(10, 7))
    plt.plot(train_distance, color='red', linestyle='-', label='train distance')
    plt.plot(validation_loss, color='orange', linestyle='-', label='valid distance')
    plt.xlabel('Epochs'); plt.ylabel('Distance'); plt.legend()
    plt.savefig(os.path.join(folder_path, "distance_pretrained_" + name + ".png"))

    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, color='red', linestyle='-', label='train loss')
    plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.legend()
    plt.savefig(os.path.join(folder_path, "loss_pretrained_" + name + ".png"))


# --------------------------------------------------------------------------- #
# Generic single-label classifier loops (orientation, arcface head)
# --------------------------------------------------------------------------- #
def train_classifier_one_epoch(model, train_loader, optimizer, scheduler, criterion, device):
    model.train()
    print('Training')
    running_loss = 0.0
    running_correct = 0
    counter = 0
    for data in tqdm(train_loader, total=len(train_loader)):
        counter += 1
        image, labels = data
        image = image.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(image)
        loss = criterion(outputs, labels)
        running_loss += loss.item()
        _, preds = torch.max(outputs.data, 1)
        running_correct += (preds == labels).sum().item()
        loss.backward()
        optimizer.step()

    before_lr = optimizer.param_groups[0]["lr"]
    scheduler.step()
    after_lr = optimizer.param_groups[0]["lr"]
    if before_lr != after_lr:
        print("Lr decay: %f -> %f" % (before_lr, after_lr))

    epoch_loss = running_loss / counter
    epoch_acc = 100. * (running_correct / len(train_loader.dataset))
    return epoch_loss, epoch_acc


def evaluate_classifier(model, test_loader, criterion, device):
    model.eval()
    print('Validation')
    running_loss = 0.0
    running_correct = 0
    counter = 0
    with torch.no_grad():
        for data in tqdm(test_loader, total=len(test_loader)):
            counter += 1
            image, labels = data
            image = image.to(device)
            labels = labels.to(device)
            outputs = model(image)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, preds = torch.max(outputs.data, 1)
            running_correct += (preds == labels).sum().item()
    epoch_loss = running_loss / counter
    epoch_acc = 100. * (running_correct / len(test_loader.dataset))
    return epoch_loss, epoch_acc


# --------------------------------------------------------------------------- #
# Triplet metric-learning loops (identification, triplet method)
# --------------------------------------------------------------------------- #
def train_triplet_one_epoch(model, train_loader, optimizer, scheduler, criterion, device, verbose=True):
    model.train()
    iterable = tqdm(train_loader) if verbose else train_loader
    if verbose:
        print('Training')

    running_loss = 0.0
    running_distance = 0.0
    counter = 0
    for data in iterable:
        counter += 1
        anchor_img, positive_img, negative_img, _ = data
        anchor_img = anchor_img.to(device)
        positive_img = positive_img.to(device)
        negative_img = negative_img.to(device)
        optimizer.zero_grad()
        anchor_output = model(anchor_img)
        positive_output = model(positive_img)
        negative_output = model(negative_img)
        loss = criterion(anchor_output, positive_output, negative_output)
        # Accumulate Python floats, NOT the loss tensors: keeping the tensors would
        # retain every iteration's autograd graph for the whole epoch (huge memory +
        # allocator thrash -> ~30x slowdown).  Reported averages are identical.
        running_loss += loss.item()
        with torch.no_grad():
            running_distance += criterion(anchor_output, positive_output, anchor_output).item()
        loss.backward()
        optimizer.step()

    before_lr = optimizer.param_groups[0]["lr"]
    scheduler.step()
    after_lr = optimizer.param_groups[0]["lr"]
    if verbose and before_lr != after_lr:
        print("Lr decay: %f -> %f" % (before_lr, after_lr))

    return running_loss / counter, running_distance / counter


def validate_triplet(model, test_loader, criterion, device, final, log_fail=False, chunk=512):
    """Validation distance, plus (when ``final``) top-1 accuracy.

    The accuracy is "is each query's own anchor the nearest of all anchors".  The
    original computed this with a Python double loop over (query x anchor) calling
    the criterion for every pair -- O(N^2) ~ 400M criterion calls for 20k cards, which
    effectively hangs.  Here it is one vectorised ``torch.cdist`` per query-chunk
    (margin cancels in the comparison, so plain Euclidean distance is used); the
    result is identical.
    """
    model.eval()
    print('Validation')
    valid_running_loss = 0.0
    anchors, positives, labels = [], [], []
    with torch.no_grad():
        for data in tqdm(test_loader):
            anchor_img, positive_img, anchor_label = data
            anchor_outputs = model(anchor_img.to(device))
            positive_outputs = model(positive_img.to(device))
            losses = criterion(anchor_outputs, positive_outputs, anchor_outputs)
            valid_running_loss += losses.sum().item()
            if final:
                anchors.append(anchor_outputs)
                positives.append(positive_outputs)
                labels += list(anchor_label)

    epoch_loss = valid_running_loss / len(test_loader.dataset)
    epoch_acc = -1
    if final:
        A = torch.cat(anchors)        # (N, D) one anchor embedding per query
        P = torch.cat(positives)      # (N, D) query embeddings
        n = A.shape[0]
        correct = 0
        with torch.no_grad():
            for s in range(0, n, chunk):
                e = min(s + chunk, n)
                dists = torch.cdist(P[s:e], A)                          # (b, N)
                own = dists[torch.arange(e - s), torch.arange(s, e)]    # own-anchor distance
                best, best_idx = dists.min(dim=1)
                # Correct iff no other anchor is strictly closer than the own anchor.
                ok = own <= best + 1e-6
                correct += int(ok.sum())
                if log_fail:
                    for k in range(e - s):
                        if not bool(ok[k]):
                            print("Card: " + labels[s + k] + " is wrongly recognized as: " + labels[int(best_idx[k])])
        epoch_acc = 100 * (correct / n)
    return epoch_loss, epoch_acc

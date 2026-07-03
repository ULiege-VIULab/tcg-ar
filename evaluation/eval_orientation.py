"""
Evaluate the orientation classifier on the synthetic test set, or on a balanced
"dummy" set built from the real cards (each ground-truth-upright real crop is kept
'straight' or rotated 180 degrees as 'flip').
"""

import argparse

import torch
from torchvision import datasets
from torch.utils.data import DataLoader

from core.models.orientation import OrientationModel
from core.config import (REAL_ORIENTATION_SET_FOLDER, ORIENTATION_IMAGE_SIZE,
                         ORIENTATION_BATCH_SIZE, ORIENTATION_NUM_WORKERS)
from core.transforms import get_valid_transform
from core.training_utils import evaluate_classifier
import core.real_data as rd


def evaluate_synthetic(arch="efficientnet_b0"):
    return OrientationModel().evaluate(arch=arch)


def evaluate_real(arch="efficientnet_b0"):
    n_s, n_f = rd.build_real_orientation_dataset()
    print(f"Built real orientation set: {n_s} straight, {n_f} flip")
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    model = OrientationModel()
    model.load_orientation_model(device, arch=arch)
    dataset = datasets.ImageFolder(REAL_ORIENTATION_SET_FOLDER, transform=get_valid_transform(ORIENTATION_IMAGE_SIZE))
    loader = DataLoader(dataset, batch_size=ORIENTATION_BATCH_SIZE, shuffle=False, num_workers=ORIENTATION_NUM_WORKERS)
    criterion = torch.nn.CrossEntropyLoss()
    loss, acc = evaluate_classifier(model.model, loader, criterion, device)
    print(f"[real] orientation test loss: {loss:.3f}, accuracy: {acc:.3f}")
    return loss, acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the orientation model")
    parser.add_argument("--dataset", choices=["synthetic", "real"], default="synthetic")
    parser.add_argument("--arch", default="efficientnet_b0")
    args = parser.parse_args()
    evaluate_real(args.arch) if args.dataset == "real" else evaluate_synthetic(args.arch)

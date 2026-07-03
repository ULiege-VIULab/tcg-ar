"""
Torchvision transform pipelines shared by training, evaluation and inference.
Single copy of what used to live in every model module.
"""

import numpy as np
from PIL import Image
from torchvision import transforms

from core.image_ops import data_augmentation


class DataAugmentation:
    """Wraps the NumPy-based ``data_augmentation(image)`` so it can sit in a
    ``torchvision.transforms.Compose`` pipeline (PIL in -> PIL out)."""

    def __call__(self, img: Image.Image) -> Image.Image:
        arr = np.array(img)
        arr = data_augmentation(arr)
        return Image.fromarray(arr)


def get_train_transform(image_size):
    """Train transform for the classifier-style models (orientation)."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        DataAugmentation(),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_identification_train_transform(image_size):
    """Train transform used by the triplet identification model (root behaviour)."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_valid_transform(image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_inference_transform(image_size):
    """Inference transform: takes a raw numpy/cv2 (RGB) image -> normalized tensor."""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

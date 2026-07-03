"""Train an orientation classifier. ``--arch`` selects the backbone (default
efficientnet_b0, the deployed one); alternatives are saved to their own weight
paths and never overwrite the deployed model."""

import argparse

from core.models.orientation import OrientationModel
from core.config import OUTPUT_FOLDER_PATH, ORIENTATION_DEFAULT_ARCH, ORIENTATION_ARCH_WEIGHTS


def train(work_dir=OUTPUT_FOLDER_PATH, arch=ORIENTATION_DEFAULT_ARCH):
    OrientationModel().train(work_dir=work_dir, arch=arch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the orientation classifier")
    parser.add_argument("--arch", default=ORIENTATION_DEFAULT_ARCH, choices=list(ORIENTATION_ARCH_WEIGHTS))
    args = parser.parse_args()
    train(arch=args.arch)

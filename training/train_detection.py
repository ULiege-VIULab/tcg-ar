"""Train the Oriented R-CNN card detector (mmrotate).

Wrapper over ``core.models.detection.DetectionModel``.  The shipped config carries
machine-specific absolute data paths, so the train/val/test sets are overridden
here to the ``ASSETS_ROOT``-relative folders (portable).  After training, the final
checkpoint is copied to ``DETECTION_WEIGHT_PATH`` so the new detector becomes the
one the pipeline uses by default.
"""

import os
import shutil
import argparse

from core.models.detection import DetectionModel
from core.config import (DETECTION_CONFIG_PATH, DETECTION_MODEL_PATH, DETECTION_WEIGHT_PATH,
                         DETECTION_TRAINING_SET_FOLDER_PATH, DETECTION_VALIDATION_SET_FOLDER_PATH,
                         DETECTION_TEST_SET_FOLDER_PATH)


def _data_cfg_options():
    # Forward slashes: mmcv dumps the merged config to a .py file via yapf, which
    # would parse Windows backslashes as string escapes ("unterminated string
    # literal").  Forward-slash paths are valid for file IO on Windows and for yapf.
    def fwd(p):
        return p.replace("\\", "/")
    return {
        "data.train.ann_file": fwd(DETECTION_TRAINING_SET_FOLDER_PATH + "annotations/"),
        "data.train.img_prefix": fwd(DETECTION_TRAINING_SET_FOLDER_PATH + "images/"),
        "data.val.ann_file": fwd(DETECTION_VALIDATION_SET_FOLDER_PATH + "annotations/"),
        "data.val.img_prefix": fwd(DETECTION_VALIDATION_SET_FOLDER_PATH + "images/"),
        "data.test.ann_file": fwd(DETECTION_TEST_SET_FOLDER_PATH + "annotations/"),
        "data.test.img_prefix": fwd(DETECTION_TEST_SET_FOLDER_PATH + "images/"),
    }


def _integrate_weight(work_dir):
    """Copy the final checkpoint into the default weight path used by the pipeline."""
    candidate = os.path.join(work_dir, "latest.pth")
    if not os.path.exists(candidate):
        epochs = [f for f in os.listdir(work_dir) if f.startswith("epoch_") and f.endswith(".pth")]
        if not epochs:
            print("WARNING: no checkpoint found in", work_dir)
            return
        candidate = os.path.join(work_dir, max(epochs, key=lambda f: int(f[6:-4])))
    os.makedirs(os.path.dirname(DETECTION_WEIGHT_PATH), exist_ok=True)
    shutil.copy2(candidate, DETECTION_WEIGHT_PATH)
    print(f"Integrated detection weight: {candidate} -> {DETECTION_WEIGHT_PATH}")


def train(config_path=DETECTION_CONFIG_PATH, work_dir=DETECTION_MODEL_PATH, integrate=True):
    # Forward slashes everywhere: mmcv dumps the merged config (work_dir included)
    # to a .py via yapf, which mis-parses Windows backslashes (a trailing '\' before
    # the closing quote becomes an escaped quote -> "unterminated string literal").
    work_dir = work_dir.replace("\\", "/")
    model = DetectionModel()
    model.train(config_path=config_path, work_dir=work_dir, cfg_options=_data_cfg_options())
    if integrate:
        _integrate_weight(work_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the detection model")
    parser.add_argument("--config", default=DETECTION_CONFIG_PATH)
    parser.add_argument("--work-dir", default=DETECTION_MODEL_PATH)
    parser.add_argument("--no-integrate", action="store_true", help="don't copy the final weight to the default path")
    args = parser.parse_args()
    train(args.config, args.work_dir, integrate=not args.no_integrate)

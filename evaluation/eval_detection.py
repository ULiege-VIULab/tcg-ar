"""
Evaluate the detection model (mAP) on the synthetic test set or the real test set,
plus an FPS benchmark.  Reuses ``DetectionModel.test`` (mmrotate's evaluator).

The shipped mmrotate config carries machine-specific absolute data paths, so the
test set's ``ann_file`` / ``img_prefix`` are overridden here to the configured
folders, making the evaluation portable and switchable synthetic <-> real.
"""

import argparse

from core.models.detection import DetectionModel
from core.config import (DETECTION_CONFIG_PATH, DETECTION_WEIGHT_PATH, DETECTION_MODEL_PATH,
                         DETECTION_TEST_SET_FOLDER_PATH, REAL_SET_FOLDER_PATH)


def evaluate_map(config_path=DETECTION_CONFIG_PATH, checkpoint_path=DETECTION_WEIGHT_PATH,
                 work_dir=DETECTION_MODEL_PATH, test_folder=DETECTION_TEST_SET_FOLDER_PATH):
    model = DetectionModel()
    cfg_options = {
        "data.test.ann_file": test_folder + "annotations/",
        "data.test.img_prefix": test_folder + "images/",
    }
    result = model.test(config_path=config_path, checkpoint_path=checkpoint_path,
                        work_dir=work_dir, eval_metrics="mAP", cfg_options=cfg_options)
    print("metrics:", result["metrics"])
    return result["metrics"]


def benchmark(nb_image=100, config_path=DETECTION_CONFIG_PATH, checkpoint_path=DETECTION_WEIGHT_PATH):
    DetectionModel().benchmark_fps(nb_image=nb_image, config_path=config_path, checkpoint_path=checkpoint_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the detection model")
    parser.add_argument("--dataset", choices=["synthetic", "real"], default="synthetic")
    parser.add_argument("--fps", action="store_true", help="run an FPS benchmark instead of mAP")
    parser.add_argument("--config", default=DETECTION_CONFIG_PATH)
    parser.add_argument("--checkpoint", default=DETECTION_WEIGHT_PATH)
    args = parser.parse_args()

    if args.fps:
        benchmark(config_path=args.config, checkpoint_path=args.checkpoint)
    else:
        test_folder = REAL_SET_FOLDER_PATH if args.dataset == "real" else DETECTION_TEST_SET_FOLDER_PATH
        evaluate_map(args.config, args.checkpoint, test_folder=test_folder)

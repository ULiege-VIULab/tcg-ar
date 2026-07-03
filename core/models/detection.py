"""
Oriented R-CNN card detector (mmrotate 1.x / mmdet 3.x / mmengine).
Class-based wrapper providing train, test (mAP), single-image inference,
card extraction and an FPS benchmark.
"""

import os
import os.path as osp
import time
import ast
from typing import Iterable

import cv2
import torch
import numpy as np
from tqdm import tqdm

from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.hooks import Hook
from mmengine.registry import HOOKS

from core.config import (DETECTION_CONFIG_PATH, DETECTION_WEIGHT_PATH, DETECTION_MODEL_PATH,
                         DETECTION_TEST_SET_FOLDER_PATH)
from core.image_ops import add_red_boxes


@HOOKS.register_module(force=True)
class TqdmBarHook(Hook):
    """tqdm progress bar over the whole training run, updated per iteration with
    the running loss and current epoch. Any error in the bar is silently ignored
    so it never disturbs the (expensive) training."""

    def before_run(self, runner):
        try:
            self._bar = tqdm(total=runner.max_iters, dynamic_ncols=True, mininterval=1.0,
                             desc="train", initial=runner.iter)
        except Exception:
            self._bar = None

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):  # noqa: ARG002
        if getattr(self, "_bar", None) is None:
            return
        try:
            self._bar.update(1)
            if isinstance(outputs, dict):
                loss = outputs.get("loss")
                if loss is not None:
                    self._bar.set_postfix(loss=round(float(loss), 3), epoch=runner.epoch + 1)
        except Exception:
            pass

    def after_run(self, runner):
        if getattr(self, "_bar", None) is not None:
            try:
                self._bar.close()
            except Exception:
                pass


class DetectionModel:
    def __init__(self):
        self.model = None

    @staticmethod
    def _parse_cfg_options_like_cli(cfg_options):
        if cfg_options is None:
            return {}
        if isinstance(cfg_options, dict):
            return cfg_options
        out = {}
        for kv in cfg_options:
            if "=" not in kv:
                raise ValueError(f"Invalid cfg option '{kv}', expected key=value")
            key, val = kv.split("=", 1)
            val = val.strip()
            if ("," in val) and not (val.startswith(("[", "(", "{", '"', "'"))
                                     and val.endswith(("]", ")", "}", '"', "'"))):
                parsed = []
                for p in (p.strip() for p in val.split(",")):
                    try:
                        parsed.append(ast.literal_eval(p))
                    except Exception:
                        low = p.lower()
                        parsed.append({"true": True, "false": False, "none": None}.get(low, p))
            else:
                try:
                    parsed = ast.literal_eval(val)
                except Exception:
                    low = val.lower()
                    parsed = {"true": True, "false": False, "none": None}.get(low, val)
            out[key] = parsed
        return out

    def train(self, config_path: str, *, work_dir: str | None = None, resume_from: str | None = None,
              auto_resume: bool = False, no_validate: bool = False, gpus: int | None = None,
              gpu_ids: Iterable[int] | None = None, seed: int | None = None, diff_seed: bool = False,
              deterministic: bool = False, cfg_options=None, launcher: str = "none", local_rank: int = 0):
        if 'LOCAL_RANK' not in os.environ:
            os.environ['LOCAL_RANK'] = str(local_rank)

        cfg = Config.fromfile(config_path)
        merged_opts = self._parse_cfg_options_like_cli(cfg_options)
        if merged_opts:
            cfg.merge_from_dict(merged_opts)

        if work_dir is not None:
            cfg.work_dir = work_dir
        elif not cfg.get('work_dir'):
            cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(config_path))[0])

        if resume_from is not None:
            cfg.resume = True
            cfg.load_from = resume_from
        elif auto_resume:
            cfg.resume = True

        if seed is not None:
            cfg.randomness = dict(seed=seed, deterministic=deterministic, diff_rank_seed=diff_seed)

        # Inject the tqdm progress-bar hook on top of whatever the config already has.
        custom_hooks = list(cfg.get('custom_hooks', []) or [])
        custom_hooks.append(dict(type='TqdmBarHook', priority='VERY_LOW'))
        cfg.custom_hooks = custom_hooks

        runner = Runner.from_cfg(cfg)
        runner.train()

    def test(self, config_path: str, checkpoint_path: str, *, work_dir: str | None = None,
             out: str | None = None, fuse_conv_bn_flag: bool = False, gpu_ids: list | None = None,
             format_only: bool = False, eval_metrics=None, show: bool = False, show_dir: str | None = None,
             show_score_thr: float = 0.3, gpu_collect: bool = False, tmpdir: str | None = None,
             cfg_options=None, eval_options: dict | None = None, launcher: str = "none", local_rank: int = 0):
        if not (out or eval_metrics or format_only or show or show_dir):
            raise ValueError('Specify at least one of: out / eval / format_only / show / show_dir')
        if eval_metrics and format_only:
            raise ValueError('eval and format_only cannot both be specified')
        if 'LOCAL_RANK' not in os.environ:
            os.environ['LOCAL_RANK'] = str(local_rank)

        cfg = Config.fromfile(config_path)
        merged_opts = self._parse_cfg_options_like_cli(cfg_options)
        if merged_opts:
            cfg.merge_from_dict(merged_opts)

        cfg.load_from = checkpoint_path
        if work_dir is not None:
            cfg.work_dir = work_dir

        runner = Runner.from_cfg(cfg)
        metrics = runner.test()

        if out is not None and metrics is not None:
            import mmcv
            mmcv.dump(metrics, out)

        return {"metrics": metrics, "out_file": out}

    def load_detection_model(self, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        from mmrotate.utils import register_all_modules
        from mmdet.apis import init_detector
        register_all_modules(init_default_scope=False)
        self.model = init_detector(DETECTION_CONFIG_PATH, DETECTION_WEIGHT_PATH, device=device)
        return self.model

    def detect_card_location(self, image):
        from mmdet.apis import inference_detector
        result = inference_detector(self.model, image)
        pred = result.pred_instances
        raw = pred.bboxes
        # mmrotate 1.x may return RotatedBoxes (has .tensor) or a plain Tensor
        bboxes = (raw.tensor if hasattr(raw, 'tensor') else raw).cpu().numpy()
        return [[int(b[0]), int(b[1]), int(b[2]), int(b[3]), b[4] / np.pi * 180] for b in bboxes]

    def extract_cards(self, image, bboxes):
        cards = []
        for bbox in bboxes:
            rot_mat = cv2.getRotationMatrix2D((bbox[0], bbox[1]), bbox[4], 1.0)
            rot_mat[0, 2] -= (bbox[0] - bbox[2] / 2)
            rot_mat[1, 2] -= (bbox[1] - bbox[3] / 2)
            card = cv2.warpAffine(image, rot_mat, (bbox[2], bbox[3]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            cards.append(cv2.rotate(card, cv2.ROTATE_90_CLOCKWISE))
        return cards

    def inference(self, image):
        return self.detect_card_location(image)

    def benchmark_fps(self, nb_image=100, config_path=DETECTION_CONFIG_PATH, checkpoint_path=DETECTION_WEIGHT_PATH):
        from mmrotate.utils import register_all_modules
        from mmdet.apis import init_detector, inference_detector
        register_all_modules(init_default_scope=False)
        for device in [torch.device('cuda'), torch.device('cpu')]:
            self.model = init_detector(config_path, checkpoint_path, device=device)
            sec_elapsed = 0.0
            for i in range(nb_image):
                image = cv2.imread(DETECTION_TEST_SET_FOLDER_PATH + "images/" + str(i) + ".png")
                start = time.time()
                inference_detector(self.model, image)
                sec_elapsed += time.time() - start
            print(f"FPS on {device}: {nb_image / sec_elapsed:.2f} frame/sec.")

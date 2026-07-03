"""
Post-install compatibility patches for Python 3.14 + mmcv 2.2.0 + mmdet 3.x + mmrotate 1.0.0rc1.

Run once after installing all dependencies (idempotent — safe to re-run):
    python -m scripts.patch_mmlibs
"""
import importlib.util
import pathlib
import sys

# ── helpers ───────────────────────────────────────────────────────────────────

def _pkg_root(name: str) -> pathlib.Path:
    spec = importlib.util.find_spec(name)
    if spec is None:
        print(f"  [ERROR] Package '{name}' is not installed — install dependencies first.")
        sys.exit(1)
    locs = spec.submodule_search_locations
    if locs:
        return pathlib.Path(list(locs)[0])
    return pathlib.Path(spec.origin).parent


_ok = _skip = _warn = 0


def patch_file(path: pathlib.Path, old: str, new: str, desc: str) -> None:
    global _ok, _skip, _warn
    text = path.read_text(encoding='utf-8')
    if new in text:
        print(f"  [skip] {desc}")
        _skip += 1
        return
    if old not in text:
        print(f"  [WARN] {desc}")
        print(f"         expected text not found in {path.name} — skipping (version mismatch?)")
        _warn += 1
        return
    path.write_text(text.replace(old, new, 1), encoding='utf-8')
    print(f"  [OK]   {desc}")
    _ok += 1


def write_file(path: pathlib.Path, content: str, desc: str) -> None:
    global _ok, _skip
    if path.exists() and path.read_text(encoding='utf-8') == content:
        print(f"  [skip] {desc}")
        _skip += 1
        return
    path.write_text(content, encoding='utf-8')
    print(f"  [OK]   {desc}")
    _ok += 1


# ── new file contents to inject into mmrotate ─────────────────────────────────

ORIENTED_RCNN_PY = """\
# Copyright (c) OpenMMLab. All rights reserved.
# Backported from mmrotate main branch for mmrotate 1.0.0rc1 + mmdet 3.x.
from mmdet.models.detectors.two_stage import TwoStageDetector

from mmrotate.registry import MODELS


@MODELS.register_module()
class OrientedRCNN(TwoStageDetector):
    \"\"\"Oriented R-CNN for Object Detection.

    `Oriented R-CNN for Object Detection <https://arxiv.org/abs/2108.05699>`_.

    Uses OrientedRPNHead (from mmrotate) to generate rotated region proposals
    and RotatedShared2FCBBoxHead + RotatedSingleRoIExtractor (both in
    mmrotate) for the second stage.  All core logic lives in TwoStageDetector;
    registering under the mmrotate scope is the only thing this class adds.
    \"\"\"
"""

ORIENTED_STANDARD_ROI_HEAD_PY = """\
# Copyright (c) OpenMMLab. All rights reserved.
# Backported from mmrotate main branch for mmrotate 1.0.0rc1 + mmdet 3.x.
from mmdet.models.roi_heads.standard_roi_head import StandardRoIHead

from mmrotate.registry import MODELS


@MODELS.register_module()
class OrientedStandardRoIHead(StandardRoIHead):
    \"\"\"Standard RoI head for Oriented R-CNN.

    Inherits all logic from mmdet's StandardRoIHead.  Works with rotated
    proposals because mmdet's bbox2roi calls get_box_tensor() which handles
    RotatedBoxes natively, producing (N, 6) rois consumed by
    RotatedSingleRoIExtractor.
    \"\"\"
"""


def main() -> None:
    print("Applying mmrotate/mmdet/mmengine compatibility patches for Python 3.14...\n")

    # ── mmengine ──────────────────────────────────────────────────────────────
    mmengine = _pkg_root('mmengine')
    print("[mmengine]")
    patch_file(
        mmengine / 'utils' / 'dl_utils' / 'misc.py',
        old="    ext_loader = pkgutil.find_loader('mmcv._ext')\n    return ext_loader is not None",
        new="    import importlib.util\n    ext_loader = importlib.util.find_spec('mmcv._ext')\n    return ext_loader is not None",
        desc="pkgutil.find_loader → importlib.util.find_spec (removed in Python 3.14)"
    )

    # ── mmdet ─────────────────────────────────────────────────────────────────
    mmdet = _pkg_root('mmdet')
    print("\n[mmdet]")
    patch_file(
        mmdet / '__init__.py',
        old="mmcv_maximum_version = '2.2.0'",
        new="mmcv_maximum_version = '2.3.0'",
        desc="mmcv_maximum_version 2.2.0 → 2.3.0"
    )

    # ── mmrotate ──────────────────────────────────────────────────────────────
    mmrotate = _pkg_root('mmrotate')
    print("\n[mmrotate]")

    patch_file(
        mmrotate / '__init__.py',
        old="mmcv_maximum_version = '2.1.0'",
        new="mmcv_maximum_version = '2.3.0'",
        desc="mmcv_maximum_version 2.1.0 → 2.3.0"
    )
    patch_file(
        mmrotate / '__init__.py',
        old="mmdet_maximum_version = '3.1.0'",
        new="mmdet_maximum_version = '3.4.0'",
        desc="mmdet_maximum_version 3.1.0 → 3.4.0"
    )

    reg = mmrotate / 'registry.py'
    for name in ('DATASETS', 'METRICS', 'TASK_UTILS', 'TRANSFORMS'):
        patch_file(
            reg,
            old=f"from mmengine.registry import {name} as MMENGINE_{name}",
            new=f"from mmdet.registry import {name} as MMENGINE_{name}",
            desc=f"registry.py: {name} parent mmengine → mmdet"
        )

    patch_file(
        mmrotate / 'models' / 'detectors' / 'refine_single_stage.py',
        old="from collections import Sequence",
        new="from collections.abc import Sequence",
        desc="refine_single_stage.py: collections.Sequence → collections.abc.Sequence (Python 3.10+)"
    )

    write_file(
        mmrotate / 'models' / 'detectors' / 'oriented_rcnn.py',
        ORIENTED_RCNN_PY,
        desc="detectors/oriented_rcnn.py: create OrientedRCNN stub"
    )
    patch_file(
        mmrotate / 'models' / 'detectors' / '__init__.py',
        old=(
            "from .h2rbox import H2RBoxDetector\n"
            "from .refine_single_stage import RefineSingleStageDetector\n"
            "\n"
            "__all__ = ['RefineSingleStageDetector', 'H2RBoxDetector']"
        ),
        new=(
            "from .h2rbox import H2RBoxDetector\n"
            "from .oriented_rcnn import OrientedRCNN\n"
            "from .refine_single_stage import RefineSingleStageDetector\n"
            "\n"
            "__all__ = ['RefineSingleStageDetector', 'H2RBoxDetector', 'OrientedRCNN']"
        ),
        desc="detectors/__init__.py: export OrientedRCNN"
    )

    write_file(
        mmrotate / 'models' / 'roi_heads' / 'oriented_standard_roi_head.py',
        ORIENTED_STANDARD_ROI_HEAD_PY,
        desc="roi_heads/oriented_standard_roi_head.py: create OrientedStandardRoIHead stub"
    )
    patch_file(
        mmrotate / 'models' / 'roi_heads' / '__init__.py',
        old=(
            "from .gv_ratio_roi_head import GVRatioRoIHead\n"
            "from .roi_extractors import RotatedSingleRoIExtractor\n"
            "\n"
            "__all__ = [\n"
            "    'RotatedShared2FCBBoxHead', 'RotatedSingleRoIExtractor', 'GVRatioRoIHead'\n"
            "]"
        ),
        new=(
            "from .gv_ratio_roi_head import GVRatioRoIHead\n"
            "from .oriented_standard_roi_head import OrientedStandardRoIHead\n"
            "from .roi_extractors import RotatedSingleRoIExtractor\n"
            "\n"
            "__all__ = [\n"
            "    'RotatedShared2FCBBoxHead', 'RotatedSingleRoIExtractor', 'GVRatioRoIHead',\n"
            "    'OrientedStandardRoIHead'\n"
            "]"
        ),
        desc="roi_heads/__init__.py: export OrientedStandardRoIHead"
    )

    patch_file(
        mmrotate / 'utils' / 'setup_env.py',
        old="    import mmrotate.datasets  # noqa: F401,F403",
        new=(
            "    import mmdet.utils as _mmdet_utils\n"
            "    _mmdet_utils.register_all_modules(init_default_scope=False)\n"
            "    import mmrotate.datasets  # noqa: F401,F403"
        ),
        desc="setup_env.py: register mmdet modules before mmrotate imports"
    )

    # ── summary ───────────────────────────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print(f"Done: {_ok} applied, {_skip} already up-to-date, {_warn} warnings.")
    if _warn:
        print("Check WARN lines above — installed package versions may differ from tested.")
        sys.exit(1)
    else:
        print("All patches applied. Restart your Python process before running the app.")


if __name__ == '__main__':
    main()

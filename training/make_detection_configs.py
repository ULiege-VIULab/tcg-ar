"""Generate mmrotate detector configs for the experiment comparison.

For each alternative method we keep the *exact* data pipeline, optimizer and 5-epoch
schedule of the deployed Oriented R-CNN config and only swap the model architecture
(taken from the method's stock ``r50_fpn_1x_dota_le90`` config), with ``num_classes``
forced to 1 (single ``card`` class). The deployed config/weights are never touched.

Run once:  python -m training.make_detection_configs
Then train e.g.:  python -m training.train_detection --config <out>/rotated_fcos.py --work-dir work_dirs/rotated_fcos --no-integrate
"""

import os
import copy

from mmengine.config import Config

from core.config import DETECTION_CONFIG_PATH, _DETECTION_MODEL_DIR

# method name -> stock mmrotate config (relative to mmrotate/.mim/configs)
METHODS = {
    "rotated_retinanet": "rotated_retinanet/rotated_retinanet_obb_r50_fpn_1x_dota_le90.py",
    "rotated_fcos":      "rotated_fcos/rotated_fcos_r50_fpn_1x_dota_le90.py",
    "roi_trans":         "roi_trans/roi_trans_r50_fpn_1x_dota_le90.py",
    "gliding_vertex":    "gliding_vertex/gliding_vertex_r50_fpn_1x_dota_le90.py",
}

OUT_DIR = os.path.join(_DETECTION_MODEL_DIR, "configs")


def _mmrotate_config_dir():
    import mmrotate
    pkg_dir = os.path.dirname(mmrotate.__file__)
    # mmrotate 1.x ships configs inside the package; 0.x used the .mim directory.
    for candidate in (os.path.join(pkg_dir, 'configs'), os.path.join(pkg_dir, '.mim', 'configs')):
        if os.path.isdir(candidate):
            return candidate
    raise RuntimeError(f"Cannot find mmrotate configs directory under {pkg_dir}")


def _set_num_classes(node, n=1):
    if isinstance(node, dict):
        for k, v in node.items():
            if k == "num_classes":
                node[k] = n
            else:
                _set_num_classes(v, n)
    elif isinstance(node, (list, tuple)):
        for v in node:
            _set_num_classes(v, n)


def generate():
    os.makedirs(OUT_DIR, exist_ok=True)
    cfg_dir = _mmrotate_config_dir()
    base = Config.fromfile(DETECTION_CONFIG_PATH)        # flattened ORCNN (data+schedule)

    written = []
    for name, rel in METHODS.items():
        method_cfg = Config.fromfile(os.path.join(cfg_dir, rel))
        new = copy.deepcopy(base)

        model = copy.deepcopy(method_cfg["model"])
        # Some configs keep train_cfg/test_cfg at the top level; nest them in the
        # model so build_detector(cfg.model) sees them (as the ORCNN config does).
        if "train_cfg" in method_cfg and model.get("train_cfg") is None:
            model["train_cfg"] = copy.deepcopy(method_cfg["train_cfg"])
        if "test_cfg" in method_cfg and model.get("test_cfg") is None:
            model["test_cfg"] = copy.deepcopy(method_cfg["test_cfg"])
        _set_num_classes(model, 1)
        new["model"] = model

        # Avoid double-passing: keep cfg.train_cfg/test_cfg unset at top level.
        for k in ("train_cfg", "test_cfg"):
            if k in new:
                del new[k]

        # Forward-slash work_dir so the dumped/merged config parses on Windows.
        new["work_dir"] = ("work_dirs/" + name)

        out_path = os.path.join(OUT_DIR, name + ".py").replace("\\", "/")
        new.dump(out_path)
        written.append(out_path)
        print(f"[{name}] {rel} -> {out_path}")

    print(f"\nWrote {len(written)} configs to {OUT_DIR}")
    return written


if __name__ == "__main__":
    generate()

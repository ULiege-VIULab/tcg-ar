"""
Reader / writer for the real captured test set (``assets/AI database/real/``).

The real set has 1920x1080 images with detection ground truth in both DOTA
(``annotations/``) and YOLO-OBB (``annotations_yolo/``) formats.  Card identities
are annotated with ``inference/annotate_identities.py`` and stored DOTA-like in
``annotations_identity/``.

This module is the single place that knows the real-set layout: it is used by the
annotation tool and by the real-set evaluations.
"""

import os
import glob
import random

import cv2
import numpy as np

from core.config import (REAL_IMAGES_FOLDER, REAL_ANNOTATIONS_FOLDER, REAL_ANNOTATIONS_YOLO_FOLDER,
                         REAL_IDENTITY_FOLDER, REAL_ORIENTATION_SET_FOLDER)

UNKNOWN_ID = "unknown"
_CLASS_MAPPING = {0: "card"}


# --------------------------------------------------------------------------- #
# Images + detection boxes
# --------------------------------------------------------------------------- #
def real_image_indices():
    """Sorted integer indices of the real images (``0.png`` -> 0, ...)."""
    idx = []
    for p in glob.glob(os.path.join(REAL_IMAGES_FOLDER, "*.png")):
        name = os.path.splitext(os.path.basename(p))[0]
        if name.isdigit():
            idx.append(int(name))
    return sorted(idx)


def num_real_images():
    return len(real_image_indices())


def load_real_image(i):
    return cv2.imread(os.path.join(REAL_IMAGES_FOLDER, f"{i}.png"))


def _read_dota_file(path):
    """Return [(poly[8 ints], class, difficulty), ...] from a DOTA .txt."""
    boxes = []
    if not os.path.exists(path):
        return boxes
    with open(path, "r") as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) < 10:
                continue
            poly = [int(round(float(v))) for v in parts[0:8]]
            boxes.append((poly, parts[8], parts[9]))
    return boxes


def read_dota_boxes(i):
    """Detection ground-truth polygons for real image ``i`` (list of 8-int polys)."""
    return [poly for poly, _cls, _diff in _read_dota_file(os.path.join(REAL_ANNOTATIONS_FOLDER, f"{i}.txt"))]


# --------------------------------------------------------------------------- #
# Cropping a card from a 4-corner polygon (mirrors DetectionModel.extract_cards)
# --------------------------------------------------------------------------- #
def polygon_to_obb(poly):
    """4-corner polygon -> (cx, cy, w, h, angle) via minAreaRect."""
    pts = np.array(poly, dtype=np.float32).reshape(4, 2)
    (cx, cy), (w, h), angle = cv2.minAreaRect(pts)
    return cx, cy, w, h, angle


def crop_card(image, poly, flip=False):
    """Deskew + crop the card inside ``poly`` to an upright portrait image.

    Same idea as ``DetectionModel.extract_cards``: rotate the oriented box to axis
    aligned, crop it, force portrait.  The 180-degree ambiguity (upright vs upside
    down) is resolved separately (ArcFace + user), so ``flip`` rotates the result."""
    cx, cy, w, h, angle = polygon_to_obb(poly)
    w, h = max(1, int(round(w))), max(1, int(round(h)))
    rot = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rot[0, 2] -= (cx - w / 2)
    rot[1, 2] -= (cy - h / 2)
    card = cv2.warpAffine(image, rot, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    if w > h:                                   # cards are taller than wide
        card = cv2.rotate(card, cv2.ROTATE_90_CLOCKWISE)
    if flip:
        card = cv2.rotate(card, cv2.ROTATE_180)
    return card


# --------------------------------------------------------------------------- #
# YOLO-OBB -> DOTA converter (verified equivalent to the reference converter.py)
# --------------------------------------------------------------------------- #
def yolo_to_dota(line, width, height, class_mapping=_CLASS_MAPPING):
    """Convert one YOLO-OBB line (class + 8 normalised coords) to a DOTA line."""
    parts = line.strip().split()
    if len(parts) < 9:
        return None
    class_name = class_mapping.get(int(parts[0]), f"class_{parts[0]}")
    coords = []
    for k in range(1, 9):
        dim = width if (k % 2 == 1) else height   # odd k -> x*width, even k -> y*height
        coords.append(int(float(parts[k]) * dim))
    return " ".join(str(c) for c in coords) + f" {class_name} 0"


def verify_conversion():
    """Assert yolo_to_dota reproduces the existing real DOTA annotations exactly.
    Returns (files_checked, lines_checked)."""
    files = lines = 0
    for path in sorted(glob.glob(os.path.join(REAL_ANNOTATIONS_YOLO_FOLDER, "*.txt"))):
        base = os.path.splitext(os.path.basename(path))[0]
        image = load_real_image(int(base)) if base.isdigit() else cv2.imread(
            os.path.join(REAL_IMAGES_FOLDER, base + ".png"))
        h, w = image.shape[:2]
        produced = []
        with open(path) as f:
            for line in f.readlines():
                d = yolo_to_dota(line, w, h)
                if d is not None:
                    produced.append(d)
        existing = []
        with open(os.path.join(REAL_ANNOTATIONS_FOLDER, base + ".txt")) as f:
            existing = [ln.strip() for ln in f.readlines() if ln.strip()]
        assert produced == existing, f"mismatch in {base}.txt:\n  got {produced[:2]}\n  exp {existing[:2]}"
        files += 1
        lines += len(produced)
    return files, lines


# --------------------------------------------------------------------------- #
# Identity annotations (DOTA-like: x1..y4 <card_id> <difficulty> <orientation>)
# --------------------------------------------------------------------------- #
def identity_path(i):
    return os.path.join(REAL_IDENTITY_FOLDER, f"{i}.txt")


def is_annotated(i):
    return os.path.exists(identity_path(i))


def read_identity_annotations(i):
    """Return [{'poly': [8 ints], 'card_id': str, 'orientation': int}, ...]."""
    entries = []
    path = identity_path(i)
    if not os.path.exists(path):
        return entries
    with open(path, "r") as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) < 10:
                continue
            poly = [int(round(float(v))) for v in parts[0:8]]
            card_id = parts[8]
            orientation = int(parts[10]) if len(parts) >= 11 else 0
            entries.append({"poly": poly, "card_id": card_id, "orientation": orientation})
    return entries


def write_identity_annotation(i, entries):
    """``entries``: list of {'poly': [8 ints], 'card_id': str, 'orientation': 0/1}."""
    os.makedirs(REAL_IDENTITY_FOLDER, exist_ok=True)
    lines = []
    for e in entries:
        coords = " ".join(str(int(round(v))) for v in e["poly"])
        lines.append(f"{coords} {e['card_id']} 0 {int(e.get('orientation', 0))}")
    with open(identity_path(i), "w") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))


# --------------------------------------------------------------------------- #
# Iteration for evaluation + orientation dummy set
# --------------------------------------------------------------------------- #
def iter_real_crops(upright=True, include_unknown=False):
    """Yield (image_idx, poly, crop, card_id, orientation) for every annotated box.

    With ``upright`` the crop is rotated to its ground-truth upright orientation
    (so identification is measured independently of the orientation model)."""
    for i in real_image_indices():
        entries = read_identity_annotations(i)
        if not entries:
            continue
        image = load_real_image(i)
        for e in entries:
            if e["card_id"] == UNKNOWN_ID and not include_unknown:
                continue
            crop = crop_card(image, e["poly"], flip=(upright and e["orientation"] == 1))
            yield i, e["poly"], crop, e["card_id"], e["orientation"]


def build_real_orientation_dataset(flip_ratio=0.5, seed=0):
    """Build a balanced straight/flip orientation set from the upright real crops.

    Each upright crop is kept as 'straight' or rotated 180 degrees as 'flip'
    (~``flip_ratio`` flipped), matching the requested dummy dataset.  Returns
    (n_straight, n_flip)."""
    straight_dir = os.path.join(REAL_ORIENTATION_SET_FOLDER, "straight")
    flip_dir = os.path.join(REAL_ORIENTATION_SET_FOLDER, "flip")
    os.makedirs(straight_dir, exist_ok=True)
    os.makedirs(flip_dir, exist_ok=True)

    crops = [crop for _i, _poly, crop, _cid, _o in iter_real_crops(upright=True)]
    rng = random.Random(seed)
    order = list(range(len(crops)))
    rng.shuffle(order)
    n_flip = int(round(flip_ratio * len(crops)))
    flip_set = set(order[:n_flip])

    n_s = n_f = 0
    for k, crop in enumerate(crops):
        if k in flip_set:
            cv2.imwrite(os.path.join(flip_dir, f"{k}.png"), cv2.rotate(crop, cv2.ROTATE_180))
            n_f += 1
        else:
            cv2.imwrite(os.path.join(straight_dir, f"{k}.png"), crop)
            n_s += 1
    return n_s, n_f

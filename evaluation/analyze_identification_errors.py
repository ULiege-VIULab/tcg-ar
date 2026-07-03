"""Evaluate identification on the real set for both heads (arcface, triplet), with and
without the deck restriction, and -- for every ArcFace top-1 error -- save a composite
image (the input crop + the five top-5 reference card images, the ground truth marked)
to ``work_dirs/arcface_errors/<cond>/<top5hit|top5miss>/`` so the error types can be
inspected.

Run:  python -m evaluation.analyze_identification_errors
"""
import os
import shutil

import cv2
import numpy as np
import torch
from tqdm import tqdm

from core import config
from core.config import IDENTIFICATION_IMAGE_SIZE, POKEMON_CARD_DATABASE_FOLDER_PATH, PROJECT_ROOT
from core.models import identification as idm
from core.transforms import get_inference_transform
import core.real_data as rd

OUT_ROOT = os.path.join(PROJECT_ROOT, "work_dirs", "arcface_errors")
TILE_H = 240          # reference/crop height in the composite
TILE_W = 172          # reference/crop width
PAD = 8
LABEL_H = 26


def _device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _batched(items, n):
    for i in range(0, len(items), n):
        yield items[i:i + n]


def _card_tile(img_bgr, header, footer, border=None):
    """A fixed-size labelled tile: header text on top, image, footer text below."""
    tile = np.full((LABEL_H + TILE_H + LABEL_H, TILE_W, 3), 30, np.uint8)
    if img_bgr is None:
        cv2.putText(tile, "no image", (6, LABEL_H + TILE_H // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    else:
        h, w = img_bgr.shape[:2]
        s = min(TILE_W / w, TILE_H / h)
        rs = cv2.resize(img_bgr, (max(1, int(w * s)), max(1, int(h * s))))
        y0 = LABEL_H + (TILE_H - rs.shape[0]) // 2
        x0 = (TILE_W - rs.shape[1]) // 2
        tile[y0:y0 + rs.shape[0], x0:x0 + rs.shape[1]] = rs
    cv2.putText(tile, header[:22], (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (235, 235, 235), 1, cv2.LINE_AA)
    cv2.putText(tile, footer[:22], (4, LABEL_H + TILE_H + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (235, 235, 235), 1, cv2.LINE_AA)
    if border is not None:
        cv2.rectangle(tile, (0, 0), (TILE_W - 1, tile.shape[0] - 1), border, 3)
    return tile


def _ref_image(card_id):
    p = POKEMON_CARD_DATABASE_FOLDER_PATH + str(card_id) + ".jpg"
    return cv2.imread(p) if os.path.exists(p) else None


def save_error_composite(out_dir, idx, crop_bgr, gt, proposals):
    """proposals: [(card_id, score), ...] top-5 best->worst."""
    os.makedirs(out_dir, exist_ok=True)
    GREEN, RED = (60, 200, 60), (60, 60, 220)
    tiles = [_card_tile(crop_bgr, "INPUT", "gt=" + gt, border=(200, 200, 200))]
    for rank, (cid, score) in enumerate(proposals, 1):
        is_gt = (cid == gt)
        tiles.append(_card_tile(_ref_image(cid), f"#{rank} {score:.2f}",
                                cid + (" [GT]" if is_gt else ""),
                                border=GREEN if is_gt else (RED if rank == 1 else None)))
    gap = np.full((tiles[0].shape[0], PAD, 3), 30, np.uint8)
    row = tiles[0]
    for t in tiles[1:]:
        row = np.hstack([row, gap, t])
    cv2.imwrite(os.path.join(out_dir, f"{idx:03d}_{gt}__top1_{proposals[0][0]}.png"), row)


def evaluate(method, deck_restricted, crops, save_errors=False, batch_size=40, top_k=5):
    device = _device()
    model = idm.load_model(method, fine_tuned=True, device=device)
    if deck_restricted:
        present = sorted({gt for _i, _c, gt in crops})
        anchor_list = [cid + ".jpg" for cid in present]
    else:
        anchor_list = idm.all_card_anchor_list()
    anchors = idm.evaluate_anchors(model, device, anchor_list=anchor_list)
    transform = get_inference_transform(IDENTIFICATION_IMAGE_SIZE)
    cond = "deck" if deck_restricted else "open"

    correct1 = correct5 = total = n_err = 0
    for batch in tqdm(list(_batched(crops, batch_size)), desc=f"{method}/{cond}"):
        tensors = torch.stack([transform(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for _i, c, _g in batch])
        proposals = idm.identify_topk(tensors, anchors, model, device, k=top_k, method=method)
        for (img_idx, crop, gt), props in zip(batch, proposals):
            preds = [cid for cid, _s in props]
            total += 1
            top1_ok = preds[0] == gt
            top5_ok = gt in preds
            correct1 += top1_ok
            correct5 += top5_ok
            if save_errors and not top1_ok:
                n_err += 1
                sub = os.path.join(OUT_ROOT, cond, "top5hit" if top5_ok else "top5miss")
                save_error_composite(sub, img_idx, crop, gt, props)

    acc1 = 100.0 * correct1 / total if total else 0.0
    acc5 = 100.0 * correct5 / total if total else 0.0
    msg = f"[{method:7s} | {cond:4s}] top-1 {acc1:5.2f}%  top-5 {acc5:5.2f}%  ({total} cards)"
    if save_errors:
        msg += f"  -> {n_err} error composites saved"
    print(msg)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return acc1, acc5


def main():
    # one upright GT crop per annotated real card (orientation taken from the annotation)
    crops = [(i, crop, gt) for i, _poly, crop, gt, _o in rd.iter_real_crops(upright=True)]
    print(f"real set: {len(crops)} annotated cards\n")

    if os.path.isdir(OUT_ROOT):
        shutil.rmtree(OUT_ROOT)

    # ArcFace: save error composites (both conditions). Triplet: numbers only.
    evaluate("arcface", deck_restricted=False, crops=crops, save_errors=True)
    evaluate("arcface", deck_restricted=True,  crops=crops, save_errors=True)
    evaluate("triplet", deck_restricted=False, crops=crops)
    evaluate("triplet", deck_restricted=True,  crops=crops)
    print(f"\nArcFace error composites under: {OUT_ROOT}")


if __name__ == "__main__":
    main()

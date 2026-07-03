"""
Find cards whose stored image is actually a card back (wrong scans / bad downloads)
and write a banned list the rest of the code checks.

Method: embed every card image and the reference back card (``back1-1``) with the
base ArcFace model, then flag cards whose embedding is very close (cosine) to the
back -- since every card back is the same picture, a wrong scan collapses onto it.

The banned list is written to ``WRONG_SCAN_LIST_FILE`` as ``{card_id: similarity}``
(sorted, highest first) and is consumed by
``core.models.identification.all_card_anchor_list`` to exclude those cards from
training, inference and evaluation.

Usage:
    python -m installation.find_wrong_scans --dry-run         # inspect the distribution
    python -m installation.find_wrong_scans --threshold 0.85  # write the banned list
"""

import os
import json
import argparse

import torch
import torch.nn.functional as F

from core.config import POKEMON_CARD_DATABASE_FOLDER_PATH, BACK_CARD_ID, WRONG_SCAN_LIST_FILE
from core.models import identification as idm


def back_similarities():
    """Return [(card_id, cosine_similarity_to_back), ...] sorted high->low (excludes
    the back card itself)."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = idm.load_model("arcface", fine_tuned=False, device=device)
    anchor_list = sorted(f for f in os.listdir(POKEMON_CARD_DATABASE_FOLDER_PATH) if f.endswith(".jpg"))
    print(f"Embedding {len(anchor_list)} card images with base ArcFace...")
    anchors = idm.evaluate_anchors(model, device, anchor_list=anchor_list)
    if BACK_CARD_ID not in anchors:
        raise SystemExit(f"Reference back card '{BACK_CARD_ID}' image not found in the card database.")

    ids = list(anchors.keys())
    matrix = F.normalize(torch.stack([anchors[i] for i in ids]), p=2, dim=1)
    back = F.normalize(anchors[BACK_CARD_ID].unsqueeze(0), p=2, dim=1)
    sims = torch.mm(matrix, back.t()).squeeze(1).tolist()
    pairs = [(ids[k], sims[k]) for k in range(len(ids)) if ids[k] != BACK_CARD_ID]
    pairs.sort(key=lambda x: -x[1])
    return pairs


def find_wrong_scans(threshold=0.85, write=True):
    pairs = back_similarities()

    # Distribution, to calibrate the threshold (backs cluster near 1.0).
    print("\nTop 25 most back-like cards:")
    for cid, s in pairs[:25]:
        print(f"  {cid:20s} {s:.3f}")
    for cut in (0.95, 0.9, 0.85, 0.8, 0.75, 0.7):
        print(f"  cards >= {cut:.2f}: {sum(1 for _c, s in pairs if s >= cut)}")

    wrong = {cid: round(s, 4) for cid, s in pairs if s >= threshold}
    print(f"\n{len(wrong)} cards flagged as wrong scans (>= {threshold}).")
    if write:
        os.makedirs(os.path.dirname(WRONG_SCAN_LIST_FILE), exist_ok=True)
        with open(WRONG_SCAN_LIST_FILE, "w", encoding="utf-8") as f:
            json.dump(wrong, f, ensure_ascii=False, indent=2)
        print(f"Wrote banned list -> {WRONG_SCAN_LIST_FILE}")
    return pairs, wrong


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect wrongly-scanned (back-showing) cards via ArcFace")
    parser.add_argument("--threshold", type=float, default=0.85, help="cosine-to-back threshold to ban")
    parser.add_argument("--dry-run", action="store_true", help="print the distribution without writing")
    args = parser.parse_args()
    find_wrong_scans(threshold=args.threshold, write=not args.dry_run)

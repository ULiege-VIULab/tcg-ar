"""
Installation / database orchestrator (replaces the old ``installer.py``).

Every piece of logic lives in ``core.databases``; this file is only a thin,
CLI-driven sequencer.  All output is written under ``config.ASSETS_ROOT``.

Examples
--------
    # full first-time setup
    python -m installation.install --all

    # only (re)build the synthetic identification dataset
    python -m installation.install --identification-dataset

    # rebuild just the JSON metadata databases
    python -m installation.install --json-databases
"""

import argparse
import sys

from core import databases as db
from core.config import ASSETS_ROOT, POSITIVE_DATA_NUMBER, WIDTH, HEIGHT


def build_metadata_databases():
    """Pokemon list, card encyclopedia and the six identification JSON databases."""
    print("[1/3] Scraping Pokemon database (pokepedia)...")
    db.create_pokemon_database()
    print("[2/3] Building card encyclopedia (pokemon_card.json)...")
    db.create_card_file()
    print("[3/3] Building the six identification JSON databases...")
    db.create_card_databases()


def download_sprites():
    print("Downloading 2D sprites...")
    db.create_2D_database()
    print("Downloading 2D-animated sprites...")
    db.create_2D_animated_database()


def download_cards():
    print("Downloading the card-image database...")
    db.download_card_database()


def precompute_embeddings():
    """Pre-compute and cache card embeddings so first app start-up is fast.

    Requires the card-image database (--cards) and JSON metadata (--metadata)
    to be present.  Uses the base (non-fine-tuned) ArcFace weights.
    """
    import torch
    from core.models import identification as idm
    from core import config as _cfg

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    method = _cfg.IDENTIFICATION_METHOD
    model_path = _cfg.model_save_path(method, fine_tuned=False)
    print(f"Pre-computing embeddings  [method={method}  device={device}]")
    print(f"  Model: {model_path}")

    try:
        model = idm.load_model(method, fine_tuned=False, device=device)
    except FileNotFoundError:
        print("  ERROR: model weights not found. Run `python -m scripts.download_assets` first.",
              file=sys.stderr)
        return

    anchor_list = idm.all_card_anchor_list()
    print(f"  Cards: {len(anchor_list)}")

    def _progress(pct):
        filled = int(40 * pct / 100)
        sys.stdout.write(f"\r  [{'█' * filled}{'░' * (40 - filled)}] {pct:3d}%")
        sys.stdout.flush()

    idm.evaluate_anchors(model, device, anchor_list=anchor_list,
                         progress_callback=_progress, model_path=model_path)
    print("\n  Embeddings cached.")


def generate_datasets(detection=True, orientation=True, identification=True):
    if detection:
        print("Generating synthetic detection dataset...")
        db.generate_detection_datasets()
    if orientation:
        print("Generating synthetic orientation dataset...")
        db.generate_orientation_datasets()
    if identification:
        print("Generating synthetic identification dataset...")
        db.generate_identification_dataset(POSITIVE_DATA_NUMBER)


def main():
    parser = argparse.ArgumentParser(description="PTCG-AR installation / database builder")
    parser.add_argument("--all", action="store_true", help="run the complete first-time setup")
    parser.add_argument("--metadata", action="store_true", help="pokemon db + card file + json databases")
    parser.add_argument("--json-databases", action="store_true", help="only the six identification json databases")
    parser.add_argument("--cards", action="store_true", help="download the card-image database")
    parser.add_argument("--sprites", action="store_true", help="download 2D / 2D-animated sprites")
    parser.add_argument("--detection-dataset", action="store_true")
    parser.add_argument("--orientation-dataset", action="store_true")
    parser.add_argument("--identification-dataset", action="store_true")
    parser.add_argument("--embeddings", action="store_true",
                        help="pre-compute and cache card embeddings (requires --cards + --metadata first)")
    args = parser.parse_args()

    print(f"Assets root: {ASSETS_ROOT}")

    if args.all:
        build_metadata_databases()
        download_cards()
        download_sprites()
        generate_datasets()
        precompute_embeddings()
        print("Installation complete.")
        return

    did_something = False
    if args.metadata:
        build_metadata_databases(); did_something = True
    if args.json_databases:
        db.create_card_databases(); did_something = True
    if args.cards:
        download_cards(); did_something = True
    if args.sprites:
        download_sprites(); did_something = True
    if args.detection_dataset or args.orientation_dataset or args.identification_dataset:
        generate_datasets(args.detection_dataset, args.orientation_dataset, args.identification_dataset)
        did_something = True
    if args.embeddings:
        precompute_embeddings(); did_something = True

    if not did_something:
        parser.print_help()


if __name__ == "__main__":
    main()

"""Download the large TCG-AR artifacts that are not shipped in the git repository:
the pre-trained inference models and the real evaluation set.

Hosted on Google Drive.  Uses ``gdown`` (listed in requirements.txt) to handle
Drive's virus-scan confirmation for large files.

Usage:
    python -m scripts.download_assets            # fetch everything that has a URL
    python -m scripts.download_assets --list     # show what would be fetched, where
    python -m scripts.download_assets --only models
    python -m scripts.download_assets --force    # re-download even if present

Everything else (card images, sprites, JSON metadata, synthetic datasets) is built
locally with ``python -m installation.install ...`` -- see README.md / docs/DATA.md.
"""

import argparse
import os
import re
import sys
import zipfile
import tempfile

# Repo root = parent of this scripts/ folder; assets live inside it.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, "assets")

# Each entry: (group, key, env-var override, Google Drive file ID, destination relative
# to assets/, kind)  where kind is "file" or "zipdir".
#
# To override a URL set the matching env var to any direct download URL or a Drive
# share link; the script will extract the file ID automatically.
MANIFEST = [
    ("models", "detection",
     "TCGAR_DETECTION_URL",
     "1o1f1430boX6PMsRgfNhPvg8IOMfygTsJ",
     "AI models/Detection model/custom_oriented_rcnn_weight.pth",
     "file"),
    ("models", "orientation",
     "TCGAR_ORIENTATION_URL",
     "1Z0_55xZDYR9lTuuy447GXhj9gU8cesUO",
     "AI models/Orientation model/orientation_classifier.pth",
     "file"),
    ("models", "identification",
     "TCGAR_ARCFACE_URL",
     "1VuO3rFyvLHoBac3u86TAZFSkgD6huUNp",
     "AI models/Identification model/identification_model_51_arcface.pth",
     "file"),
    ("real", "real_set",
     "TCGAR_REALSET_URL",
     "1q34Ka2J5w6Xz_wUnY-MzdBQmRqukZWbH",
     "AI database/real",
     "zipdir"),
    ("background", "background",
     "TCGAR_BACKGROUND_URL",
     None,   # not yet available
     "AI database/background",
     "zipdir"),
]

_DRIVE_ID_RE = re.compile(r"/d/([a-zA-Z0-9_-]+)")


def _resolve_id(value):
    """Return a Google Drive file ID from either a bare ID or a share URL."""
    if value is None:
        return None
    m = _DRIVE_ID_RE.search(value)
    return m.group(1) if m else value  # already a bare ID


def _file_id_for(env_var, default_id):
    raw = os.environ.get(env_var, default_id)
    return _resolve_id(raw)


def _is_set(file_id):
    return bool(file_id)


def _present(dest_abs, kind):
    if kind == "file":
        return os.path.isfile(dest_abs)
    if not os.path.isdir(dest_abs):
        return False
    return any(f != ".gitkeep" for f in os.listdir(dest_abs))


def _progress_bar(current, total, width=40):
    if total > 0:
        filled = int(width * current / total)
        bar = "█" * filled + "░" * (width - filled)
        pct = 100 * current / total
        sys.stdout.write(f"\r    [{bar}] {pct:5.1f}%")
        sys.stdout.flush()


def _download(file_id, dest_abs, kind):
    try:
        import gdown
    except ImportError:
        print("  ERROR: gdown is not installed. Run:  pip install gdown", file=sys.stderr)
        sys.exit(1)

    if kind == "file":
        os.makedirs(os.path.dirname(dest_abs), exist_ok=True)
        gdown.download(id=file_id, output=dest_abs, quiet=False)
    else:  # zipdir: download to a temp file, extract into dest
        os.makedirs(dest_abs, exist_ok=True)
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            gdown.download(id=file_id, output=tmp_path, quiet=False)
            print(f"    Extracting to {dest_abs} …")
            with zipfile.ZipFile(tmp_path) as zf:
                zf.extractall(dest_abs)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


def main():
    ap = argparse.ArgumentParser(description="Download the large TCG-AR artifacts.")
    ap.add_argument("--only", choices=sorted({g for g, *_ in MANIFEST}),
                    help="fetch only one group (models / real / background)")
    ap.add_argument("--force", action="store_true",
                    help="re-download even if already present")
    ap.add_argument("--list", action="store_true",
                    help="list the artifacts and exit without downloading")
    args = ap.parse_args()

    print(f"Assets folder: {ASSETS}\n")
    missing = []

    for group, key, env_var, default_id, rel, kind in MANIFEST:
        if args.only and group != args.only:
            continue

        file_id = _file_id_for(env_var, default_id)
        dest_abs = os.path.join(ASSETS, rel)

        if not _is_set(file_id):
            status = "not available yet"
        elif _present(dest_abs, kind) and not args.force:
            status = "already present"
        else:
            status = "will download" if args.list else "downloading…"

        print(f"[{group}/{key}]  {rel}")
        print(f"    {status}")

        if args.list or not _is_set(file_id):
            if not _is_set(file_id):
                missing.append((group, key, env_var))
            continue

        if _present(dest_abs, kind) and not args.force:
            continue

        print(f"    Drive file ID: {file_id}")
        try:
            _download(file_id, dest_abs, kind)
            print("    done ✓")
        except Exception as e:
            print(f"    FAILED: {e}")

    if missing:
        print("\nThe following assets have no URL/ID configured yet:")
        for group, key, env_var in missing:
            print(f"    [{group}/{key}]  set {env_var}=<Drive-share-URL-or-file-ID>")

    print("\nReminder: card images, sprites and JSON databases are built with "
          "`python -m installation.install ...` (see README.md).")


if __name__ == "__main__":
    main()

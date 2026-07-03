"""Delete all cached embedding files from assets/embedding_cache/.

Run as:
    python -m scripts.clear_embedding_cache
"""
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.config import EMBEDDING_CACHE_PATH

files = glob.glob(EMBEDDING_CACHE_PATH + "*.pt")
for f in files:
    os.remove(f)
    print(f"Deleted {f}")
print(f"{len(files)} cache file(s) deleted.")

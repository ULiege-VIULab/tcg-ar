"""Verify the renderer's additive LRU model cache (the 1.2.0 perf fix):
- load_models is ADDITIVE: loading a different path set does not evict the
  previous models (so per-view renders / wait<->battle<->idle role switches
  never re-decode a GIF -- the bug that made SV rendering ~100x too slow);
- a path already cached is not re-decoded;
- the cache honours MODEL_CACHE_MAX (least-recently-used dropped).

Needs the SV animated database built; skips cleanly otherwise.
    python -m tests.render_cache_check
"""
import glob
import os
import sys

import imageio

import inference.render_module as rm
from inference.render_module import Multi_frame_renderer
from core.config import SV_ANIMATED_MODEL_FOLDER

gifs = sorted(glob.glob(os.path.join(SV_ANIMATED_MODEL_FOLDER, "*.gif")))
if len(gifs) < 6:
    print("SKIP: SV animated database not built (need >=6 gifs).")
    sys.exit(0)

# count GIF decodes
_orig = imageio.mimread
_n = {"c": 0}
def _counting(*a, **k):
    _n["c"] += 1
    return _orig(*a, **k)
imageio.mimread = _counting

r = Multi_frame_renderer(number_of_view=1)

# additive: loading B after A keeps A
_n["c"] = 0
r.load_models([gifs[0]])
assert _n["c"] == 1 and gifs[0] in r.pokemon_dict, "first load should decode once"
r.load_models([gifs[1]])
assert _n["c"] == 2, "second distinct path should decode once more"
assert gifs[0] in r.pokemon_dict and gifs[1] in r.pokemon_dict, "A must NOT be evicted by B (additive)"
print("[ok] load_models is additive (no eviction on set mismatch)")

# cached path is not re-decoded
_n["c"] = 0
r.load_models([gifs[0]])
r.load_models([gifs[1], gifs[0]])
assert _n["c"] == 0, f"cached paths re-decoded ({_n['c']})"
print("[ok] cached sprites are never re-decoded")

# LRU cap: shrink the cap and overflow it
rm.MODEL_CACHE_MAX = 3
for g in gifs[:5]:            # touch g0..g4 in order; g0/g1 become oldest
    r.load_models([g])
assert len(r.pokemon_dict) <= 3, f"cache exceeded cap: {len(r.pokemon_dict)}"
assert gifs[4] in r.pokemon_dict and gifs[3] in r.pokemon_dict, "most-recently-used dropped"
assert gifs[0] not in r.pokemon_dict, "least-recently-used not evicted"
print(f"[ok] LRU cap honoured (kept {len(r.pokemon_dict)} <= 3, evicted oldest)")

# parallel lists stayed consistent after the LRU rebuild
n = len(r.pokemon_dict)
assert len(r.pokemon_models) == n == len(r.pokemon_models_side) == len(r.num_frames_in_gif) \
       == len(r.gif_duration) == len(r.current_frame_num) == len(r.time_elapsed), "list/dict desync"
for p, idx in r.pokemon_dict.items():
    assert 0 <= idx < n, "stale index after eviction"
print("[ok] parallel lists reindexed consistently after eviction")

print("\nAll render cache checks passed.")

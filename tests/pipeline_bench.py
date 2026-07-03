"""Headless micro-benchmark of the per-frame CREATION costs (no camera/OBS needed),
to make the cost of producing N shots explicit and to confirm the removed redundant
copy in Shared_frame_buffer.put. The SENDING/encoding cost lives in separate
GStreamer processes and is measured live via PTCG_PROFILE=1 on real hardware.

Run:  python -m tests.pipeline_bench
"""
import copy
import time
import numpy as np

from core.config import FRAMERATE
from inference.caster_module import Caster_renderer, Broadcast_state


class FakeGameState:
    def __init__(self):
        self._cards = [
            {"x": 500, "y": 540, "dex": 25, "id": "swsh1-1"},
            {"x": 700, "y": 700, "dex": 1, "id": "swsh1-3"},
            {"x": 1400, "y": 540, "dex": 4, "id": "swsh1-2"},
        ]
    def get_number_of_card(self): return len(self._cards)
    def get_number_of_pokemon(self, i): return 1
    def is_pokemon_card(self, i): return bool(self._cards[i]["dex"])
    def get_card_location(self, i): return (self._cards[i]["x"], self._cards[i]["y"])
    def get_pokemon_card_id(self, i): return self._cards[i]["id"]


def _bench(fn, iters):
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return 1000.0 * (time.perf_counter() - t0) / iters


def main():
    frame = (np.random.rand(1080, 1920, 3) * 255).astype(np.uint8)

    # 1) Caster overlay render: cold (rebuild panels) vs warm (cached panels).
    r = Caster_renderer()
    gs = FakeGameState()
    bs = Broadcast_state(); bs.stadium = "Worlds"
    db = {"swsh1-1": {"name": "Pikachu", "hp": "60", "types": ["Lightning"]},
          "swsh1-2": {"name": "Charizard", "hp": "170", "types": ["Fire"]},
          "swsh1-3": {"name": "Bulbasaur", "hp": "70", "types": ["Grass"]}}
    t0 = time.perf_counter(); r.render(frame, gs, bs, db, 0); caster_cold = 1000 * (time.perf_counter() - t0)
    caster_warm = _bench(lambda: r.render(frame, gs, bs, db, 0), 30)

    # 2) The shared-memory write: redundant deepcopy (old) vs direct assign (new).
    dst = np.empty_like(frame)
    def old_put():
        dst[:] = copy.deepcopy(frame)
    def new_put():
        dst[:] = frame
    put_old = _bench(old_put, 200)
    put_new = _bench(new_put, 200)

    budget = 1000.0 / FRAMERATE
    print("=== PTCG-AR creation micro-benchmark (per frame) ===")
    print(f"timer budget @ {FRAMERATE} fps      : {budget:6.1f} ms")
    print(f"caster render (cold, rebuild)   : {caster_cold:6.1f} ms")
    print(f"caster render (warm, cached)    : {caster_warm:6.1f} ms")
    print(f"buffer put  (old: deepcopy)     : {put_old:6.1f} ms / stream")
    print(f"buffer put  (new: direct)       : {put_new:6.1f} ms / stream   (saved {put_old - put_new:.1f} ms)")
    print("Note: AR render_frame + RTSP x264 encoding are measured live with "
          "PTCG_PROFILE=1 (encoding runs in separate processes).")
    print("BENCH_DONE")


if __name__ == "__main__":
    main()

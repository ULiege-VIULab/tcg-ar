"""Verify the caster panel cache: (1) a cached re-render equals a fresh render for
the same state, (2) the cache makes per-frame rendering much cheaper, (3) changing
state invalidates the cache. Headless (no Qt)."""
import time
import numpy as np

from inference.caster_module import Caster_renderer, Broadcast_state


class FakeGameState:
    def __init__(self):
        self._cards = [
            {"x": 500, "y": 540, "n": 1, "id": "swsh1-1"},
            {"x": 700, "y": 700, "n": 1, "id": "swsh1-3"},
            {"x": 1400, "y": 540, "n": 1, "id": "swsh1-2"},
        ]
    def get_number_of_card(self): return len(self._cards)
    def get_number_of_pokemon(self, i): return self._cards[i]["n"]
    def is_pokemon_card(self, i): return self._cards[i]["n"] > 0
    def get_card_location(self, i): return (self._cards[i]["x"], self._cards[i]["y"])
    def get_pokemon_card_id(self, i): return self._cards[i]["id"]


def main():
    r = Caster_renderer()
    gs = FakeGameState()
    bs = Broadcast_state()
    bs.player1["name"] = "Ash"; bs.player2["name"] = "Gary"; bs.stadium = "Worlds"
    db = {
        "swsh1-1": {"name": "Celebi", "hp": "70", "types": ["Grass"]},
        "swsh1-2": {"name": "Charizard", "hp": "170", "types": ["Fire"]},
        "swsh1-3": {"name": "Pikachu", "hp": "60", "types": ["Lightning"]},
    }
    frame = (np.random.rand(1080, 1920, 3) * 255).astype(np.uint8)

    # First render (cold) builds the panel layer; second (warm) reuses it.
    t0 = time.perf_counter(); a = r.render(frame, gs, bs, db, 0); cold = time.perf_counter() - t0
    t0 = time.perf_counter(); b = r.render(frame, gs, bs, db, 0); warm = time.perf_counter() - t0

    # Same frame + same state -> identical output, and warm path is much cheaper.
    assert np.array_equal(a, b), "cached render differs from fresh render"
    assert warm < cold, f"warm ({warm*1e3:.1f}ms) not faster than cold ({cold*1e3:.1f}ms)"
    assert r._base_img is not None and r._base_sig is not None

    # The panel layer is SHARED across views: rendering another view with the same
    # state must reuse the exact same cached base image (no rebuild).
    base_obj = r._base_img
    r.render(frame, gs, bs, db, view_id=1)
    assert r._base_img is base_obj, "panel base was rebuilt for a second view (not shared)"

    # Changing operator state must rebuild the panel (different pixels + new base).
    bs.player1["score"] = 2
    c = r.render(frame, gs, bs, db, 0)
    assert not np.array_equal(b, c), "score change did not invalidate the cache"
    assert r._base_img is not base_obj, "score change did not rebuild the panel base"

    print(f"COLD={cold*1e3:.1f}ms  WARM={warm*1e3:.1f}ms  speedup={cold/warm:.1f}x")
    print("CASTER_CACHE_OK")


if __name__ == "__main__":
    main()

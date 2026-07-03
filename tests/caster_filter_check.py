"""Verify the broadcast view excludes back cards and non-Pokemon (Trainer/Energy)
cards, and that an empty side draws no bench row. Headless (no Qt)."""
import numpy as np

from core.config import BACK_CARD_ID
from inference.caster_module import Caster_renderer, Broadcast_state, board_layout


class FakeGameState:
    """Cards carry (x, y, pokedex-slots, id). A non-Pokemon / back card has all-zero
    pokedex slots (mirrors render_module: noPokemon = zero national Pokedex number)."""
    def __init__(self, cards):
        self._cards = cards
    def get_number_of_card(self): return len(self._cards)
    def get_number_of_pokemon(self, i): return len(self._cards[i]["dex"])
    def is_pokemon_card(self, i): return any(self._cards[i]["dex"])
    def get_card_location(self, i): return (self._cards[i]["x"], self._cards[i]["y"])
    def get_pokemon_card_id(self, i): return self._cards[i]["id"]


def main():
    # Left side: one real Pokemon (active) + one Trainer (should be dropped).
    # Right side: only a back card + an Energy card -> nothing should show.
    cards = [
        {"x": 500, "y": 540, "dex": [25], "id": "swsh1-1"},        # real Pokemon (left active)
        {"x": 300, "y": 700, "dex": [0],  "id": "trainer-9"},      # non-Pokemon (Trainer)
        {"x": 1400, "y": 540, "dex": [0], "id": BACK_CARD_ID},      # card back (right)
        {"x": 1600, "y": 700, "dex": [0], "id": "energy-1"},        # non-Pokemon (Energy)
    ]
    gs = FakeGameState(list(cards))

    layout = board_layout(gs, side_swap=False)
    # Left: only the real Pokemon survives -> it's the active, no bench.
    assert layout["player1"]["active"] is not None
    assert layout["player1"]["active"]["card_id"] == "swsh1-1"
    assert layout["player1"]["bench"] == []
    # Right: back + energy filtered out -> empty side.
    assert layout["player2"]["active"] is None
    assert layout["player2"]["bench"] == []

    # Now add two more real Pokemon on the left so a bench exists (3 reals total
    # on the left -> 1 active + 2 bench; the Trainer is still filtered out).
    cards.append({"x": 600, "y": 800, "dex": [1], "id": "swsh1-2"})
    cards.append({"x": 800, "y": 800, "dex": [4], "id": "swsh1-3"})
    layout2 = board_layout(FakeGameState(cards), side_swap=False)
    real_ids = {"swsh1-1", "swsh1-2", "swsh1-3"}
    bench_ids = [b["card_id"] for b in layout2["player1"]["bench"]]
    assert layout2["player1"]["active"]["card_id"] in real_ids
    assert len(bench_ids) == 2 and set(bench_ids) <= real_ids
    assert "trainer-9" not in bench_ids and "energy-1" not in bench_ids

    # Render must not crash and must differ between empty-bench and benched states.
    r = Caster_renderer()
    bs = Broadcast_state()
    db = {"swsh1-1": {"name": "Pikachu", "hp": "60", "types": ["Lightning"]},
          "swsh1-2": {"name": "Bulbasaur", "hp": "70", "types": ["Grass"]},
          "swsh1-3": {"name": "Charmander", "hp": "70", "types": ["Fire"]}}
    frame = (np.random.rand(1080, 1920, 3) * 255).astype(np.uint8)
    img_empty = r.render(frame, gs, bs, db, view_id=0)            # right side empty, left no bench
    img_bench = r.render(frame, FakeGameState(list(cards)), bs, db, view_id=1)  # left now has bench
    assert img_empty.shape == (1080, 1920, 3)
    assert not np.array_equal(img_empty, img_bench)

    print("CASTER_FILTER_OK")


if __name__ == "__main__":
    main()

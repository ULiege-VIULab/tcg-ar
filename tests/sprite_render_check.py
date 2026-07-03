"""Verify render_frame sprite changes (apply to every view, identity homography here):
(1) painter's order -- a sprite lower in the image (nearer the camera) is drawn over
    one higher up (further);
(2) left-board sprites are mirrored horizontally so the players face each other.
Headless: a minimal Multi_frame_renderer is populated with fake models (no GIFs)."""
import numpy as np

from core.config import WIDTH, HEIGHT
from inference.render_module import Multi_frame_renderer


def make_renderer(models):
    r = Multi_frame_renderer(number_of_view=1)
    for k, m in enumerate(models):
        r.pokemon_dict[chr(ord("A") + k)] = k
        r.pokemon_original_models.append(m)
        r.pokemon_models.append(m)
        r.gif_duration.append(None)
        r.num_frames_in_gif.append(1)
        r.current_frame_num.append(np.zeros((1,), dtype=np.int32))
        r.time_elapsed.append(0)
    return r


class FakeGS:
    """cards: list of (path, x_col, y_row)."""
    def __init__(self, cards): self.cards = cards
    def get_number_of_card(self): return len(self.cards)
    def get_card_location(self, i): return (self.cards[i][1], self.cards[i][2])
    def get_pokemon_path(self, i): return [self.cards[i][0]]


def solid(color, h=100, w=100):
    m = np.zeros((1, h, w, 4), np.uint8)
    m[0, :, :, 0:3] = color
    m[0, :, :, 3] = 255
    return m


def main():
    H = np.eye(3)

    # ---- (1) painter's order ----
    RED, BLUE = (0, 0, 255), (255, 0, 0)
    r = make_renderer([solid(RED), solid(BLUE)])
    # A (red) higher up (row 300), B (blue) lower (row 360); same column -> overlap.
    gs = FakeGS([("A", 500, 300), ("B", 500, 360)])
    frame = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    out = r.render_frame(0, frame, gs, H)
    # overlap rows ~310-350, col 500: the NEARER (lower) sprite B must win.
    assert tuple(int(v) for v in out[330, 500]) == BLUE, "nearer sprite not drawn on top"
    # a row only A covers (e.g. 270) stays red
    assert tuple(int(v) for v in out[270, 500]) == RED

    # ---- (2) left-side mirror ----
    # Asymmetric model: left columns blue, right columns red.
    asym = np.zeros((1, 40, 40, 4), np.uint8)
    asym[0, :, 0:20, 0:3] = BLUE
    asym[0, :, 20:40, 0:3] = RED
    asym[0, :, :, 3] = 255
    r2 = make_renderer([asym])

    # Card on the LEFT (x=300<960) -> flipped: sprite's left half shows red, right blue.
    left = r2.render_frame(0, np.zeros((HEIGHT, WIDTH, 3), np.uint8), FakeGS([("A", 300, 300)]), H)
    assert tuple(int(v) for v in left[300, 285]) == RED   # sprite left half (cols 280-299)
    assert tuple(int(v) for v in left[300, 310]) == BLUE  # sprite right half (cols 300-319)

    # Card on the RIGHT (x=1200>=960) -> not flipped: left half blue, right half red.
    right = r2.render_frame(0, np.zeros((HEIGHT, WIDTH, 3), np.uint8), FakeGS([("A", 1200, 300)]), H)
    assert tuple(int(v) for v in right[300, 1185]) == BLUE
    assert tuple(int(v) for v in right[300, 1210]) == RED

    # ---- (3) bottom-anchor (side views): sprite bottom sits on the card ----
    # A 100-tall solid sprite at card row 300. Centred (zenithal) -> rows ~250..350.
    # Bottom-anchored (side) -> shifted up by h/2 -> rows ~200..300 (bottom edge at card).
    r3 = make_renderer([solid(GREEN := (0, 255, 0))])
    gs1 = FakeGS([("A", 700, 300)])   # right side, no mirror effect on a solid sprite
    centred = r3.render_frame(0, np.zeros((HEIGHT, WIDTH, 3), np.uint8), gs1, H, bottom_anchor=False)
    anchored = r3.render_frame(0, np.zeros((HEIGHT, WIDTH, 3), np.uint8), gs1, H, bottom_anchor=True)

    def covered_rows(img, col=700):
        ys = np.where(np.all(img[:, col] == GREEN, axis=1))[0]
        return int(ys.min()), int(ys.max())
    c_top, c_bot = covered_rows(centred)
    a_top, a_bot = covered_rows(anchored)
    assert abs(((c_top + c_bot) // 2) - 300) <= 2          # centred: middle on the card
    assert abs(a_bot - 300) <= 2                            # anchored: bottom edge on the card
    assert a_top < c_top and a_bot < c_bot                  # anchored sprite sits higher
    # side view renders smaller: ~150% / 200% = 0.75x the zenithal height
    ratio = (a_bot - a_top + 1) / (c_bot - c_top + 1)
    assert abs(ratio - 0.75) < 0.05, f"side/zenith height ratio {ratio:.2f} != 0.75"

    print("SPRITE_RENDER_OK")


if __name__ == "__main__":
    main()

"""
Broadcast / "caster" overlay renderer.

Produces an official-stream-style 1920x1080 frame: the camera (AR) view in the
centre, and a panel for each player on the left/right showing the ACTIVE Pokemon's
card + name + HP bar + ability/attacks, with up to 5 BENCH cards below.  Card art
and stats come from the recognised cards (``card_database`` = pokemon_card.json);
the un-perceivable info (player names/countries/scores, stadium, each active's
current HP and status) comes from a ``Broadcast_state`` filled in the GUI.

Drawn with PIL for clean text/rounded panels, then returned as a BGR numpy frame
so it flows through the normal RTSP output pipeline.
"""

import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from core.config import (WIDTH, HEIGHT, POKEMON_CARD_DATABASE_FOLDER_PATH,
                         ENERGY_TYPE_COLORS, DEFAULT_ENERGY_COLOR, UI_ASSETS_FOLDER,
                         BACK_CARD_ID)

PANEL_W = 480
CENTER_X0, CENTER_X1 = PANEL_W, WIDTH - PANEL_W
CENTER_W = CENTER_X1 - CENTER_X0
MAX_BENCH = 5

# Colours (RGB).
BG = (18, 24, 36)
PANEL_TOP = (44, 96, 170)
PANEL_BOTTOM = (20, 44, 92)
WHITE = (245, 248, 252)
SUBTLE = (180, 198, 220)
HP_BG = (30, 40, 55)


# --------------------------------------------------------------------------- #
# Board logic
# --------------------------------------------------------------------------- #
def board_layout(game_state, side_swap=False, center=(WIDTH // 2, HEIGHT // 2), max_bench=MAX_BENCH):
    """Split the Pokemon cards on the board into the two players' active + bench.

    Left half of the board (x < center) is one physical side, right half the other;
    per side the card nearest the board centre is the active, the rest are bench
    (capped, ordered left->right).  ``side_swap`` decides which physical side maps to
    player1 (left panel).  Non-Pokemon cards are ignored here."""
    left, right = [], []
    for i in range(game_state.get_number_of_card()):
        # Only real Pokemon belong in the broadcast active/bench sets. Trainer/Energy
        # cards (which keep a zero-Pokedex Pokemon slot) and the card back are skipped
        # here -- they still appear in the GUI side panel.
        if not game_state.is_pokemon_card(i) or game_state.get_pokemon_card_id(i) == BACK_CARD_ID:
            continue
        x, y = game_state.get_card_location(i)
        item = {"index": i, "card_id": game_state.get_pokemon_card_id(i), "x": x, "y": y,
                "dist": (x - center[0]) ** 2 + (y - center[1]) ** 2}
        (left if x < center[0] else right).append(item)

    def split(cards):
        if not cards:
            return None, []
        cards = sorted(cards, key=lambda c: c["dist"])
        return cards[0], sorted(cards[1:], key=lambda c: c["x"])[:max_bench]

    la, lb = split(left)
    ra, rb = split(right)
    p1 = {"active": la, "bench": lb}
    p2 = {"active": ra, "bench": rb}
    return {"player1": p2, "player2": p1} if side_swap else {"player1": p1, "player2": p2}


class Broadcast_state:
    """Operator-entered info that the system cannot perceive.  ``active['left']`` is
    the left panel (player1), ``active['right']`` the right panel (player2)."""

    def __init__(self):
        self.player1 = {"name": "Player 1", "country": "", "score": 0}
        self.player2 = {"name": "Player 2", "country": "", "score": 0}
        self.stadium = ""
        self.side_swap = False
        self.active = {"left": {"hp": None, "status": ""}, "right": {"hp": None, "status": ""}}


# --------------------------------------------------------------------------- #
# Renderer
# --------------------------------------------------------------------------- #
def _font(size, bold=True):
    names = (("segoeuib.ttf", "arialbd.ttf") if bold else ("segoeui.ttf", "arial.ttf"))
    for n in names:
        try:
            return ImageFont.truetype(n, size)
        except Exception:
            continue
    return ImageFont.load_default()


class Caster_renderer:
    def __init__(self):
        self.f_big = _font(34)
        self.f_name = _font(26)
        self.f_med = _font(20)
        self.f_small = _font(16)
        self.f_tiny = _font(13, bold=False)
        self._card_cache = {}
        self._energy = {}
        for t in ENERGY_TYPE_COLORS:
            p = os.path.join(UI_ASSETS_FOLDER, "energy_" + t + ".png")
            if os.path.exists(p):
                self._energy[t] = Image.open(p).convert("RGBA").resize((22, 22), Image.LANCZOS)
        # Cache of the rendered side-panel layer (everything except the live camera
        # feed and stadium banner). It depends only on the board + operator info, NOT
        # on which camera view we are drawing, so a single base is SHARED across all
        # views and rebuilt only when the state signature changes (once per change,
        # not once per view per frame).
        self._base_sig = None
        self._base_img = None

    # ---- helpers --------------------------------------------------------------
    def _card_image(self, card_id):
        if card_id in self._card_cache:
            return self._card_cache[card_id]
        img = None
        path = POKEMON_CARD_DATABASE_FOLDER_PATH + str(card_id) + ".jpg"
        if os.path.exists(path):
            bgr = cv2.imread(path)
            if bgr is not None:
                img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        self._card_cache[card_id] = img
        return img

    @staticmethod
    def _meta(card_database, card_id):
        return card_database.get(card_id, {}) if card_database else {}

    @staticmethod
    def _type_color(meta):
        types = meta.get("types") or []
        return ENERGY_TYPE_COLORS.get(types[0], DEFAULT_ENERGY_COLOR) if types else DEFAULT_ENERGY_COLOR

    def _paste_card(self, base, card_id, box):
        """Fit the card image into box (x0,y0,w,h), keeping aspect, centred."""
        x0, y0, w, h = box
        img = self._card_image(card_id)
        if img is None:
            d = ImageDraw.Draw(base)
            d.rounded_rectangle([x0, y0, x0 + w, y0 + h], radius=8, fill=(60, 70, 90))
            self._text(base, str(card_id), self.f_small, WHITE, (x0 + w // 2, y0 + h // 2), anchor="mm")
            return
        scale = min(w / img.width, h / img.height)
        nw, nh = int(img.width * scale), int(img.height * scale)
        img = img.resize((nw, nh), Image.LANCZOS).convert("RGBA")
        base.paste(img, (x0 + (w - nw) // 2, y0 + (h - nh) // 2), img)

    def _text(self, base, text, font, fill, pos, anchor="la"):
        ImageDraw.Draw(base).text(pos, text, font=font, fill=fill, anchor=anchor)

    def _hp_bar(self, base, x, y, w, current, maximum, color):
        d = ImageDraw.Draw(base)
        h = 26
        d.rounded_rectangle([x, y, x + w, y + h], radius=h // 2, fill=HP_BG)
        if maximum and maximum > 0:
            frac = max(0.0, min(1.0, current / maximum))
            if frac > 0:
                d.rounded_rectangle([x, y, x + int(w * frac), y + h], radius=h // 2, fill=color)
        label = f"{current}/{maximum}" if maximum else "—"
        self._text(base, label, self.f_small, WHITE, (x + w // 2, y + h // 2), anchor="mm")

    def _energy_dots(self, base, x, y, cost):
        for k, t in enumerate(cost[:6]):
            icon = self._energy.get(t)
            if icon is not None:
                base.paste(icon, (x + k * 24, y), icon)
        return x + len(cost[:6]) * 24

    # ---- panels ---------------------------------------------------------------
    def _panel_bg(self, base, x0):
        # Vertical blue gradient panel.
        ramp = np.linspace(0, 1, HEIGHT)[:, None]
        col = (np.array(PANEL_TOP) * (1 - ramp) + np.array(PANEL_BOTTOM) * ramp).astype(np.uint8)
        grad = Image.fromarray(np.repeat(col[:, None, :], PANEL_W, axis=1))
        base.paste(grad, (x0, 0))

    def _draw_player(self, base, side, layout, pinfo, active_override, card_database):
        x0 = 0 if side == "left" else CENTER_X1
        pad = 24
        d = ImageDraw.Draw(base)

        # Header: name + country + score
        self._text(base, pinfo.get("name") or "—", self.f_name, WHITE, (x0 + pad, 22))
        country = pinfo.get("country") or ""
        self._text(base, country, self.f_small, SUBTLE, (x0 + pad, 58))
        d.rounded_rectangle([x0 + PANEL_W - 86, 18, x0 + PANEL_W - pad, 74], radius=10, fill=(12, 28, 60))
        self._text(base, str(pinfo.get("score", 0)), self.f_big, WHITE,
                   (x0 + PANEL_W - 55, 46), anchor="mm")
        d.line([x0 + pad, 86, x0 + PANEL_W - pad, 86], fill=(90, 130, 190), width=2)

        active = layout.get("active")
        if active:
            meta = self._meta(card_database, active["card_id"])
            color = self._type_color(meta)
            self._paste_card(base, active["card_id"], (x0 + pad, 98, PANEL_W - 2 * pad, 300))
            name = meta.get("name") or active["card_id"]
            self._text(base, name, self.f_name, WHITE, (x0 + pad, 406))
            try:
                maximum = int(meta.get("hp")) if meta.get("hp") else 0
            except (TypeError, ValueError):
                maximum = 0
            current = active_override.get("hp")
            current = maximum if current is None else current
            self._hp_bar(base, x0 + pad, 444, PANEL_W - 2 * pad, current, maximum, color)
            status = active_override.get("status") or ""
            if status:
                d.rounded_rectangle([x0 + pad, 478, x0 + pad + 12 + 9 * len(status), 504], radius=8, fill=(150, 60, 60))
                self._text(base, status, self.f_small, WHITE, (x0 + pad + 8, 481))

            yy = 514
            for ab in (meta.get("abilities") or [])[:1]:
                d.rounded_rectangle([x0 + pad, yy, x0 + pad + 70, yy + 22], radius=6, fill=(190, 70, 70))
                self._text(base, "ABILITY", self.f_tiny, WHITE, (x0 + pad + 8, yy + 4))
                self._text(base, ab.get("name", ""), self.f_small, WHITE, (x0 + pad + 80, yy + 2))
                yy += 30
            for atk in (meta.get("attacks") or [])[:3]:
                xx = self._energy_dots(base, x0 + pad, yy, atk.get("cost") or [])
                self._text(base, atk.get("name", ""), self.f_small, WHITE, (xx + 6, yy + 2))
                if atk.get("damage"):
                    self._text(base, str(atk["damage"]), self.f_med, WHITE, (x0 + PANEL_W - pad, yy), anchor="ra")
                yy += 30

        # Bench -- only drawn when there is at least one benched Pokemon.
        bench = layout.get("bench", [])
        if not bench:
            return
        by = 770
        d.line([x0 + pad, by - 14, x0 + PANEL_W - pad, by - 14], fill=(90, 130, 190), width=2)
        self._text(base, "BENCH", self.f_small, SUBTLE, (x0 + pad, by - 36))
        for b in bench[:MAX_BENCH]:
            meta = self._meta(card_database, b["card_id"])
            self._paste_card(base, b["card_id"], (x0 + pad, by, 56, 56))
            nm = meta.get("name") or b["card_id"]
            self._text(base, nm[:22], self.f_small, WHITE, (x0 + pad + 66, by + 6))
            try:
                mx = int(meta.get("hp")) if meta.get("hp") else 0
            except (TypeError, ValueError):
                mx = 0
            self._hp_bar(base, x0 + pad + 66, by + 30, PANEL_W - 2 * pad - 66, mx, mx, self._type_color(meta))
            by += 60

    # ---- caching --------------------------------------------------------------
    @staticmethod
    def _signature(layout, bs):
        """Hashable summary of everything the side-panel layer depends on. When it is
        unchanged we reuse the cached panel image instead of redrawing it."""
        def pl_sig(p):
            a = p["active"]["card_id"] if p["active"] else None
            return (a, tuple(b["card_id"] for b in p["bench"]))
        return (
            bs.stadium, bool(bs.side_swap),
            bs.player1.get("name"), bs.player1.get("country"), bs.player1.get("score"),
            bs.player2.get("name"), bs.player2.get("country"), bs.player2.get("score"),
            bs.active["left"]["hp"], bs.active["left"]["status"],
            bs.active["right"]["hp"], bs.active["right"]["status"],
            pl_sig(layout["player1"]), pl_sig(layout["player2"]),
        )

    def _build_base(self, layout, broadcast_state, card_database):
        """The static layer: blue gradient side panels + both players' active/bench
        info. Excludes the live camera feed and stadium banner (drawn per frame)."""
        base = Image.new("RGB", (WIDTH, HEIGHT), BG)
        self._panel_bg(base, 0)
        self._panel_bg(base, CENTER_X1)
        self._draw_player(base, "left", layout["player1"], broadcast_state.player1,
                          broadcast_state.active["left"], card_database)
        self._draw_player(base, "right", layout["player2"], broadcast_state.player2,
                          broadcast_state.active["right"], card_database)
        return base

    # ---- main -----------------------------------------------------------------
    def render(self, camera_frame, game_state, broadcast_state, card_database, view_id=0):
        layout = board_layout(game_state, broadcast_state.side_swap)
        sig = self._signature(layout, broadcast_state)
        # The panel layer is identical for every camera view, so it is built once and
        # shared: only the first view after a state change pays the rebuild cost.
        if sig != self._base_sig or self._base_img is None:
            self._base_img = self._build_base(layout, broadcast_state, card_database)
            self._base_sig = sig
        base = self._base_img

        # Composite onto a copy so the cached panel layer is preserved.
        out = base.copy()

        # Centre: the camera / AR feed COVERS the centre column (fills its full height
        # and width, centre-cropping any overflow) so there are no black letterbox bars
        # on the stream -- at the cost of losing the cropped edges of the image.
        if camera_frame is not None:
            rgb = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[0], rgb.shape[1]
            scale = max(CENTER_W / w, HEIGHT / h)
            nw, nh = max(CENTER_W, int(round(w * scale))), max(HEIGHT, int(round(h * scale)))
            feed = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
            cx0 = (nw - CENTER_W) // 2
            cy0 = (nh - HEIGHT) // 2
            feed = feed[cy0:cy0 + HEIGHT, cx0:cx0 + CENTER_W]
            out.paste(Image.fromarray(feed), (CENTER_X0, 0))

        # Stadium banner (top centre, on top of the feed).
        if broadcast_state.stadium:
            d = ImageDraw.Draw(out)
            tw = 18 + 14 * len(broadcast_state.stadium)
            cx = WIDTH // 2
            d.rounded_rectangle([cx - tw // 2, 16, cx + tw // 2, 58], radius=12, fill=(20, 40, 80))
            self._text(out, broadcast_state.stadium.upper(), self.f_med, WHITE, (cx, 37), anchor="mm")

        return cv2.cvtColor(np.asarray(out), cv2.COLOR_RGB2BGR)

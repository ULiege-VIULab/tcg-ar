"""
Generate the UI icon assets the GUI references (they were missing) and small
energy-type icons for the broadcast overlay, into ``UI_ASSETS_FOLDER``.

Run once:  python -m inference.generate_ui_assets
Everything is drawn with PIL so the assets are reproducible.
"""

import os

from PIL import Image, ImageDraw, ImageFont

from core.config import UI_ASSETS_FOLDER, ENERGY_TYPE_COLORS

S = 128                                  # icon canvas size
BRAND = (38, 110, 200, 255)              # brand blue
BRAND_DARK = (24, 70, 140, 255)
WHITE = (255, 255, 255, 255)
INK = (40, 48, 60, 255)


def _canvas():
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    return img, ImageDraw.Draw(img)


def _font(size):
    for name in ("seguibl.ttf", "segoeuib.ttf", "arialbd.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _centered(draw, text, font, fill, cx=S // 2, cy=S // 2):
    l, t, r, b = draw.textbbox((0, 0), text, font=font)
    draw.text((cx - (r - l) / 2 - l, cy - (b - t) / 2 - t), text, font=font, fill=fill)


def gear():
    img, d = _canvas()
    import math
    cx = cy = S // 2
    R, r = 46, 30
    pts = []
    teeth = 8
    for i in range(teeth * 2):
        ang = math.pi * i / teeth
        rad = R if i % 2 == 0 else R - 14
        pts.append((cx + rad * math.cos(ang), cy + rad * math.sin(ang)))
    d.polygon(pts, fill=BRAND)
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=BRAND)
    d.ellipse([cx - 14, cy - 14, cx + 14, cy + 14], fill=(0, 0, 0, 0))
    d.ellipse([cx - 14, cy - 14, cx + 14, cy + 14], outline=WHITE, width=6)
    return img


def exit_icon():
    img, d = _canvas()
    d.rounded_rectangle([18, 16, 70, S - 16], radius=10, fill=BRAND)
    d.rounded_rectangle([30, 28, 70, S - 28], radius=6, fill=(0, 0, 0, 0))
    # arrow out
    d.line([58, S // 2, 108, S // 2], fill=BRAND_DARK, width=12)
    d.polygon([(96, S // 2 - 20), (118, S // 2), (96, S // 2 + 20)], fill=BRAND_DARK)
    return img


def help_icon():
    img, d = _canvas()
    d.ellipse([10, 10, S - 10, S - 10], fill=BRAND)
    _centered(d, "?", _font(82), WHITE)
    return img


def toggle_icon():
    img, d = _canvas()
    d.rounded_rectangle([14, 40, S - 14, S - 40], radius=24, fill=BRAND)
    d.ellipse([S - 70, 38, S - 16, 90], fill=WHITE)
    return img


def side_panel():
    img, d = _canvas()
    d.rounded_rectangle([14, 18, S - 14, S - 18], radius=12, fill=BRAND, outline=BRAND_DARK, width=4)
    d.rectangle([S - 56, 18, S - 14, S - 18], fill=WHITE)
    for y in (34, 56, 78):
        d.rectangle([26, y, S - 66, y + 10], fill=WHITE)
    return img


def deck():
    img, d = _canvas()
    for k, off in enumerate((18, 10, 2)):
        col = (BRAND if k == 2 else BRAND_DARK)
        d.rounded_rectangle([24 + off, 14 + off, 24 + off + 64, 14 + off + 90], radius=10,
                            fill=col, outline=WHITE, width=4)
    return img


def broadcast():
    img, d = _canvas()
    cx, cy = S // 2, S // 2 + 18
    d.ellipse([cx - 16, cy - 16, cx + 16, cy + 16], fill=BRAND)
    import math
    for rad, w in ((30, 8), (48, 8)):
        for sgn in (-1, 1):
            box = [cx - rad, cy - rad, cx + rad, cy + rad]
            start = 200 if sgn < 0 else -20
            d.arc(box, start, start + 40, fill=BRAND, width=w)
    return img


def energy_icon(type_name, color):
    sz = 64
    img = Image.new("RGBA", (sz, sz), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse([3, 3, sz - 3, sz - 3], fill=(color[0], color[1], color[2], 255), outline=WHITE, width=4)
    _centered(d, type_name[0], _font(34), WHITE, sz // 2, sz // 2)
    return img


# Note: the application icon (AppIcon.png, the TCG-AR brand mark) is a
# committed asset in this folder, NOT generated here - do not overwrite it.
ICONS = {
    "Parameter": gear, "Exit": exit_icon, "Help": help_icon,
    "EnableDisable": toggle_icon, "SidePanel": side_panel, "Deck": deck, "Broadcast": broadcast,
}


def generate():
    os.makedirs(UI_ASSETS_FOLDER, exist_ok=True)
    for name, fn in ICONS.items():
        fn().save(os.path.join(UI_ASSETS_FOLDER, name + ".png"))
    for type_name, color in ENERGY_TYPE_COLORS.items():
        energy_icon(type_name, color).save(os.path.join(UI_ASSETS_FOLDER, "energy_" + type_name + ".png"))
    print(f"Wrote {len(ICONS)} icons + {len(ENERGY_TYPE_COLORS)} energy icons to {UI_ASSETS_FOLDER}")


if __name__ == "__main__":
    generate()

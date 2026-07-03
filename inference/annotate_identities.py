"""
Identity annotation tool for the real test set.

For every detection box in ``assets/AI database/real/`` it crops the card, uses the
base ArcFace model to propose the ~10 nearest cards (auto-resolving the 180-degree
flip), and lets you confirm one by clicking its image.  If the right card is not
proposed, a search panel (card id / name / number / set, like the deck page) lets
you pick any card from the database by image.  A "Flip 180" toggle and a
"Skip / Unknown" button are provided.

Annotations are stored DOTA-like in ``real/annotations_identity/<i>.txt`` and the
tool resumes from the first un-annotated image.

Run:  python -m inference.annotate_identities
"""

import sys
import json

import cv2
import torch
import numpy as np
from PySide6 import QtCore, QtWidgets, QtGui

from core.config import (POKEMON_CARD_DATABASE_FOLDER_PATH, POKEMON_CARD_DATABASE_FILE,
                         IDENTIFICATION_IMAGE_SIZE)
from core.transforms import get_inference_transform
from core.models import identification as idm
import core.real_data as rd
from inference.user_interface import Card_search_bar

PROPOSALS = 10
TILE_H = 150


def card_image_path(card_id):
    return POKEMON_CARD_DATABASE_FOLDER_PATH + card_id + ".jpg"


def bgr_to_pixmap(bgr, height=TILE_H):
    h, w = bgr.shape[:2]
    width = max(1, int(w / h * height))
    img = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA)
    qimg = QtGui.QImage(img.data, img.shape[1], img.shape[0], 3 * img.shape[1],
                        QtGui.QImage.Format_RGB888).rgbSwapped()
    return QtGui.QPixmap.fromImage(qimg.copy())


class CardTile(QtWidgets.QFrame):
    """Clickable card image + id label."""

    def __init__(self, parent, card_id, on_click, subtitle=None):
        super().__init__(parent)
        self.card_id = card_id
        self.on_click = on_click
        self.setFrameShape(QtWidgets.QFrame.Box)
        self.setCursor(QtCore.Qt.PointingHandCursor)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(3, 3, 3, 3)

        pic = QtWidgets.QLabel(self)
        img = cv2.imread(card_image_path(card_id))
        if img is not None:
            pic.setPixmap(bgr_to_pixmap(img))
        else:
            pic.setText("(no image)")
        pic.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(pic)

        text = card_id if subtitle is None else f"{card_id}\n{subtitle}"
        label = QtWidgets.QLabel(text, self)
        label.setAlignment(QtCore.Qt.AlignCenter)
        label.setFont(QtGui.QFont("Times", 9))
        layout.addWidget(label)

    def mousePressEvent(self, event):
        self.on_click(self.card_id)


class SearchDialog(QtWidgets.QDialog):
    """Search the whole card database by id/name, number, or set and pick one."""

    MAX_RESULTS = 60

    def __init__(self, parent, card_meta):
        super().__init__(parent)
        self.setWindowTitle("Search the card database")
        self.resize(900, 700)
        self.card_meta = card_meta
        self.selected_id = None

        layout = QtWidgets.QVBoxLayout(self)
        self.search_bar = Card_search_bar(self, "Find a card:")
        self.search_bar.search_name_text.textChanged.connect(self.refresh)
        self.search_bar.search_number_text.textChanged.connect(self.refresh)
        self.search_bar.scroll_list_extension.currentIndexChanged.connect(self.refresh)
        layout.addWidget(self.search_bar)

        self.info = QtWidgets.QLabel("", self)
        layout.addWidget(self.info)

        self.scroll = QtWidgets.QScrollArea(self)
        self.scroll.setWidgetResizable(True)
        self.grid_host = QtWidgets.QWidget()
        self.grid = QtWidgets.QGridLayout(self.grid_host)
        self.scroll.setWidget(self.grid_host)
        layout.addWidget(self.scroll)

        self.refresh()

    def _matches(self):
        name = self.search_bar.search_name_text.text().lower().strip()
        number = self.search_bar.search_number_text.text().lower().strip()
        set_id = self.search_bar.scroll_list_extension.currentData()
        out = []
        for cid, meta in self.card_meta.items():
            if name and name not in cid.lower() and name not in meta["name"].lower():
                continue
            if number and number != meta["number"].lower():
                continue
            if set_id and set_id.lower() != meta["set"].lower():
                continue
            out.append(cid)
            if len(out) > self.MAX_RESULTS:
                break
        return out

    def refresh(self):
        while self.grid.count():
            w = self.grid.takeAt(0).widget()
            if w:
                w.deleteLater()
        matches = self._matches()
        self.info.setText(f"{len(matches)} shown (max {self.MAX_RESULTS}; refine the search)")
        for k, cid in enumerate(matches):
            tile = CardTile(self.grid_host, cid, self._pick, subtitle=self.card_meta[cid]["name"])
            self.grid.addWidget(tile, k // 6, k % 6)

    def _pick(self, card_id):
        self.selected_id = card_id
        self.accept()


class AnnotatorWindow(QtWidgets.QMainWindow):
    def __init__(self, model, anchors, device, transform, card_meta):
        super().__init__()
        self.model = model
        self.anchors = anchors
        self.device = device
        self.transform = transform
        self.card_meta = card_meta
        self.setWindowTitle("PTCG-AR — real-set identity annotation")
        self.resize(1300, 850)

        self.indices = rd.real_image_indices()
        # Resume at first un-annotated image.
        self.img_pos = next((k for k, i in enumerate(self.indices) if not rd.is_annotated(i)), 0)
        self.box_pos = 0
        self.image = None
        self.boxes = []
        self.entries = []

        self._build_ui()
        self._load_image()

    # --- UI scaffold ---------------------------------------------------------
    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)

        # Left: current crop + controls
        left = QtWidgets.QVBoxLayout()
        self.header = QtWidgets.QLabel("")
        self.header.setFont(QtGui.QFont("Times", 14, QtGui.QFont.Bold))
        left.addWidget(self.header)
        self.crop_label = QtWidgets.QLabel()
        self.crop_label.setAlignment(QtCore.Qt.AlignCenter)
        self.crop_label.setMinimumSize(260, 360)
        self.crop_label.setFrameShape(QtWidgets.QFrame.Box)
        left.addWidget(self.crop_label)
        self.assigned_label = QtWidgets.QLabel("")
        left.addWidget(self.assigned_label)

        for text, slot in [("Flip 180°", self.flip_crop), ("Search database…", self.open_search),
                           ("Skip / Unknown", self.skip_box)]:
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(slot)
            left.addWidget(b)

        nav = QtWidgets.QHBoxLayout()
        for text, slot in [("◀ Prev box", self.prev_box), ("Next box ▶", self.next_box)]:
            b = QtWidgets.QPushButton(text); b.clicked.connect(slot); nav.addWidget(b)
        left.addLayout(nav)
        nav2 = QtWidgets.QHBoxLayout()
        for text, slot in [("◀◀ Prev image", self.prev_image), ("Save & next image ▶▶", self.next_image)]:
            b = QtWidgets.QPushButton(text); b.clicked.connect(slot); nav2.addWidget(b)
        left.addLayout(nav2)
        left.addStretch(1)
        root.addLayout(left, 0)

        # Right: proposals grid + whole-image preview
        right = QtWidgets.QVBoxLayout()
        right.addWidget(QtWidgets.QLabel("ArcFace proposals — click the correct card:"))
        self.prop_host = QtWidgets.QWidget()
        self.prop_grid = QtWidgets.QGridLayout(self.prop_host)
        right.addWidget(self.prop_host)
        right.addWidget(QtWidgets.QLabel("Full image (current box highlighted):"))
        self.full_label = QtWidgets.QLabel()
        self.full_label.setAlignment(QtCore.Qt.AlignCenter)
        right.addWidget(self.full_label, 1)
        root.addLayout(right, 1)

    # --- data flow -----------------------------------------------------------
    def _current_index(self):
        return self.indices[self.img_pos]

    def _load_image(self):
        i = self._current_index()
        self.image = rd.load_real_image(i)
        self.boxes = rd.read_dota_boxes(i)
        existing = {tuple(e["poly"]): e for e in rd.read_identity_annotations(i)}
        self.entries = []
        for poly in self.boxes:
            prev = existing.get(tuple(poly))
            self.entries.append({"poly": poly, "card_id": prev["card_id"] if prev else None,
                                 "orientation": prev["orientation"] if prev else 0,
                                 "proposed": prev is not None})
        self.box_pos = 0
        self._show_box()

    def _show_box(self):
        i = self._current_index()
        n = len(self.boxes)
        annotated = sum(1 for e in self.entries if e["card_id"] is not None)
        self.header.setText(f"Image {i}  ({self.img_pos+1}/{len(self.indices)})   "
                            f"box {self.box_pos+1}/{n}   [{annotated}/{n} done]")
        if n == 0:
            self.crop_label.setText("(no detections)")
            self._render_full()
            return
        entry = self.entries[self.box_pos]
        # Compute / refresh proposals for this box at the current orientation.
        crop = rd.crop_card(self.image, entry["poly"], flip=(entry["orientation"] == 1))
        self.crop_label.setPixmap(bgr_to_pixmap(crop, height=340))
        self.assigned_label.setText(
            "Assigned: " + (entry["card_id"] if entry["card_id"] else "—") + f"   (orientation flip={entry['orientation']})")
        # Auto orientation + proposals only the first time we see a box.
        if not entry.get("proposed"):
            flip, props = idm.best_orientation_topk(
                rd.crop_card(self.image, entry["poly"]), self.anchors, self.model,
                self.device, self.transform, k=PROPOSALS, method="arcface")
            entry["orientation"] = flip
            entry["_props"] = props
            entry["proposed"] = True
            crop = rd.crop_card(self.image, entry["poly"], flip=(entry["orientation"] == 1))
            self.crop_label.setPixmap(bgr_to_pixmap(crop, height=340))
        self._render_proposals(entry.get("_props", []))
        self._render_full()

    def _render_proposals(self, props):
        while self.prop_grid.count():
            w = self.prop_grid.takeAt(0).widget()
            if w:
                w.deleteLater()
        for k, (cid, score) in enumerate(props):
            tile = CardTile(self.prop_host, cid, self.assign_card,
                            subtitle=f"{self.card_meta.get(cid, {}).get('name', '')}  {score:.2f}")
            self.prop_grid.addWidget(tile, k // 5, k % 5)

    def _render_full(self):
        vis = self.image.copy()
        for k, poly in enumerate(self.boxes):
            pts = np.array(poly, np.int32).reshape(4, 2)
            color = (0, 0, 255) if k == self.box_pos else (0, 200, 0)
            cv2.polylines(vis, [pts], True, color, 3)
        self.full_label.setPixmap(bgr_to_pixmap(vis, height=420))

    # --- actions -------------------------------------------------------------
    def assign_card(self, card_id):
        if self.boxes:
            self.entries[self.box_pos]["card_id"] = card_id
        self.next_box()

    def skip_box(self):
        if self.boxes:
            self.entries[self.box_pos]["card_id"] = rd.UNKNOWN_ID
        self.next_box()

    def flip_crop(self):
        if not self.boxes:
            return
        e = self.entries[self.box_pos]
        e["orientation"] ^= 1
        crop = rd.crop_card(self.image, e["poly"], flip=(e["orientation"] == 1))
        self.crop_label.setPixmap(bgr_to_pixmap(crop, height=340))
        # Re-rank proposals at the new orientation.
        e["_props"] = idm.identify_topk(
            torch.stack([self.transform(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))]),
            self.anchors, self.model, self.device, k=PROPOSALS, method="arcface")[0]
        self._render_proposals(e["_props"])
        self.assigned_label.setText(f"Assigned: {e['card_id'] or '—'}   (orientation flip={e['orientation']})")

    def open_search(self):
        dlg = SearchDialog(self, self.card_meta)
        if dlg.exec() and dlg.selected_id:
            self.assign_card(dlg.selected_id)

    def next_box(self):
        if self.box_pos < len(self.boxes) - 1:
            self.box_pos += 1
            self._show_box()
        else:
            self._show_box()  # refresh counts

    def prev_box(self):
        if self.box_pos > 0:
            self.box_pos -= 1
            self._show_box()

    def _save(self):
        i = self._current_index()
        to_write = [{"poly": e["poly"], "card_id": e["card_id"] or rd.UNKNOWN_ID,
                     "orientation": e["orientation"]} for e in self.entries]
        rd.write_identity_annotation(i, to_write)

    def next_image(self):
        self._save()
        if self.img_pos < len(self.indices) - 1:
            self.img_pos += 1
            self._load_image()

    def prev_image(self):
        self._save()
        if self.img_pos > 0:
            self.img_pos -= 1
            self._load_image()

    def closeEvent(self, event):
        self._save()
        event.accept()


def load_card_metadata():
    """Map card_id -> {name, number, set} for the search panel."""
    with open(POKEMON_CARD_DATABASE_FILE, "r", encoding="utf-8") as f:
        raw = json.load(f)
    meta = {}
    for cid, c in raw.items():
        meta[cid] = {"name": c.get("name") or "", "number": str(c.get("number") or ""),
                     "set": (c.get("set") or {}).get("id", "") if isinstance(c.get("set"), dict) else ""}
    # The card back is a real, searchable card.
    from core.config import BACK_CARD_ID
    meta.setdefault(BACK_CARD_ID, {"name": "Card Back", "number": "", "set": ""})
    return meta


def _load_or_build_anchors(model, device):
    """Build the all-cards anchor embeddings, cached next to the weight so later
    launches are instant (loading ~40 MB instead of re-reading 20k card images)."""
    from core.config import model_save_path
    weight = model_save_path("arcface", fine_tuned=False)
    cache = weight + ".anchors.pt"
    try:
        if os.path.exists(cache) and os.path.getmtime(cache) >= os.path.getmtime(weight):
            return torch.load(cache, map_location="cpu", weights_only=False)
    except Exception:
        pass
    anchors = idm.evaluate_anchors(model, device, anchor_list=idm.all_card_anchor_list())
    try:
        torch.save({k: v.cpu() for k, v in anchors.items()}, cache)
    except Exception:
        pass
    return anchors


def main():
    app = QtWidgets.QApplication([])
    pix = QtGui.QPixmap(640, 130); pix.fill(QtGui.QColor("white"))
    splash = QtWidgets.QSplashScreen(pix)
    splash.showMessage("PTCG-AR annotation\n\nLoading ArcFace model and building card anchors\n(one-time per model, then cached)…",
                       QtCore.Qt.AlignmentFlag.AlignCenter, QtGui.QColor("black"))
    splash.show()
    app.processEvents()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = idm.load_model("arcface", fine_tuned=False, device=device)
    app.processEvents()
    anchors = _load_or_build_anchors(model, device)
    app.processEvents()
    transform = get_inference_transform(IDENTIFICATION_IMAGE_SIZE)
    card_meta = load_card_metadata()

    win = AnnotatorWindow(model, anchors, device, transform, card_meta)
    win.show()
    splash.finish(win)
    app.exec()


if __name__ == "__main__":
    main()

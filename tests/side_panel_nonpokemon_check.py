"""Regression test for the side panel showing non-Pokemon / card-back entries
(e.g. 'back1-1') that are absent from the Pokemon card database. Previously this
raised KeyError in Info and could crash on a missing image. Headless Qt."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtWidgets
from inference import user_interface as ui


def main():
    app = QtWidgets.QApplication([])

    # Stand-in for Card_menu: only Pokemon cards are in the database.
    menu = QtWidgets.QWidget()
    menu.card_database = {"swsh1-1": {"name": "Celebi", "hp": "70", "types": ["Grass"]}}
    box = QtWidgets.QWidget(menu)  # stand-in for Card_box (Info uses parent().parent())

    # Non-Pokemon / back card not in the database -> graceful fallback, no KeyError.
    info_missing = ui.Info(box, "back1-1")
    assert "back1-1" in info_missing.name.text()

    # Known Pokemon card -> full info.
    info_known = ui.Info(box, "swsh1-1")
    assert "Celebi" in info_known.name.text()

    # Missing image -> placeholder instead of crashing on None.shape.
    pic = ui.Card_picture(box, None)
    assert pic is not None

    print("SIDE_PANEL_NONPOKEMON_OK")


if __name__ == "__main__":
    main()

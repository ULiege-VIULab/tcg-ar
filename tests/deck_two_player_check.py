"""Headless check of the two-deck (one per player) feature: per-deck named save/load,
non-depleting database via get_selection, and the de-duplicated union written to the
active deck file. Uses a small monkeypatched card database; offscreen Qt."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import json
import tempfile

from PySide6 import QtWidgets
from inference import user_interface as ui


def main():
    tmp = tempfile.mkdtemp()
    cards = ["sv1-1", "sv1-2", "sv1-3", "sv1-4", "sv1-5"]

    # small fake card database (images never read: loading is lazy + offscreen)
    card_dir = os.path.join(tmp, "cards") + os.sep
    os.makedirs(card_dir, exist_ok=True)
    for c in cards:
        open(card_dir + c + ".jpg", "w").close()
    card_json = os.path.join(tmp, "pokemon_card.json")
    with open(card_json, "w", encoding="utf-8") as f:
        json.dump({c: {"name": c, "number": str(i + 1), "set": {"id": "sv1"}}
                   for i, c in enumerate(cards)}, f)
    deck_dir = os.path.join(tmp, "deck list") + os.sep
    os.makedirs(deck_dir, exist_ok=True)
    active_deck = deck_dir + "deck.txt"
    with open(active_deck, "w") as f:                      # P1 starts from this (backward compat)
        f.write("sv1-1\nsv1-2")

    # point the module at the fake data
    ui.POKEMON_CARD_DATABASE_FOLDER_PATH = card_dir
    ui.POKEMON_CARD_DATABASE_FILE = card_json
    ui.DECK_LIST_FOLDER = deck_dir
    ui.IDENTIFICATION_FINE_TUNE_DECK_LIST_FILE = active_deck

    app = QtWidgets.QApplication([])
    win = QtWidgets.QMainWindow()
    dc = ui.Deck_list_creator(win, None)

    # 1) initial load: P1 = deck.txt, P2 empty, database = all cards
    assert sorted(dc.deck_p1.get_ID_list()) == ["sv1-1", "sv1-2"]
    assert dc.deck_p2.get_ID_list() == []
    assert sorted(dc.card_database.get_ID_list()) == sorted(cards)

    # 2) select a card in the database and add it to BOTH decks via the UI path: the
    #    selection is kept and the database is not depleted, so one selection serves both
    grid = dc.card_database.main_widget
    w = next(x for x in grid.widget_list if x.get_ID() == "sv1-3")   # in neither deck yet
    w.select()
    idx = grid.layout.indexOf(w)
    r, c, _, _ = grid.layout.getItemPosition(idx)
    dc.card_database.position_selected = [(r, c)]
    assert dc.card_database.peek_selection() == ["sv1-3"]
    dc.add_to_deck(dc.deck_p1)
    dc.add_to_deck(dc.deck_p2)                                       # same selection, second deck
    assert "sv1-3" in dc.deck_p1.get_ID_list() and "sv1-3" in dc.deck_p2.get_ID_list()
    assert "sv1-3" in dc.card_database.get_ID_list()                 # database not depleted
    assert dc.card_database.position_selected == [(r, c)] and w.is_selected()  # selection kept
    dc.deck_p2.add_cards(["sv1-4"])      # P2 also gets a distinct card

    # 4) named save / load round-trip
    name_edit = QtWidgets.QLineEdit("attack deck")
    dc.save_deck(dc.deck_p1, name_edit)
    assert os.path.exists(deck_dir + "attack deck.txt")
    combo = QtWidgets.QComboBox(); combo.addItem("attack deck.txt")
    dc.load_deck(dc.deck_p2, combo, QtWidgets.QLineEdit())
    assert sorted(dc.deck_p2.get_ID_list()) == ["sv1-1", "sv1-2", "sv1-3"]   # P2 replaced by P1's saved deck

    # 5) active deck = de-duplicated union of both decks
    dc.deck_p1.set_cards(["sv1-1", "sv1-2"])
    dc.deck_p2.set_cards(["sv1-2", "sv1-5"])      # sv1-2 shared
    union = dc.save_active_deck()
    assert union == ["sv1-1", "sv1-2", "sv1-5"]
    with open(active_deck) as f:
        on_disk = [l.strip() for l in f.read().splitlines() if l.strip()]
    assert on_disk == ["sv1-1", "sv1-2", "sv1-5"]

    print("DECK_TWO_PLAYER_OK")


if __name__ == "__main__":
    main()

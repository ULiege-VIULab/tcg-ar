"""Headless smoke test for the broadcast GUI additions: theme loads, the
Broadcast_panel builds and writes into a Broadcast_state, and the panel's
auto-refresh reads a live board_layout. Run with QT_QPA_PLATFORM=offscreen."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtWidgets

from inference.style import STYLESHEET
from inference.caster_module import Broadcast_state, board_layout
from inference import user_interface as ui


class FakeGameState:
    """Minimal Game_state stand-in: one Pokemon left, one right."""
    def __init__(self):
        self._cards = [
            {"x": 500, "y": 540, "n": 1, "id": "swsh1-1"},
            {"x": 1400, "y": 540, "n": 1, "id": "swsh1-2"},
        ]
    def get_number_of_card(self): return len(self._cards)
    def get_number_of_pokemon(self, i): return self._cards[i]["n"]
    def is_pokemon_card(self, i): return self._cards[i]["n"] > 0
    def get_card_location(self, i): return (self._cards[i]["x"], self._cards[i]["y"])
    def get_pokemon_card_id(self, i): return self._cards[i]["id"]


class FakeCentral(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.broadcast_state = Broadcast_state()
        self.game_state = FakeGameState()
        self.card_database = {
            "swsh1-1": {"name": "Celebi", "hp": "70", "types": ["Grass"]},
            "swsh1-2": {"name": "Charizard", "hp": "170", "types": ["Fire"]},
        }


def main():
    app = QtWidgets.QApplication([])
    app.setStyleSheet(STYLESHEET)
    assert STYLESHEET.strip(), "stylesheet empty"

    central = FakeCentral()
    panel = ui.Broadcast_panel(central)

    # board_layout splits the two pokemon correctly
    layout = board_layout(central.game_state, side_swap=False)
    assert layout["player1"]["active"]["card_id"] == "swsh1-1"
    assert layout["player2"]["active"]["card_id"] == "swsh1-2"

    # auto-refresh fills active names + resets HP to full
    panel._refresh_actives()
    assert "Celebi" in panel.fields["left"]["active_name"].text()
    assert "Charizard" in panel.fields["right"]["active_name"].text()
    assert central.broadcast_state.active["left"]["hp"] == 70
    assert central.broadcast_state.active["right"]["hp"] == 170

    # editing widgets writes through to the shared state
    panel.fields["left"]["name"].setText("Ash")
    panel.fields["left"]["score"].setValue(3)
    panel.stadium_edit.setText("World Championships")
    panel.fields["left"]["hp"].setValue(40)
    panel.fields["left"]["status"].setCurrentText("Burned")
    panel.side_swap_check.setChecked(True)

    bs = central.broadcast_state
    assert bs.player1["name"] == "Ash"
    assert bs.player1["score"] == 3
    assert bs.stadium == "World Championships"
    assert bs.active["left"]["hp"] == 40
    assert bs.active["left"]["status"] == "Burned"
    assert bs.side_swap is True

    print("BROADCAST_GUI_SMOKE_OK")


if __name__ == "__main__":
    main()

import json, os, cv2, numpy as np
from core.config import (POKEMON_CARD_DATABASE_FILE, POKEMON_CARD_DATABASE_FOLDER_PATH,
                         WIDTH, HEIGHT, REAL_IMAGES_FOLDER)
from inference.caster_module import Caster_renderer, Broadcast_state, board_layout

cards = json.load(open(POKEMON_CARD_DATABASE_FILE, encoding="utf-8"))
def good(cid, c):
    return (c.get("supertype") == "Pokémon" and c.get("hp") and c.get("attacks")
            and os.path.exists(POKEMON_CARD_DATABASE_FOLDER_PATH + cid + ".jpg"))
picks = [cid for cid, c in cards.items() if good(cid, c)][:14]
print("picked", len(picks), "cards, e.g.", picks[:3])

class Stub:
    def __init__(self, items): self.items = items
    def get_number_of_card(self): return len(self.items)
    def get_number_of_pokemon(self, i): return 1
    def get_card_location(self, i): return self.items[i][1], self.items[i][2]
    def get_pokemon_card_id(self, i): return self.items[i][0]

items = [(picks[0], 880, 540)]                                  # left active (near centre)
items += [(picks[1 + k], 180 + k * 70, 300) for k in range(5)]  # left bench
items += [(picks[6], 1040, 540)]                                # right active
items += [(picks[7 + k], 1480 + k * 50, 760) for k in range(5)] # right bench
gs = Stub(items)

# board_layout sanity
lay = board_layout(gs)
assert lay["player1"]["active"]["card_id"] == picks[0], lay["player1"]["active"]
assert lay["player2"]["active"]["card_id"] == picks[6]
assert len(lay["player1"]["bench"]) == 5 and len(lay["player2"]["bench"]) == 5
print("board_layout OK: actives + 5 bench each")

bs = Broadcast_state()
bs.player1 = {"name": "ISAIAH BRADNER", "country": "United States", "score": 1}
bs.player2 = {"name": "RILEY MCKAY", "country": "Canada", "score": 0}
bs.stadium = "Artazon"
bs.active["left"] = {"hp": 240, "status": ""}
bs.active["right"] = {"hp": 50, "status": "Asleep"}

frame = cv2.imread(os.path.join(REAL_IMAGES_FOLDER, "0.png"))
if frame is None:
    frame = (np.random.rand(HEIGHT, WIDTH, 3) * 255).astype("uint8")
out = Caster_renderer().render(frame, gs, bs, cards)
cv2.imwrite("tests/caster_preview.png", out)
print("saved tests/caster_preview.png", out.shape)

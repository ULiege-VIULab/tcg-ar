"""Unit checks for the Scarlet/Violet animated sprite integration:
form-key normalization parity, SV path resolution + role/fallback rules, and the
active/idle role scheduling. Self-contained (builds a tiny fake SV database in a
temp dir); no network, no GPU.

    python -m tests.sv_sprite_check
"""
import os
import random
import tempfile

import core.databases as databases
import inference.render_module as rm
from inference.render_module import Game_state, Multi_frame_renderer, _sv_norm_form


# --------------------------------------------------------------------------- #
# 1. normalization parity + specific mappings
# --------------------------------------------------------------------------- #
FORMS = ["Base", "Normal Form", "Alolan Form", "Galarian Form", "Stellar Form",
         "Terastal Form", "Teal Mask", "Cornerstone Mask", "Mega", "Gigantamax",
         "Paldean Form, Combat Breed", ""]
for f in FORMS:
    assert _sv_norm_form(f) == databases._sv_normalize_form(f), f
# normalization is literal ("Base"->"base"); the base-form -> "" mapping is
# _sv_lookup's is_default job, checked in section 2.
assert _sv_norm_form("Base") == "base"
assert _sv_norm_form("Normal Form") == "normal"
assert _sv_norm_form("Alolan Form") == "alolan"
assert _sv_norm_form("Stellar Form") == "stellar"
assert _sv_norm_form("Teal Mask") == "teal-mask"
print("[ok] form-key normalization parity + mappings")


# --------------------------------------------------------------------------- #
# 2. _sv_lookup resolution / role / fallback (fake SV database in a temp dir)
# --------------------------------------------------------------------------- #
tmp = tempfile.mkdtemp(prefix="sv_check_")
rm.SV_ANIMATED_MODEL_FOLDER = tmp + os.sep    # redirect the lookup folder

# spec: number -> form_key -> [roles present]
spec = {
    25:   {"": ["wait", "battle", "idle1", "idle2"]},                      # Pikachu, base only
    52:   {"": ["wait", "battle", "idle1", "idle2"],
          "alolan": ["wait", "battle", "idle1", "idle2"],
          "galarian": ["wait"]},                                          # form with only 'wait'
    1017: {"teal-mask": ["wait", "battle"],
          "cornerstone-mask": ["wait", "battle"]},                        # no base ""
}
sv_index = {}
for num, forms in spec.items():
    for fk, rolelist in forms.items():
        stem = f"{num:04d}" + (f"__{fk}" if fk else "")
        files = {}
        for role in rolelist:
            fname = f"{stem}.{role}.gif"
            open(os.path.join(tmp, fname), "wb").close()   # touch (existence is all _sv_lookup checks)
            files[role] = fname
        sv_index.setdefault(str(num), {})[fk] = files


class FakeGS:
    pass


gs = FakeGS()
gs.sv_index = sv_index


def look(dex, form, form_number, female, role):
    return Game_state._sv_lookup(gs, dex, form, form_number, female, role)


def base(p):
    return os.path.basename(p) if p else None

# base form + role selection
assert base(look(25, "Base", 0, False, "battle")) == "0025.battle.gif"
assert base(look(25, "Base", 0, False, "idle1")) == "0025.idle1.gif"
assert base(look(25, "Base", 0, False, "wait")) == "0025.wait.gif"
# role missing on a variant -> falls back to 'wait'
assert base(look(52, "Galarian Form", 3, False, "battle")) == "0052__galarian.wait.gif"
# special forms match by normalized name; base form -> "" key
assert base(look(52, "Alolan Form", 2, False, "wait")) == "0052__alolan.wait.gif"
assert base(look(52, "Base", 0, False, "wait")) == "0052.wait.gif"
# default form when SV has no bare base but has the named default (Ogerpon)
assert base(look(1017, "Teal Mask", 0, False, "battle")) == "1017__teal-mask.battle.gif"
assert base(look(1017, "Cornerstone Mask", 3, False, "wait")) == "1017__cornerstone-mask.wait.gif"
# not covered -> None (caller then falls back to gen1-8 animated / static)
assert look(999, "Base", 0, False, "wait") is None
assert look(52, "Mega", 4, False, "wait") is None            # form absent, never uses base
print("[ok] SV resolution: role selection, wait-fallback, form matching, misses")


# --------------------------------------------------------------------------- #
# 3. active/idle role scheduling
# --------------------------------------------------------------------------- #
random.seed(1234)
r = Multi_frame_renderer.__new__(Multi_frame_renderer)
r._idle_state = {}

# active card is always 'battle', and its idle state is cleared
assert r._card_role(0, True, 0.0) == rm.SV_ROLE_BATTLE
assert 0 not in r._idle_state

# benched card: mostly wait, occasionally idle1/idle2, and a fidget is followed by wait
seen = {}
prev = None
now = 0.0
for _ in range(4000):
    now += 0.5
    role = r._card_role(1, False, now)
    seen[role] = seen.get(role, 0) + 1
    if prev in rm.SV_ROLE_IDLES and role != prev:
        assert role == rm.SV_ROLE_WAIT, f"fidget not followed by wait: {prev}->{role}"
    prev = role
assert seen.get("wait", 0) > 0 and (seen.get("idle1", 0) + seen.get("idle2", 0)) > 0, seen
assert seen["wait"] > seen.get("idle1", 0) + seen.get("idle2", 0), seen   # wait dominates
print(f"[ok] role scheduling: {seen}")


# --------------------------------------------------------------------------- #
# 4. get_pokemon_path wires the shared-memory fields into _sv_lookup (SV hit)
# --------------------------------------------------------------------------- #
G = Game_state
db_list = [None] * 25
db_list[24] = {"Forms": [{"Form": "Base", "English_name": "Pikachu"}]}


class DummyGS(Game_state):
    def __init__(self): pass            # skip the heavy shared-memory init
    def update_object(self): pass


d = DummyGS()
d.existing_shm = False                 # make __del__/delete() a clean no-op
d.shared_list_variable = None
d.database = db_list
d.sv_index = sv_index
d.form_dictionary = {"Base": ""}       # used only by the fallback (shiny) path
row = [0] * 19
row[G.NUMBER_POKEMON_INDEX] = 1
row[G.POKEMON_POKEDEX_NUMBER_INDEX_1] = 25   # Pikachu
row[G.FORM_INDEX_1] = 1                        # form_number 0 (base)
row[G.MODEL_INDEX_1] = 1                       # animated model type
d.pokemon_list = [row]

got = G.get_pokemon_path(d, 0, role="battle", sv_enabled=True)
assert base(got[0]) == "0025.battle.gif", got
got_wait = G.get_pokemon_path(d, 0, role="wait", sv_enabled=True)
assert base(got_wait[0]) == "0025.wait.gif", got_wait
# shiny -> SV skipped by get_pokemon_path (would fall back to the 2D sources)
row[G.SHINY_INDEX_1] = 1
got_shiny = G.get_pokemon_path(d, 0, role="battle", sv_enabled=True)
assert "0025.battle.gif" not in (got_shiny[0] or ""), got_shiny
print("[ok] get_pokemon_path -> SV hit by role; shiny bypasses SV")


# --------------------------------------------------------------------------- #
# 5. SV is opt-in: default (sv_enabled=False) never returns SV paths
# --------------------------------------------------------------------------- #
row[G.SHINY_INDEX_1] = 0
default_off = G.get_pokemon_path(d, 0, role="battle")               # sv_enabled defaults to False
assert "0025" not in (default_off[0] or ""), default_off           # no SV file -> fell back
on = G.get_pokemon_path(d, 0, role="battle", sv_enabled=True)
assert base(on[0]) == "0025.battle.gif", on
assert rm._is_sv_path(on[0]) and not rm._is_sv_path(default_off[0])
print("[ok] SV opt-in: off -> fallback path, on -> SV path")

print("\nAll SV sprite checks passed.")

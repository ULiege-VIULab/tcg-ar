"""Prove the optimized create_card_databases indexing == the original nested loops,
on fake card data (no network)."""
import json
import random


class FakeCard:
    def __init__(self, cid, types, subtypes, nums):
        self.id = cid
        self.types = types
        self.subtypes = subtypes
        self.nationalPokedexNumbers = nums


def original(pokemon_card, types, subtypes, MAX):
    pokemon_card_type = {}
    for t in types:
        sm = {}
        for st in subtypes:
            sm[st] = [c.id.replace("?", "question_hires") for c in pokemon_card
                      if t in (c.types or []) and st in (c.subtypes or [])]
        pokemon_card_type[t] = sm
    pokemon_card_num = {}
    for n in range(1, MAX + 1):
        tm = {}
        for t in types:
            sm = {}
            for st in subtypes:
                sm[st] = [c.id.replace("?", "question_hires") for c in pokemon_card
                          if (c.nationalPokedexNumbers or []) and n in (c.nationalPokedexNumbers or [])
                          and t in (c.types or []) and st in (c.subtypes or [])]
            tm[t] = sm
        pokemon_card_num[n] = tm
    return pokemon_card_type, pokemon_card_num


def optimized(pokemon_card, types, subtypes, MAX):
    type_index = {}
    num_index = {}
    for card in pokemon_card:
        cid = card.id.replace("?", "question_hires")
        for t in (card.types or []):
            for st in (card.subtypes or []):
                type_index.setdefault((t, st), []).append(cid)
                for n in (card.nationalPokedexNumbers or []):
                    num_index.setdefault((n, t, st), []).append(cid)
    pct = {t: {st: type_index.get((t, st), []) for st in subtypes} for t in types}
    pcn = {n: {t: {st: num_index.get((n, t, st), []) for st in subtypes} for t in types}
           for n in range(1, MAX + 1)}
    return pct, pcn


def run():
    random.seed(1)
    types = ["Fire", "Water", "Grass", "Lightning"]
    subtypes = ["Basic", "Stage 1", "Stage 2", "VMAX"]
    MAX = 60
    cards = []
    for i in range(400):
        ts = random.sample(types, random.randint(0, 2))
        sts = random.sample(subtypes, random.randint(0, 2))
        nums = random.sample(range(1, MAX + 1), random.randint(0, 2))
        cid = f"set{i%5}-{i}" + ("?" if i % 7 == 0 else "")
        cards.append(FakeCard(cid, ts or None, sts or None, nums or None))

    o_t, o_n = original(cards, types, subtypes, MAX)
    n_t, n_n = optimized(cards, types, subtypes, MAX)
    # Compare serialized JSON (exact structure, ordering, keys-as-strings).
    assert json.dumps(o_t, ensure_ascii=False) == json.dumps(n_t, ensure_ascii=False), "type DB differs"
    assert json.dumps(o_n, ensure_ascii=False) == json.dumps(n_n, ensure_ascii=False), "number DB differs"
    print("OK: optimized create_card_databases produces identical JSON to the original")
    print(f"   ({len(cards)} cards, types={len(types)}, subtypes={len(subtypes)}, pokedex={MAX})")


if __name__ == "__main__":
    run()

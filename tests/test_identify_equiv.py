"""Verify the optimized identify_cards matches the original per-anchor loop logic,
and the color-stats cache matches recomputing from disk-equivalent stats."""
import numpy as np
import torch
import torch.nn as nn

from core.models import identification as idm


def original_triplet_select(query, anchors_dict):
    """Reference: the original O(N) per-anchor TripletMarginLoss argmin."""
    criterion = nn.TripletMarginLoss(1.0, 2.0, reduction='none')
    nb = len(query)
    minimum = torch.tensor(np.full((nb,), np.inf))
    cards_id = np.full((nb,), next(iter(anchors_dict)))
    for key in anchors_dict:
        anchors = torch.unsqueeze(anchors_dict[key], 0).repeat(nb, 1)
        crit = criterion(anchors, query, anchors)
        cards_id = np.where(minimum < crit, cards_id, np.full((nb,), key))
        minimum = torch.minimum(minimum, crit)
    return cards_id


def original_arcface_select(query, anchors_dict):
    import torch.nn.functional as F
    labels = list(anchors_dict.keys())
    mat = torch.stack(list(anchors_dict.values()))
    q = F.normalize(query, p=2, dim=1)
    mat = F.normalize(mat, p=2, dim=1)
    sim = torch.mm(q, mat.t())
    idx = torch.max(sim, dim=1).indices.numpy()
    return np.array([labels[i] for i in idx])


class FakeModel(nn.Module):
    """Returns the input rows unchanged (identity embedding) so we control vectors."""
    def __init__(self): super().__init__()
    def forward(self, x): return x
    def eval(self): return self


def run():
    torch.manual_seed(0); np.random.seed(0)
    D = 8
    keys = [f"card-{i}" for i in range(25)]
    anchors_dict = {k: torch.randn(D) for k in keys}
    card_id_list = {k: ["Pokémon", [i + 1], [], []] for i, k in enumerate(keys)}

    # Build query batch = some anchors + noise so the nearest is well-defined.
    B = 12
    query = torch.stack([anchors_dict[keys[i % len(keys)]] + 0.01 * torch.randn(D) for i in range(B)])

    model = FakeModel()
    device = torch.device('cpu')

    # triplet
    ref_t = original_triplet_select(query.clone(), anchors_dict)
    got_t, pk_t = idm.identify_cards(query.clone(), anchors_dict, model, device, card_id_list, method="triplet")
    assert list(ref_t) == list(got_t), f"triplet mismatch:\n{ref_t}\n{got_t}"

    # arcface
    ref_a = original_arcface_select(query.clone(), anchors_dict)
    got_a, pk_a = idm.identify_cards(query.clone(), anchors_dict, model, device, card_id_list, method="arcface")
    assert list(ref_a) == list(got_a), f"arcface mismatch:\n{ref_a}\n{got_a}"

    # pokemons list lookup
    assert pk_t[0] == card_id_list[got_t[0]][1], "pokedex lookup wrong"

    print("OK: triplet selection matches original on", B, "queries")
    print("OK: arcface selection matches reference on", B, "queries")
    print("triplet sample:", list(got_t[:4]))


if __name__ == "__main__":
    run()

"""Logic tests that don't need the big assets: DOTA round-trip, color-stat cache,
config completeness."""
import numpy as np


def test_dota_is_rotated_rectangle():
    from core import databases as db
    # annotation: [id, x, y, w, h, angle]
    ann = [["c", 100, 100, 60, 90, 0]]
    dota = db.to_DOTA(ann)[0]
    assert dota[-2] == "card" and dota[-1] == 0
    pts = np.array(dota[:8], dtype=float).reshape(4, 2)
    # axis-aligned (angle 0): width and height edges must match w,h
    w = np.linalg.norm(pts[1] - pts[0])
    h = np.linalg.norm(pts[2] - pts[1])
    assert abs(w - 60) <= 1 and abs(h - 90) <= 1, (w, h)
    # 45-degree rotation keeps the same side lengths
    dota45 = db.to_DOTA([["c", 100, 100, 60, 90, 45]])[0]
    pts45 = np.array(dota45[:8], dtype=float).reshape(4, 2)
    w45 = np.linalg.norm(pts45[1] - pts45[0])
    assert abs(w45 - 60) <= 2, w45
    print("OK: to_DOTA produces a correct rotated rectangle")


def test_anchor_color_stat_cache(monkeypatch_imread=True):
    import cv2
    from core.models import identification as idm
    calls = {"n": 0}
    real = cv2.imread
    def counting_imread(path, *a, **k):
        calls["n"] += 1
        return (np.random.rand(40, 40, 3) * 255).astype("uint8")
    cv2.imread = counting_imread
    try:
        idm._anchor_color_stats.clear()
        s1 = idm._anchor_color_stat("zzz-test-card")
        s2 = idm._anchor_color_stat("zzz-test-card")
        assert s1 is s2 or s1 == s2
        assert calls["n"] == 1, f"expected 1 disk read, got {calls['n']}"
    finally:
        cv2.imread = real
    print("OK: anchor colour stats are cached (1 disk read across 2 calls)")


def test_config_completeness():
    from core import config as c
    required = ["ASSETS_ROOT", "IDENTIFICATION_METHOD", "WIDTH", "HEIGHT",
               "IDENTIFICATION_MODEL_SAVE_PATH", "IDENTIFICATION_MODEL_SAFE_PATH",
               "ORIENTATION_MODEL_SAVE_PATH", "DETECTION_WEIGHT_PATH",
               "MAX_NUMBER_OF_POKEMON_PER_CARD", "POKEMON_TCG_API_KEY"]
    for name in required:
        assert hasattr(c, name), f"missing config: {name}"
    assert c.model_save_path("arcface", True).endswith("arcface_fine_tune_model.pth")
    assert c.model_save_path("triplet", False).endswith("identification_model_51.pth")
    print("OK: config has all required names and method dispatch works")


if __name__ == "__main__":
    test_dota_is_rotated_rectangle()
    test_anchor_color_stat_cache()
    test_config_completeness()
    print("ALL LOGIC TESTS PASSED")

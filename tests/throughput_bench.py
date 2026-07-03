"""Throughput benchmark for the paper's runtime numbers.

Sections (each can be selected with --only):
  reco   : orientation (per backbone) and identification forward+retrieval FPS,
           the latter for {GPU,CPU} x {full ~20k, deck ~214} candidate sets,
  render : per output form, the composite / resize / buffer-copy cost and View-FPS.
Detection FPS is measured by evaluation.eval_detection --fps per method.

Run:  python -m tests.throughput_bench [--only reco|render]
"""
import argparse
import time
import numpy as np
import torch

from core.config import (WIDTH, HEIGHT, IDENTIFICATION_IMAGE_SIZE, ORIENTATION_IMAGE_SIZE,
                         ARCFACE_EMBEDDING_SIZE, ORIENTATION_ARCH_WEIGHTS)
from inference.caster_module import Caster_renderer, Broadcast_state
from inference.render_module import Multi_frame_renderer

N_CARDS = 14
N_FULL = 20360        # usable reference cards (open set)
N_DECK = 214          # cards present in the real data (deck-restricted)
ITERS = 30


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _time(fn, iters=ITERS, warmup=5):
    for _ in range(warmup):
        fn()
    _sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync()
    return (time.perf_counter() - t0) / iters


# --------------------------------------------------------------------------- #
def bench_orientation():
    from core.models.orientation import OrientationModel
    print("== orientation forward (board of %d crops) ==" % N_CARDS)
    devices = ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]
    for arch in ORIENTATION_ARCH_WEIGHTS:
        line = f"  {arch:20s}"
        for dev in devices:
            device = torch.device(dev)
            m = OrientationModel().build_model(2, device, arch=arch).eval()
            x = torch.randn(N_CARDS, 3, ORIENTATION_IMAGE_SIZE, ORIENTATION_IMAGE_SIZE, device=device)
            with torch.no_grad():
                dt = _time(lambda: m(x), iters=20)
            line += f"  {dev}:{1.0/dt:6.0f} board/s"
            del m
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        print(line)


def bench_identification():
    from core.config import IDENTIFICATION_OUT_FEATURES
    from core.models.identification import load_model
    print("== identification forward+retrieval (board of %d crops) ==" % N_CARDS)
    devices = ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]
    # (method, embedding dim, cosine/euclid retrieval)
    methods = [("arcface", ARCFACE_EMBEDDING_SIZE, "cos"), ("triplet", IDENTIFICATION_OUT_FEATURES, "l2")]
    for meth, dim, kind in methods:
        for dev in devices:
            device = torch.device(dev)
            model = load_model(meth, fine_tuned=True, device=device).eval()
            x = torch.randn(N_CARDS, 3, IDENTIFICATION_IMAGE_SIZE, IDENTIFICATION_IMAGE_SIZE, device=device)
            for label, n in (("full", N_FULL), ("deck", N_DECK)):
                anchors = torch.randn(n, dim, device=device)
                if kind == "cos":
                    anchors = torch.nn.functional.normalize(anchors, dim=1)

                def step():
                    with torch.no_grad():
                        q = model(x)
                        if kind == "cos":
                            q = torch.nn.functional.normalize(q, dim=1)
                            (q @ anchors.t()).topk(5, dim=1)
                        else:
                            torch.cdist(q, anchors).topk(5, dim=1, largest=False)
                dt = _time(step, iters=20)
                print(f"  {meth:8s} {dev}  {label:4s} (N={n:5d}):  {1.0/dt:6.1f} board/s")
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def bench_rendering():
    print("== rendering per output form (1080p, board of %d sprites) ==" % N_CARDS)
    r = Multi_frame_renderer(number_of_view=1)
    sprite = np.zeros((1, 160, 160, 4), np.uint8); sprite[0, :, :, 0:3] = (60, 180, 240); sprite[0, :, :, 3] = 255
    paths = []
    for k in range(N_CARDS):
        key = f"m{k}"; r.pokemon_dict[key] = k
        r.pokemon_original_models.append(sprite); r.pokemon_models.append(sprite)
        r.gif_duration.append(None); r.num_frames_in_gif.append(1)
        r.current_frame_num.append(np.zeros((1,), np.int32)); r.time_elapsed.append(0); paths.append(key)

    class GS:
        def get_number_of_card(self): return N_CARDS
        def get_card_location(self, i): return (200 + (i % 7) * 220, 250 + (i // 7) * 400)
        def get_pokemon_path(self, i): return [paths[i]]
    gs = GS(); H = np.eye(3)
    frame = (np.random.rand(HEIGHT, WIDTH, 3) * 255).astype(np.uint8)

    import cv2
    fw, fh = WIDTH // 3, HEIGHT // 3

    def resize_copy(img):
        cv2.resize(img, (fw, fh), interpolation=cv2.INTER_AREA); _ = img.copy()

    ar_ms = 1000 * _time(lambda: r.render_frame(0, frame.copy(), gs, H))
    cr = Caster_renderer(); bs = Broadcast_state(); bs.stadium = "Worlds"

    class GSb:
        def get_number_of_card(self): return 0
        def get_number_of_pokemon(self, i): return 0
        def is_pokemon_card(self, i): return False
        def get_card_location(self, i): return (0, 0)
        def get_pokemon_card_id(self, i): return ""
    gsb = GSb()
    cr.render(frame, gsb, bs, {}, 0)
    caster_warm_ms = 1000 * _time(lambda: cr.render(frame, gsb, bs, {}, 0))
    # cold: force a panel rebuild each call by toggling the signature
    def caster_cold():
        cr._base_sig = None
        cr.render(frame, gsb, bs, {}, 0)
    caster_cold_ms = 1000 * _time(caster_cold)
    rc_ms = 1000 * _time(lambda: resize_copy(frame))

    forms = [
        ("Raw",                0.0,            rc_ms),
        ("Augmented (AR)",     ar_ms,          rc_ms),
        ("Broadcast (warm)",   caster_warm_ms, rc_ms),
        ("Broadcast (cold)",   caster_cold_ms, rc_ms),
    ]
    print(f"  {'form':20s} {'composite':>10s} {'resize+copy':>12s} {'total':>8s} {'view-fps':>9s}")
    for name, comp, rc in forms:
        total = comp + rc
        print(f"  {name:20s} {comp:8.2f}ms {rc:10.2f}ms {total:6.2f}ms {1000/total:7.0f}")
    # all three forms = raw + AR + broadcast(warm), 3x resize+copy
    total3 = ar_ms + caster_warm_ms + 3 * rc_ms
    print(f"  {'All three forms':20s} {ar_ms+caster_warm_ms:8.2f}ms {3*rc_ms:10.2f}ms {total3:6.2f}ms {1000/total3:7.0f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["reco", "render"], default=None)
    args = ap.parse_args()
    if args.only in (None, "render"):
        bench_rendering()
    if args.only in (None, "reco"):
        bench_orientation()
        bench_identification()
    print("THROUGHPUT_DONE")


if __name__ == "__main__":
    main()

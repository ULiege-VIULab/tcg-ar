"""
Recognition process entry point (runs in its own ``multiprocessing.Process``).

Per zenithal frame: detect cards -> crop -> resolve 180-degree flip (orientation)
-> identify -> write the result into the shared ``Game_state``.

The identification step branches on ``config.IDENTIFICATION_METHOD`` (triplet or
arcface) -- both go through the same ``core.models.identification`` API, and both
support deck hot-reload via the ``new_deck`` flag.
"""

import sys
import json
import time
import copy
import signal

import cv2
import torch
import numpy as np

from core import config
from core.config import (HEIGHT, WIDTH, CHANNELS, IDENTIFICATION_IMAGE_SIZE,
                         IDENTIFICATION_POKEMON_CARD_ID_DATABASE_FILE)
from core.shared_memory import Shared_frame_buffer
from core.transforms import get_inference_transform
from core.models.detection import DetectionModel
from core.models.orientation import OrientationModel
from core.models import identification as idm

from inference.io_module import read_a_frame_from_buffer
from inference.render_module import Game_state, Multi_frame_renderer, display_fps


def _load_models(device, method, fine_tuned=True):
    detection_model = DetectionModel()
    detection_model.load_detection_model(device)
    orientation_model = OrientationModel()
    orientation_model.load_orientation_model(device)
    identification_model = idm.load_model(method, fine_tuned=fine_tuned, device=device)
    anchors_dict = idm.evaluate_anchors(
        identification_model, device,
        model_path=config.model_save_path(method, fine_tuned))
    return detection_model, orientation_model, identification_model, anchors_dict


def _recognize_frame(frame_read, detection_model, orientation_model, identification_model,
                     anchors_dict, card_id_list, transform, device, method, color_correction):
    """Run detection -> orientation -> identification on one frame, return the state list."""
    bboxes = detection_model.detect_card_location(frame_read)
    cards = detection_model.extract_cards(frame_read, bboxes)
    if not cards:
        return []

    original_cards = []
    orientation_cards = torch.tensor([]).to(device)
    for card in cards:
        original_cards.append(copy.copy(card))
        card = cv2.cvtColor(card, cv2.COLOR_BGR2RGB)
        card = transform(card)
        orientation_cards = torch.cat((orientation_cards, torch.unsqueeze(card, 0).to(device)))

    if not orientation_cards.numel():
        return []

    orientation_outputs = orientation_model.model(orientation_cards).detach().to('cpu').numpy()
    identification_cards = orientation_cards
    for i, orientation_output in enumerate(orientation_outputs):
        if orientation_output[0] > 0:
            original_cards[i] = cv2.rotate(original_cards[i], cv2.ROTATE_180)
            card = cv2.cvtColor(original_cards[i], cv2.COLOR_BGR2RGB)
            identification_cards[i] = torch.unsqueeze(transform(card), 0).to(device)

    if color_correction:
        cards_id, pokemons_id = idm.identify_cards_with_color_correction(
            original_cards, anchors_dict, transform, identification_model, device, card_id_list, method=method)
    else:
        cards_id, pokemons_id = idm.identify_cards(
            identification_cards, anchors_dict, identification_model, device, card_id_list, method=method)

    bboxes_stack = np.stack(bboxes, 1, dtype=np.int64, casting="unsafe")
    state = [[cards_id[i], pokemons_id[i], int(bboxes_stack[0][i]), int(bboxes_stack[1][i])]
             for i in range(len(cards_id))]
    state.sort()
    return state


def identification_pipe(id_buffer, buffer_lock, barrier, color_correction, new_deck, use_fine_tuned):
    method = config.IDENTIFICATION_METHOD

    def termination_handler(signum, frame):
        print("Stop identification process")
        try:
            buffer_lock.release()
        except ValueError:
            exit(0)
        exit(0)
    signal.signal(signal.SIGTERM, termination_handler)

    try:
        pokemon_game_state = Game_state(False)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        detection_model, orientation_model, identification_model, anchors_dict = _load_models(
            device, method, fine_tuned=use_fine_tuned.value)
        card_id_list = idm.load_card_id_list()  # includes the back card entry
        barrier.wait()
    except FileNotFoundError:
        print("a file required to initialise the identification pipe could not be found", file=sys.stderr)
        barrier.wait()
        return -1

    shared_frame_buffer = Shared_frame_buffer(id=id_buffer, io=1, length=1,
                                              resolution=(HEIGHT, WIDTH, CHANNELS), existing_shm=True)
    transform = get_inference_transform(IDENTIFICATION_IMAGE_SIZE)
    print(f"Identification pipe ({method}) set up! Starting identification.")

    while True:
        if new_deck.value:
            identification_model = idm.load_model(method, fine_tuned=use_fine_tuned.value, device=device)
            anchors_dict = idm.evaluate_anchors(
                identification_model, device,
                model_path=config.model_save_path(method, use_fine_tuned.value))
            new_deck.value = not new_deck.value

        frame_read = read_a_frame_from_buffer(buffer=shared_frame_buffer, lock=buffer_lock)
        if not isinstance(frame_read, int):
            state = _recognize_frame(frame_read, detection_model, orientation_model, identification_model,
                                     anchors_dict, card_id_list, transform, device, method, color_correction.value)
            pokemon_game_state.update_state(state)


def test_pipe_on_video(video_path):
    """Headless smoke test: run the full recognition + rendering loop on a video file."""
    method = config.IDENTIFICATION_METHOD
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pokemon_game_state = Game_state(False)
    detection_model, orientation_model, identification_model, anchors_dict = _load_models(device, method)
    card_id_list = idm.load_card_id_list()  # includes the back card entry
    transform = get_inference_transform(IDENTIFICATION_IMAGE_SIZE)
    renderer = Multi_frame_renderer(number_of_view=1)

    vid = cv2.VideoCapture(video_path)
    ret, frame_read = vid.read()
    time_length = 10
    time_elapsed_array = np.zeros((time_length,))
    nb_frame_produce = 0
    new_time = time.time()

    while ret:
        old_time = new_time
        ret, frame_read = vid.read()
        if not isinstance(frame_read, int) and frame_read is not None:
            state = _recognize_frame(frame_read, detection_model, orientation_model, identification_model,
                                     anchors_dict, card_id_list, transform, device, method, color_correction=False)
            pokemon_game_state.update_state(state)
            renderer.load_models(pokemon_game_state.get_pokemon_paths())
            ar_frame = renderer.render_frame(0, frame_read, pokemon_game_state, np.eye(3))
            new_time = time.time()
            nb_frame_produce = (nb_frame_produce + 1) % time_length
            time_elapsed_array[nb_frame_produce] = new_time - old_time
            cv2.imshow('test', display_fps(ar_frame, time_elapsed_array))
            cv2.waitKey(1)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Identification pipe smoke test on a video")
    parser.add_argument("video", help="path to a video file")
    args = parser.parse_args()
    test_pipe_on_video(args.video)

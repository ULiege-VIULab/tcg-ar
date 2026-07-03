"""
All database / dataset building in one place.

This consolidates code that the original project spread across three modules and
three different API-querying functions:

* Pokemon metadata + sprite databases (was in ``render_module``),
* the card-image database (was duplicated in ``render_module`` and ``detection_module``),
* the six identification JSON databases (was in ``identification_module``),
* synthetic dataset generation for detection / orientation / identification.

Nothing here is duplicated elsewhere; the installation, training and evaluation
packages all call into this module.
"""

import os
import re
import sys
import json
import time
from pathlib import Path
from urllib.request import urlopen

import cv2
import numpy as np
import requests
from tqdm import tqdm
from bs4 import BeautifulSoup
from pokemontcgsdk import RestClient, Card, Type, Subtype, Set

from core.config import *  # noqa: F401,F403  (paths + hyper-parameters)
from core.image_ops import (
    affine_transformation, data_augmentation, add_perspective_transform,
    increase_brightness, increase_saturation, white_noise, pepper_noise,
    random_neon_light_effect, random_spot_light_effect,
)


def create_nested_folders(folder_name):
    """Single replacement for the three ``create_*_folder_nested`` helpers."""
    os.makedirs(folder_name, exist_ok=True)


def _configure_api(api_key=POKEMON_TCG_API_KEY):
    if not api_key:
        raise RuntimeError(
            "A Pokemon TCG API key is required to download/build the card databases.\n"
            "Get a free key at https://dev.pokemontcg.io and set it in your environment:\n"
            "    export POKEMON_TCG_API_KEY=your-key        # Linux/macOS\n"
            "    setx POKEMON_TCG_API_KEY your-key           # Windows (new shells)\n"
            "(The pre-built models and the live pipeline do not need a key; it is only "
            "used by `python -m installation.install`.)")
    RestClient.configure(api_key)


from urllib.error import HTTPError, URLError  # noqa: E402
try:
    from pokemontcgsdk.restclient import PokemonTcgException
except Exception:  # pragma: no cover - keep working if the sdk layout changes
    PokemonTcgException = ()

_TRANSIENT_ERRORS = (HTTPError, URLError, TimeoutError, ConnectionError,
                     requests.exceptions.RequestException, json.JSONDecodeError, PokemonTcgException)


def _retry(fn, *args, retries=6, base_delay=3.0, **kwargs):
    """Call ``fn`` retrying transient network / API failures (e.g. the Pokemon TCG
    API's intermittent HTTP 504) with exponential backoff."""
    for attempt in range(retries):
        try:
            return fn(*args, **kwargs)
        except _TRANSIENT_ERRORS as e:
            if attempt == retries - 1:
                raise
            delay = base_delay * (2 ** attempt)
            print(f"  transient error ({type(e).__name__}); retry "
                  f"{attempt + 1}/{retries - 1} in {delay:.0f}s", file=sys.stderr)
            time.sleep(delay)


_API_BASE = "https://api.pokemontcg.io/v2"


def _api_get(resource, params=None, api_key=POKEMON_TCG_API_KEY):
    """GET one page of a Pokemon TCG REST resource as raw JSON, bypassing the SDK's
    strict dataclass parser (which rejects whole pages when a single card has a
    malformed nested field such as a tcgplayer block missing ``updatedAt``).

    The request AND the JSON parse are retried together, so a 5xx status or an
    empty / non-JSON gateway response (which the flaky public API returns often)
    is treated as transient and retried instead of crashing."""
    return _api_get_full(resource, params, api_key).get("data", [])


def _api_get_full(resource, params=None, api_key=POKEMON_TCG_API_KEY):
    """Like ``_api_get`` but returns the whole payload (so callers can read
    ``totalCount`` to drive a progress bar)."""
    headers = {"X-Api-Key": api_key} if api_key else {}

    def _once():
        resp = requests.get(f"{_API_BASE}/{resource}", params=params, headers=headers, timeout=60)
        resp.raise_for_status()
        return resp.json()

    return _retry(_once)


def _fetch_all_cards_raw(select=None, q=None, api_key=POKEMON_TCG_API_KEY, page_size=250, desc="fetching cards"):
    """Paginate ``/cards`` directly and return a list of raw card dicts, showing a
    tqdm bar driven by the API's ``totalCount``."""
    all_cards = []
    page = 1
    pbar = None
    while True:
        params = {"page": page, "pageSize": page_size}
        if select:
            params["select"] = select
        if q:
            params["q"] = q
        payload = _api_get_full("cards", params, api_key)
        data = payload.get("data", [])
        if not data:
            break
        all_cards.extend(data)
        if pbar is None:
            pbar = tqdm(total=payload.get("totalCount"), desc=desc, unit="card")
        pbar.update(len(data))
        page += 1
        time.sleep(0.1)
    if pbar is not None:
        pbar.close()
    return all_cards


# --------------------------------------------------------------------------- #
# Card image database (one downloader, was duplicated 2x in the original code)
# --------------------------------------------------------------------------- #
def download_card_database(api_key=POKEMON_TCG_API_KEY, path=POKEMON_CARD_DATABASE_FOLDER_PATH, update=False):
    """Download every card image from the Pokemon TCG API into ``path``.

    With ``update=True`` only the missing images are fetched."""
    create_nested_folders(path)

    # Only id + images are needed, and the raw JSON avoids the SDK schema bug.
    cards = _fetch_all_cards_raw(select="id,images", api_key=api_key, desc="listing card images")
    for card in tqdm(cards, desc="downloading card images", unit="img"):
        card_id = card["id"].replace(":", "semicolon").replace("?", "question_hires")
        dest = os.path.join(path, card_id + ".jpg")
        if update and os.path.exists(dest):
            continue
        img_url = (card.get("images") or {}).get("large")
        if not img_url:
            continue
        resp = _retry(requests.get, img_url, timeout=30)
        card_image = cv2.imdecode(np.asarray(bytearray(resp.content), dtype=np.uint8), cv2.IMREAD_COLOR)
        if card_image is None or not cv2.imwrite(dest, card_image):
            print("Card " + card_id + " could not be saved due to an error.", file=sys.stderr)


def update_card_database(api_key=POKEMON_TCG_API_KEY, path=POKEMON_CARD_DATABASE_FOLDER_PATH):
    download_card_database(api_key=api_key, path=path, update=True)


# --------------------------------------------------------------------------- #
# Pokemon (pokepedia) database + 2D / 2D-animated sprite databases
# --------------------------------------------------------------------------- #
def create_pokemon_database():
    """Scrape pokepedia and build ``POKEMON_DATABASE_FILE`` (1025 Pokemon + forms)."""
    page = _retry(urlopen, POKEMON_DATABASE_URL)
    html_code = page.read().decode("utf-8")

    pattern = "<tbody><tr><td rowspan=\".*?\">.*?</tr></tbody>"
    pokedex_html_table = re.search(pattern, html_code, re.IGNORECASE).group()
    split_pokedex_html_table = pokedex_html_table.split("<tr><td rowspan=")[1:]

    database = []
    for entry in split_pokedex_html_table:
        split_entry = entry.split("<td")
        pattern_2 = ".*?>"
        pokemon_number = re.sub(pattern_2, "", re.sub("<.*", "", split_entry[0]))

        column = [2, 3]
        name_tag = ["Frensh_name", "English_name"]
        form_tag = ["Forme", "Form"]
        pokemon_forms = []

        while column[0] < len(split_entry):
            dict_form = {}
            for i in range(len(column)):
                sub_result = re.sub("</a>.*", "", split_entry[column[i]])
                pokemon_name = re.sub(pattern_2, "", sub_result)
                pokemon_form = re.search("<small>.*?</small>", split_entry[column[i]])
                if pokemon_form:
                    pokemon_form = pokemon_form.group()
                    pokemon_form = re.sub("</small>.*", "", pokemon_form)
                    pokemon_form = re.sub("<small>", "", pokemon_form)
                    pokemon_form = re.sub("<.*?>", ", ", pokemon_form)
                else:
                    if re.search("Mega.*? X|Méga.*? X", sub_result):
                        pokemon_form = "Mega X"
                    elif re.search("Mega.*? Y|Méga.*? Y", sub_result):
                        pokemon_form = "Mega Y"
                    elif re.search("Mega|Méga", sub_result):
                        pokemon_form = "Mega"
                    elif re.search("Ultra", sub_result):
                        pokemon_form = "Ultra"
                    else:
                        pokemon_form = "Base"
                dict_form.update({name_tag[i]: pokemon_name})
                dict_form.update({form_tag[i]: pokemon_form})
            column[0] += 7
            column[1] += 7
            pokemon_forms.append(dict_form)
        database.append({"Number": pokemon_number, "Forms": pokemon_forms})

    create_nested_folders(DATABASE_FOLDER_PATH)
    with open(POKEMON_DATABASE_FILE, 'w', encoding='utf-8') as f:
        json.dump(database, f, ensure_ascii=False, indent=4)


def _resolve_image_link(image):
    for key in ("data-srcset", "data-src", "data-fallback-src", "src"):
        try:
            return image[key]
        except Exception:
            continue
    return None


def download_gif_images(images, folder_name):
    create_nested_folders(folder_name)
    count = 0
    print(f"Total {len(images)} Image Found!")
    for image in tqdm(images):
        image_link = _resolve_image_link(image)
        try:
            r = requests.get(image_link).content
            parts = image_link.split("/")
            name = re.sub(".gif", "", parts[-1])
            info = re.sub("normal-sprites|normal-sprite|-sprites|-sprite|normal-|swsh-", "", parts[-2])
            name = (name + "-" + info + ".gif") if info else (name + ".gif")
            with open(os.path.join(folder_name, name), "xb") as f:
                f.write(r)
                count += 1
        except Exception:
            pass
    print(f"Total {count} images downloaded out of {len(images)}.")


def create_2D_animated_database():
    create_nested_folders(POKEMON_2D_ANIMATED_MODEL_FOLDER)
    for url in tqdm(POKEMON_2D_ANIMATED_DATABASE):
        r = requests.get(url)
        soup = BeautifulSoup(r.text, 'html.parser')
        images = soup.findAll('img', src=re.compile(".gif"))
        download_gif_images(images, POKEMON_2D_ANIMATED_MODEL_FOLDER)


def download_images(images, list_name, folder_name):
    create_nested_folders(folder_name)
    count = 0
    print(f"Total {len(images)} Image Found!")
    for i, image in enumerate(tqdm(images)):
        image_link = _resolve_image_link(image)
        try:
            r = requests.get("https://www.pokepedia.fr/" + image_link).content
            with open(os.path.join(folder_name, list_name[i]), "xb") as f:
                f.write(r)
                count += 1
        except Exception:
            pass
    print(f"Total {count} images downloaded out of {len(images)}.")


def create_2D_database():
    create_nested_folders(POKEMON_2D_MODEL_FOLDER)
    r = requests.get(POKEMON_DATABASE_URL)
    soup = BeautifulSoup(r.text, 'html.parser')
    images = soup.findAll('img', src=re.compile(".png"), alt=re.compile("[0-9][0-9][0-9][0-9]"))

    with open(POKEMON_DATABASE_FILE, "r", encoding="utf-8") as database_file:
        database = json.load(database_file)

    list_name = []
    for item in tqdm(database):
        for form in item["Forms"]:
            file_name = (form["English_name"] + "_" + form["Form"] + ".png").replace(":", "").replace("?", "")
            list_name.append(file_name)
    download_images(images, list_name, POKEMON_2D_MODEL_FOLDER)


# --------------------------------------------------------------------------- #
# Card encyclopedia (pokemon_card.json) + the six identification JSON databases
# --------------------------------------------------------------------------- #
def create_card_file():
    """Build ``POKEMON_CARD_DATABASE_FILE`` (full per-card metadata encyclopedia).

    Fetches the raw REST JSON directly so a single card with a malformed nested
    field (e.g. a ``tcgplayer`` block missing ``updatedAt``) cannot break a whole
    page -- which the strict ``pokemontcgsdk`` dataclass parser does.  Each entry is
    the API's own card object plus a local ``image`` path."""
    # Smaller pages: full-object responses are large and the public gateway times
    # out / returns empty bodies on big pages.
    all_cards = _fetch_all_cards_raw(page_size=100, desc="fetching encyclopedia")

    pokemon_card_id = {}
    for card in tqdm(all_cards, desc="building pokemon_card.json", unit="card"):
        card_id = card["id"].replace("?", "question_hires")
        entry = dict(card)
        entry["image"] = POKEMON_CARD_DATABASE_FOLDER_PATH + card_id + ".jpg"
        pokemon_card_id[card_id] = entry

    create_nested_folders(DATABASE_FOLDER_PATH)
    with open(POKEMON_CARD_DATABASE_FILE, 'w', encoding='utf-8') as f:
        json.dump(pokemon_card_id, f, ensure_ascii=False, indent=4)


def create_card_databases():
    """Build the six identification JSON databases used for fast offline lookups.

    Uses the raw REST JSON (not the strict SDK) so malformed nested fields on a few
    cards cannot abort the whole download."""
    create_nested_folders(DATABASE_FOLDER_PATH)

    all_cards = _fetch_all_cards_raw(select="id,supertype,types,subtypes,nationalPokedexNumbers",
                                     desc="fetching card index")
    pokemon_card = [c for c in all_cards if c.get("supertype") == "Pokémon"]
    trainer_card = [c for c in all_cards if c.get("supertype") == "Trainer"]
    energy_card = [c for c in all_cards if c.get("supertype") == "Energy"]
    types = _api_get("types")
    subtypes = _api_get("subtypes")

    pokemon_card_id = {}
    for card in all_cards:
        cid = card["id"].replace("?", "question_hires")
        pokemon_card_id[cid] = [card.get("supertype"), card.get("nationalPokedexNumbers"),
                                card.get("types"), card.get("subtypes")]
    with open(IDENTIFICATION_POKEMON_CARD_ID_DATABASE_FILE, 'w', encoding='utf-8') as f:
        json.dump(pokemon_card_id, f, ensure_ascii=False, indent=4)

    # Index every card once by (type, subtype) and (pokedex, type, subtype) instead of
    # rescanning all cards inside nested loops over pokedex x type x subtype (which was
    # billions of iterations).  The emitted JSON is byte-for-byte the same structure:
    # every type/subtype (and pokedex) combination is present, with [] where empty and
    # ids kept in card-iteration order.
    type_index = {}
    num_index = {}
    for card in pokemon_card:
        cid = card["id"].replace("?", "question_hires")
        for type_ in (card.get("types") or []):
            for subtype in (card.get("subtypes") or []):
                type_index.setdefault((type_, subtype), []).append(cid)
                for pokedex_num in (card.get("nationalPokedexNumbers") or []):
                    num_index.setdefault((pokedex_num, type_, subtype), []).append(cid)

    pokemon_card_type = {type_: {subtype: type_index.get((type_, subtype), [])
                                 for subtype in subtypes}
                         for type_ in types}
    with open(IDENTIFICATION_POKEMON_CARD_TYPE_DATABASE_FILE, 'w', encoding='utf-8') as f:
        json.dump(pokemon_card_type, f, ensure_ascii=False, indent=4)

    pokemon_card_num = {pokedex_num: {type_: {subtype: num_index.get((pokedex_num, type_, subtype), [])
                                              for subtype in subtypes}
                                      for type_ in types}
                        for pokedex_num in range(1, MAX_NUMBER_NATIONAL_POKEDEX + 1)}
    with open(IDENTIFICATION_POKEMON_CARD_NUMBER_DATABASE_FILE, 'w', encoding='utf-8') as f:
        json.dump(pokemon_card_num, f, ensure_ascii=False, indent=4)

    with open(IDENTIFICATION_POKEMON_CARD_DATABASE_FILE, 'w', encoding='utf-8') as f:
        json.dump([c["id"].replace("?", "question_hires") for c in pokemon_card], f, ensure_ascii=False, indent=4)
    with open(IDENTIFICATION_TRAINER_CARD_DATABASE_FILE, 'w', encoding='utf-8') as f:
        json.dump([c["id"] for c in trainer_card], f, ensure_ascii=False, indent=4)
    with open(IDENTIFICATION_ENERGY_CARD_DATABASE_FILE, 'w', encoding='utf-8') as f:
        json.dump([c["id"] for c in energy_card], f, ensure_ascii=False, indent=4)

    pokemon_card_sets = [{
        "id": s.get("id"),
        "images": s.get("images", {}),
        "legalities": s.get("legalities", {}),
        "name": s.get("name"), "printedTotal": s.get("printedTotal"), "ptcgoCode": s.get("ptcgoCode"),
        "releaseDate": s.get("releaseDate"), "series": s.get("series"), "total": s.get("total"),
        "updatedAt": s.get("updatedAt"),
    } for s in _api_get("sets")]
    with open(POKEMON_CARD_SET_DATABASE_FILE, 'w', encoding='utf-8') as f:
        json.dump(pokemon_card_sets, f, ensure_ascii=False, indent=4)


# --------------------------------------------------------------------------- #
# DOTA helpers + synthetic DETECTION dataset
# --------------------------------------------------------------------------- #
def to_DOTA(data_annotations):
    DOTA_data_annotations = []
    for annotation in data_annotations:
        DOTA_data_annotation = []
        center = np.array([int(annotation[1] + annotation[3] / 2), int(annotation[2] + annotation[4] / 2), 1], dtype=np.float32)[0:2]
        RM_temp = cv2.getRotationMatrix2D(center, -annotation[5], 1.0)
        RM = np.zeros((3, 3)); RM[0:2, :] = RM_temp; RM[2, 2] = 1
        points = [
            np.array([annotation[1], annotation[2], 1]),
            np.array([annotation[1] + annotation[3], annotation[2], 1]),
            np.array([annotation[1] + annotation[3], annotation[2] + annotation[4], 1]),
            np.array([annotation[1], annotation[2] + annotation[4], 1]),
        ]
        for point in points:
            point_dst = np.dot(RM, point)
            point_dst = point_dst / point_dst[2]
            DOTA_data_annotation.append(int(point_dst[0]))
            DOTA_data_annotation.append(int(point_dst[1]))
        DOTA_data_annotation.append("card")
        DOTA_data_annotation.append(0)
        DOTA_data_annotations.append(DOTA_data_annotation)
    return DOTA_data_annotations


def read_annotation_file(txt_file):
    data_annotations = []
    with open(txt_file, "r") as f:
        for txt_annotation in f.readlines():
            data_annotations.append([int(item) for item in txt_annotation.split(" ")[:-2]])
    return data_annotations


def generate_image_card_detection(card_list, dsize=(WIDTH, HEIGHT)):
    data_annotations = []
    num_card_by_image = MIN_CARD_PER_DATA + np.random.randint(NUMBER_CARD_VARIATION_PER_DATA)
    background_num = np.random.randint(NUMBER_OF_BACKGROUND_AVAILABLE)
    image = cv2.imread(BACKGROUND_FOLDER_PATH + str(background_num) + ".jpg")
    image = cv2.resize(image, dsize, interpolation=cv2.INTER_AREA)

    for _ in range(num_card_by_image):
        if np.random.rand() > 0.33:
            card_number = np.random.randint(len(card_list))
            pokemon_card_image = cv2.imread(POKEMON_CARD_DATABASE_FOLDER_PATH + card_list[card_number])
        else:
            card_number = card_list.index("back1-1.jpg")
            pokemon_card_image = cv2.imread(POKEMON_CARD_DATABASE_FOLDER_PATH + "back1-1.jpg")

        old_shape = pokemon_card_image.shape
        zoom = np.random.uniform(MIN_CARD_ZOOM_INCREASE, MAX_CARD_ZOOM_INCREASE)
        new_shape = (int(old_shape[1] * zoom), int(old_shape[0] * zoom))
        pokemon_card_image = cv2.resize(pokemon_card_image, new_shape, interpolation=cv2.INTER_AREA)
        pokemon_card_image = cv2.cvtColor(pokemon_card_image, cv2.COLOR_BGR2BGRA)

        dx = np.random.randint(WIDTH) - 200
        dy = np.random.randint(HEIGHT) - 200
        angle = np.random.randint(360)
        pokemon_card_image = affine_transformation(pokemon_card_image, dx, dy, angle)

        data_annotations.append([card_list[card_number].replace(".jpg", ""), dx, dy, new_shape[0], new_shape[1], angle])

        alpha_mask = np.stack([pokemon_card_image[:, :, CHANNELS]] * 3, -1)
        image = np.where(alpha_mask, pokemon_card_image[:, :, 0:CHANNELS], image[:, :, :])

    image = data_augmentation(image)
    if np.random.rand() < PERSPECTIVE_AUGMENTATION_THRESHOLD:
        image, data_annotations = add_perspective_transform(image, data_annotations)
    return image, data_annotations


def generate_detection_dataset(folder_path, number_of_data, card_database_folder_path=POKEMON_CARD_DATABASE_FOLDER_PATH):
    create_nested_folders(folder_path)
    create_nested_folders(folder_path + "images/")
    create_nested_folders(folder_path + "annotations/")
    card_list = os.listdir(card_database_folder_path)
    for i in tqdm(range(number_of_data)):
        image, data_annotations = generate_image_card_detection(card_list)
        cv2.imwrite(folder_path + "images/" + str(i) + ".png", image)
        data_annotations = to_DOTA(data_annotations)
        annotations_string = ""
        for annotation in data_annotations:
            annotations_string += " ".join(str(item) for item in annotation) + " \n"
        with open(folder_path + "annotations/" + str(i) + ".txt", "w") as f:
            f.write(annotations_string)


def generate_detection_datasets(train=18000, valid=1000, test=1000):
    generate_detection_dataset(DETECTION_TRAINING_SET_FOLDER_PATH, train)
    generate_detection_dataset(DETECTION_VALIDATION_SET_FOLDER_PATH, valid)
    generate_detection_dataset(DETECTION_TEST_SET_FOLDER_PATH, test)


# --------------------------------------------------------------------------- #
# Synthetic ORIENTATION dataset
# --------------------------------------------------------------------------- #
def generate_image_card_orientation(card_list, bsize=(WIDTH, HEIGHT),
                                    dsize=(ORIENTATION_IMAGE_SIZE, ORIENTATION_IMAGE_SIZE),
                                    with_data_augmentation=True):
    background_num = np.random.randint(NUMBER_OF_BACKGROUND_AVAILABLE)
    image = cv2.imread(BACKGROUND_FOLDER_PATH + str(background_num) + ".jpg")
    image = cv2.resize(image, bsize, interpolation=cv2.INTER_AREA)

    card_number = np.random.randint(len(card_list))
    pokemon_card_image = cv2.imread(POKEMON_CARD_DATABASE_FOLDER_PATH + card_list[card_number])
    shape = pokemon_card_image.shape

    pokemon_card_image = cv2.resize(pokemon_card_image, (int(shape[1] / 1.1), int(shape[0] / 1.1)), interpolation=cv2.INTER_AREA)
    new_shape = pokemon_card_image.shape
    black_color = (0, 0, 0)
    pokemon_card_image = cv2.copyMakeBorder(pokemon_card_image, 50, 0, 100, 0, cv2.BORDER_CONSTANT, value=black_color)
    angle = np.random.randint(-20, 21)
    pokemon_card_image = affine_transformation(pokemon_card_image, 0, 0, angle)

    mask = np.full(new_shape, 255, dtype=np.uint8)
    mask = cv2.copyMakeBorder(mask, 50, 0, 100, 0, cv2.BORDER_CONSTANT, value=black_color)
    mask = affine_transformation(mask, 0, 0, angle)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    image = np.where(np.stack([mask > 0] * 3, -1), pokemon_card_image, image)
    if with_data_augmentation:
        image = data_augmentation(image)

    x_crop = np.random.randint(-50, 51)
    y_crop = np.random.randint(-50, 51)
    image = image[(50 + x_crop):(50 + shape[0] + x_crop), (50 + y_crop):(50 + y_crop + shape[1]), :]
    image = cv2.resize(image, dsize, interpolation=cv2.INTER_AREA)

    if np.random.random() < 0.5:
        class_label = "straight"
    else:
        class_label = "flip"
        image = cv2.rotate(image, cv2.ROTATE_180)
    return image, class_label


def generate_orientation_dataset(folder_path, number_of_data, card_database_folder_path=POKEMON_CARD_DATABASE_FOLDER_PATH, with_data_augmentation=True):
    create_nested_folders(folder_path)
    create_nested_folders(folder_path + "straight/")
    create_nested_folders(folder_path + "flip/")
    card_list = os.listdir(card_database_folder_path)
    for i in tqdm(range(number_of_data)):
        image, class_label = generate_image_card_orientation(card_list, with_data_augmentation=with_data_augmentation)
        cv2.imwrite(folder_path + class_label + "/" + str(i) + ".png", image)


def generate_orientation_datasets(train=18000, valid=1000, test=1000):
    generate_orientation_dataset(ORIENTATION_TRAINING_SET_FOLDER_PATH, train, with_data_augmentation=False)
    generate_orientation_dataset(ORIENTATION_VALIDATION_SET_FOLDER_PATH, valid, with_data_augmentation=True)
    generate_orientation_dataset(ORIENTATION_TEST_SET_FOLDER_PATH, test, with_data_augmentation=True)


# --------------------------------------------------------------------------- #
# Synthetic IDENTIFICATION dataset (one 224x224 image per card variant)
# --------------------------------------------------------------------------- #
def generate_image_card_identification(pokemon_card_image):
    background_num = np.random.randint(NUMBER_OF_BACKGROUND_AVAILABLE)
    image = cv2.imread(BACKGROUND_FOLDER_PATH + str(background_num) + ".jpg")
    image = cv2.resize(image, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)

    shape = pokemon_card_image.shape
    factor = max(1.0, shape[0] / HEIGHT, shape[1] / WIDTH) + 0.1
    pokemon_card_image = cv2.resize(pokemon_card_image, (int(shape[1] / factor), int(shape[0] / factor)), interpolation=cv2.INTER_AREA)
    new_shape = pokemon_card_image.shape
    black_color = (0, 0, 0)
    pokemon_card_image = cv2.copyMakeBorder(pokemon_card_image, 100, 0, 200, 0, cv2.BORDER_CONSTANT, value=black_color)
    angle = np.random.randint(-10, 11)
    pokemon_card_image = affine_transformation(pokemon_card_image, 0, 0, angle)

    mask = np.full(new_shape, 255, dtype=np.uint8)
    mask = cv2.copyMakeBorder(mask, 100, 0, 200, 0, cv2.BORDER_CONSTANT, value=black_color)
    mask = affine_transformation(mask, 0, 0, angle)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    image = np.where(np.stack([mask > 0] * 3, -1), pokemon_card_image, image)

    light_augmentation = np.random.rand()
    saturation_augmentation = np.random.rand()
    noise_augmentation = np.random.rand()
    perspective_augmentation = np.random.rand()

    if light_augmentation < LIGHT_AUGMENTATION_THRESHOLD:
        if light_augmentation < (LIGHT_AUGMENTATION_THRESHOLD / 2):
            neon_mask = random_neon_light_effect(NUMBER_NEON_STRIP, NEON_STRIP_WIDTH)
            image = increase_brightness(image, neon_mask, NEON_BRIGHTNESS)
        else:
            spot_mask = random_spot_light_effect(NUMBER_SPOT_STRIP, SPOT_STRIP_RADIUS)
            image = increase_brightness(image, spot_mask, SPOT_BRIGHTNESS)

    if saturation_augmentation < SATURATION_AUGMENTATION_THRESHOLD:
        saturation_mask = np.full((HEIGHT, WIDTH), 255)
        saturation_value = np.random.randint(256)
        if np.random.rand() < 0.5:
            saturation_value = -saturation_value
        image = increase_saturation(image, saturation_mask, saturation_value)

    if noise_augmentation < NOISE_AUGMENTATION_THRESHOLD:
        if noise_augmentation < (NOISE_AUGMENTATION_THRESHOLD / 2):
            image = white_noise(image)
        else:
            image = pepper_noise(image, PEPPER_NOISE_OCCURANCE)

    if perspective_augmentation < PERSPECTIVE_AUGMENTATION_THRESHOLD:
        d = [np.random.uniform(0, PERSPECTIVE_DISTORTION_AMPLITUDE) * WIDTH,
             np.random.uniform(0, PERSPECTIVE_DISTORTION_AMPLITUDE) * HEIGHT,
             np.random.uniform(1 - PERSPECTIVE_DISTORTION_AMPLITUDE, 1) * WIDTH,
             np.random.uniform(0, PERSPECTIVE_DISTORTION_AMPLITUDE) * HEIGHT,
             np.random.uniform(0, PERSPECTIVE_DISTORTION_AMPLITUDE) * WIDTH,
             np.random.uniform(1 - PERSPECTIVE_DISTORTION_AMPLITUDE, 1) * HEIGHT,
             np.random.uniform(1 - PERSPECTIVE_DISTORTION_AMPLITUDE, 1) * WIDTH,
             np.random.uniform(1 - PERSPECTIVE_DISTORTION_AMPLITUDE, 1) * HEIGHT]
        pts1 = np.float32([[d[0], d[1]], [d[2], d[3]], [d[4], d[5]], [d[6], d[7]]])
        pts2 = np.float32([[0, 0], [WIDTH, 0], [0, HEIGHT], [WIDTH, HEIGHT]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        image = cv2.warpPerspective(image, M, (WIDTH, HEIGHT))
        x_crop = np.random.randint(-50, 51)
        y_crop = np.random.randint(-50, 51)
        image = image[(50 + x_crop):(50 + shape[0] + x_crop), (100 + y_crop):(100 - shape[1] + new_shape[1] + y_crop + shape[1]), :]
    else:
        x_crop = np.random.randint(-50, 51)
        y_crop = np.random.randint(-50, 51)
        image = image[(75 + x_crop):(75 + shape[0] + x_crop), (175 + y_crop):(175 - shape[1] + new_shape[1] + y_crop + shape[1]), :]

    return cv2.resize(image, (IDENTIFICATION_IMAGE_SIZE, IDENTIFICATION_IMAGE_SIZE), interpolation=cv2.INTER_AREA)


def generate_identification_dataset(number_of_data_per_card, update=False):
    create_nested_folders(IDENTIFICATION_DATA_FOLDER_PATH)
    card_list = os.listdir(POKEMON_CARD_DATABASE_FOLDER_PATH)
    for i in tqdm(range(len(card_list))):
        pokemon_card_image = cv2.imread(POKEMON_CARD_DATABASE_FOLDER_PATH + card_list[i])
        pokemon_card_id = card_list[i].replace(".jpg", "")
        for j in range(number_of_data_per_card):
            dest = IDENTIFICATION_DATA_FOLDER_PATH + pokemon_card_id + "-" + str(j) + ".png"
            if update and os.path.exists(dest):
                continue
            cv2.imwrite(dest, generate_image_card_identification(pokemon_card_image))


# --------------------------------------------------------------------------- #
# Background augmentation (DTD / ADE20K)
# --------------------------------------------------------------------------- #
def append_dtd_to_backgrounds(dtd_dir, backgrounds_dir, resize_to):
    dtd_path = Path(dtd_dir)
    out_path = Path(backgrounds_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    idx_pattern = re.compile(r"^(\d+)\.jpg$")
    existing = [int(m.group(1)) for p in out_path.iterdir() if (m := idx_pattern.match(p.name))]
    next_idx = (max(existing) + 1) if existing else 0

    exts = {".jpg", ".jpeg"}
    dtd_images = sorted([p for p in dtd_path.rglob("*") if p.is_file() and p.suffix.lower() in exts], key=str)
    written = 0
    for src in dtd_images:
        img = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if img is None:
            continue
        if resize_to is not None:
            img = cv2.resize(img, resize_to, interpolation=cv2.INTER_AREA)
        if cv2.imwrite(str(out_path / f"{next_idx}.jpg"), img):
            next_idx += 1
            written += 1
    return written

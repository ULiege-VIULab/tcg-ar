"""
Central configuration for the whole PTCG-AR project.

This single module replaces the duplicated ``constants.py`` / ``utils_constants.py``
files of the original code base.  Every other package (installation, training,
evaluation, inference) imports its paths and hyper-parameters from here, so there
is exactly one place to change a value.

Two things are configurable through the environment so the code stays portable:

* ``PTCG_ASSETS_ROOT`` -- where the large ``assets`` tree lives (databases, datasets,
  model weights).  It defaults to the ``assets`` folder shipped inside this repository.
* ``PTCG_IDENTIFICATION_METHOD`` -- ``"arcface"`` (default) or ``"triplet"`` -- selects
  which identification model the training / evaluation / inference code uses.
* ``POKEMON_TCG_API_KEY`` -- a free Pokemon TCG API key (https://dev.pokemontcg.io),
  read from the environment.  Only needed to download/build the card databases.
"""

import os

# --------------------------------------------------------------------------- #
# Roots
# --------------------------------------------------------------------------- #
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))            # .../ptcg-ar/core
PROJECT_ROOT = os.path.dirname(_THIS_DIR)                          # .../ptcg-ar

# Default points at the ``assets`` folder shipped inside this repository.
# Override with PTCG_ASSETS_ROOT to keep the (large) data tree elsewhere.
_DEFAULT_ASSETS_ROOT = os.path.join(PROJECT_ROOT, "assets")
ASSETS_ROOT = os.path.abspath(os.environ.get("PTCG_ASSETS_ROOT", _DEFAULT_ASSETS_ROOT))

# Folder where training plots / logs are written.
OUTPUT_FOLDER_PATH = os.path.join(PROJECT_ROOT, "work_dirs", "outputs") + os.sep


def _p(*parts):
    """Join under ASSETS_ROOT and keep a trailing separator for folder paths."""
    return os.path.join(ASSETS_ROOT, *parts)


# --------------------------------------------------------------------------- #
# Data / general
# --------------------------------------------------------------------------- #
SETTINGS_PATH = os.path.join(PROJECT_ROOT, "settings.yaml")
BUFFER_LENGTH = 1

DATABASE_FOLDER_PATH = _p("database") + os.sep
DATASET_FOLDER_PATH = _p("AI database") + os.sep
MODEL_FOLDER_PATH = _p("AI models") + os.sep

POKEMON_DATABASE_URL = "https://www.pokepedia.fr/Liste_des_Pok%C3%A9mon_dans_l%27ordre_du_Pok%C3%A9dex_National"
POKEMON_DATABASE_FILE = DATABASE_FOLDER_PATH + "pokemon_database.json"
POKEMON_2D_ANIMATED_DATABASE = [
    "https://projectpokemon.org/home/docs/spriteindex_148/3d-models-generation-1-pokémon-r90/",
    "https://projectpokemon.org/home/docs/spriteindex_148/3d-models-generation-2-pokémon-r91/",
    "https://projectpokemon.org/home/docs/spriteindex_148/3d-models-generation-3-pokémon-r92/",
    "https://projectpokemon.org/home/docs/spriteindex_148/3d-models-generation-4-pokémon-r93/",
    "https://projectpokemon.org/home/docs/spriteindex_148/3d-models-generation-5-pokémon-r94/",
    "https://projectpokemon.org/home/docs/spriteindex_148/3d-models-generation-6-pokémon-r95/",
    "https://projectpokemon.org/home/docs/spriteindex_148/3d-models-generation-7-pokémon-r96/",
    "https://projectpokemon.org/home/docs/spriteindex_148/3d-models-generation-8-pokémon-r123/",
]
NO_POKEMON_PATH = DATABASE_FOLDER_PATH + "no_pokemon.png"
POKEMON_CARD_DATABASE_FILE = DATABASE_FOLDER_PATH + "pokemon_card.json"
POKEMON_CARD_SET_DATABASE_FILE = DATABASE_FOLDER_PATH + "pokemon_card_set_database.json"
CARD_ITEM_WIDTH = 130
CARD_ITEM_HEIGHT = 160

# --------------------------------------------------------------------------- #
# Video
# --------------------------------------------------------------------------- #
WIDTH = 1920
HEIGHT = 1080
CHANNELS = 3
FRAMERATE = 30
STREAM_FRAMERATE = 30

# --------------------------------------------------------------------------- #
# Render
# --------------------------------------------------------------------------- #
MAX_NUMBER_OF_POKEMON_PER_CARD = 3  # changing this requires editing Game_state and Multi_frame_renderer
AVAILABLE_MODELS = ["2D animated", "2D"]  # ["2D animated", "2D", "3D"]
CARD_SAME_LOCATION_TOLERANCE = 0.05

# Default / range (in percent) for the on-board 2D animated model size. The renderer
# applies MODEL_ZOOM_DEFAULT_PERCENT to every animated model when it is first loaded
# (the zenithal view), and the per-card Zoom slider shares the same bounds. The side
# (auxiliary) views render the same sprite smaller, at MODEL_ZOOM_SIDE_PERCENT, i.e. a
# factor MODEL_ZOOM_SIDE_PERCENT/MODEL_ZOOM_DEFAULT_PERCENT of the zenithal size.
MODEL_ZOOM_DEFAULT_PERCENT = 200
MODEL_ZOOM_SIDE_PERCENT = 150
MODEL_ZOOM_MIN_PERCENT = 0
MODEL_ZOOM_MAX_PERCENT = 400
POKEMON_2D_ANIMATED_DATABASE_FOLDER = "2D_animated_database"
POKEMON_2D_ANIMATED_MODEL_FOLDER = DATABASE_FOLDER_PATH + "2D_animated_database" + os.sep
POKEMON_2D_DATABASE_FOLDER = "2D_database"
POKEMON_2D_MODEL_FOLDER = DATABASE_FOLDER_PATH + "2D_database" + os.sep

# --------------------------------------------------------------------------- #
# User interface / broadcast overlay
# --------------------------------------------------------------------------- #
# UI assets (menu/window icons, caster templates, energy-type icons) live with the
# code, not under ASSETS_ROOT.  Generated by inference/generate_ui_assets.py.
UI_ASSETS_FOLDER = os.path.join(PROJECT_ROOT, "inference", "assets") + os.sep

# Pokemon TCG energy type -> accent colour (R, G, B), for HP bars / cost dots / panels.
ENERGY_TYPE_COLORS = {
    "Grass": (120, 200, 80), "Fire": (235, 90, 60), "Water": (70, 160, 235),
    "Lightning": (245, 205, 60), "Psychic": (190, 110, 200), "Fighting": (200, 120, 70),
    "Darkness": (70, 80, 95), "Metal": (160, 170, 185), "Fairy": (235, 130, 190),
    "Dragon": (200, 170, 70), "Colorless": (215, 215, 205),
}
DEFAULT_ENERGY_COLOR = (180, 190, 200)

# Base RTSP URL the output streams are published to (one path per output buffer id).
# Used by the RTSP sender and shown under each preview tile when enabled.
RTSP_BASE_URL = "rtsp://localhost:8554/ptcgAR/"

# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #
# A free Pokemon TCG API key (https://dev.pokemontcg.io) is required only to download
# or rebuild the card databases. Set it in the environment: PTCG / POKEMON_TCG_API_KEY.
POKEMON_TCG_API_KEY = os.environ.get("POKEMON_TCG_API_KEY", "")
POKEMON_CARD_DATABASE_FOLDER = "card_database"
POKEMON_CARD_DATABASE_FOLDER_PATH = DATABASE_FOLDER_PATH + "card_database" + os.sep
MIN_CARD_PER_DATA = 5
NUMBER_CARD_VARIATION_PER_DATA = 11
BACKGROUND_FOLDER_PATH = DATASET_FOLDER_PATH + "background" + os.sep
NUMBER_OF_BACKGROUND_AVAILABLE = 28000
MIN_CARD_ZOOM_INCREASE = 0.2
MAX_CARD_ZOOM_INCREASE = 0.33
LIGHT_AUGMENTATION_THRESHOLD = 0.33
LED_BRIGHTNESS = 100
NUMBER_NEON_STRIP = 6
NEON_STRIP_WIDTH = 150
NEON_BRIGHTNESS = 128
NUMBER_SPOT_STRIP = 6
SPOT_STRIP_RADIUS = 100
SPOT_BRIGHTNESS = 128
SATURATION_AUGMENTATION_THRESHOLD = 0.1
MAX_SATURATION_VALUE = 100
NOISE_AUGMENTATION_THRESHOLD = 0.2
PEPPER_NOISE_OCCURANCE = 0.1
PERSPECTIVE_AUGMENTATION_THRESHOLD = 0.2
PERSPECTIVE_DISTORTION_AMPLITUDE = 0.1

DETECTION_DATASET_FOLDER_PATH = DATASET_FOLDER_PATH + "detection" + os.sep
DETECTION_TRAINING_SET_FOLDER_PATH = DETECTION_DATASET_FOLDER_PATH + "training" + os.sep
DETECTION_VALIDATION_SET_FOLDER_PATH = DETECTION_DATASET_FOLDER_PATH + "validation" + os.sep
DETECTION_TEST_SET_FOLDER_PATH = DETECTION_DATASET_FOLDER_PATH + "test" + os.sep
DETECTION_REAL_SET_FOLDER_PATH = DETECTION_DATASET_FOLDER_PATH + "real" + os.sep

# --------------------------------------------------------------------------- #
# Real captured-and-annotated test set ("AI database/real/")
# Detection boxes (DOTA + YOLO) are ground truth; identities are annotated with
# the tool in inference/annotate_identities.py.  Used to evaluate detection,
# orientation and identification on real data.
# --------------------------------------------------------------------------- #
REAL_SET_FOLDER_PATH = DATASET_FOLDER_PATH + "real" + os.sep
REAL_IMAGES_FOLDER = REAL_SET_FOLDER_PATH + "images" + os.sep
REAL_ANNOTATIONS_FOLDER = REAL_SET_FOLDER_PATH + "annotations" + os.sep            # DOTA detection GT
REAL_ANNOTATIONS_YOLO_FOLDER = REAL_SET_FOLDER_PATH + "annotations_yolo" + os.sep  # YOLO-OBB source
REAL_IDENTITY_FOLDER = REAL_SET_FOLDER_PATH + "annotations_identity" + os.sep      # annotated identities
REAL_ORIENTATION_SET_FOLDER = REAL_SET_FOLDER_PATH + "orientation" + os.sep        # generated straight/flip set

_DETECTION_MODEL_DIR = MODEL_FOLDER_PATH + "Detection model" + os.sep
DETECTION_MODEL_PATH = _DETECTION_MODEL_DIR + "model" + os.sep          # train work_dir
DETECTION_CONFIG_PATH = _DETECTION_MODEL_DIR + "custom_oriented_rcnn_config.py"
DETECTION_CONFIG_PATH_REAL = _DETECTION_MODEL_DIR + "custom_oriented_rcnn_config_real.py"
DETECTION_WEIGHT_PATH = _DETECTION_MODEL_DIR + "custom_oriented_rcnn_weight.pth"

# --------------------------------------------------------------------------- #
# Orientation
# --------------------------------------------------------------------------- #
ORIENTATION_DATA_FOLDER_PATH = DATASET_FOLDER_PATH + "orientation" + os.sep
ORIENTATION_TRAINING_SET_FOLDER_PATH = ORIENTATION_DATA_FOLDER_PATH + "training" + os.sep
ORIENTATION_VALIDATION_SET_FOLDER_PATH = ORIENTATION_DATA_FOLDER_PATH + "validation" + os.sep
ORIENTATION_TEST_SET_FOLDER_PATH = ORIENTATION_DATA_FOLDER_PATH + "test" + os.sep
ORIENTATION_MODEL_FOLDER_PATH = MODEL_FOLDER_PATH + "Orientation model" + os.sep
ORIENTATION_MODEL_SAVE_PATH = ORIENTATION_MODEL_FOLDER_PATH + "orientation_classifier.pth"
ORIENTATION_MODEL_SAFE_PATH = ORIENTATION_MODEL_SAVE_PATH  # legacy alias (root spelling)
# Default deployed architecture, plus the comparison backbones (experiments only).
# The deployed efficientnet_b0 weight above is never overwritten by the alternatives.
ORIENTATION_DEFAULT_ARCH = "efficientnet_b0"
ORIENTATION_ARCH_WEIGHTS = {
    "efficientnet_b0": ORIENTATION_MODEL_SAVE_PATH,
    "resnet18": ORIENTATION_MODEL_FOLDER_PATH + "orientation_resnet18.pth",
    "mobilenet_v3_small": ORIENTATION_MODEL_FOLDER_PATH + "orientation_mobilenet_v3_small.pth",
    "shufflenet_v2_x1_0": ORIENTATION_MODEL_FOLDER_PATH + "orientation_shufflenet_v2_x1_0.pth",
}
ORIENTATION_IMAGE_SIZE = 224
ORIENTATION_VALID_SPLIT = 0.1
ORIENTATION_BATCH_SIZE = 16
ORIENTATION_NUM_WORKERS = 8
ORIENTATION_LR = 0.001
ORIENTATION_EPOCHS = 10
NB_CLASS_LABEL = 2

# --------------------------------------------------------------------------- #
# Identification
# --------------------------------------------------------------------------- #
IDENTIFICATION_METHOD = os.environ.get("PTCG_IDENTIFICATION_METHOD", "arcface").lower()

IDENTIFICATION_DATA_FOLDER_PATH = DATASET_FOLDER_PATH + "identification" + os.sep
_IDENTIFICATION_MODEL_DIR = MODEL_FOLDER_PATH + "Identification model" + os.sep
IDENTIFICATION_MODEL_SAVE_PATH = _IDENTIFICATION_MODEL_DIR + "identification_model_51.pth"
IDENTIFICATION_FINE_TUNE_MODEL_SAVE_PATH = _IDENTIFICATION_MODEL_DIR + "identification_fine_tune_model.pth"
# Legacy aliases (the original root code spelled it "SAFE").
IDENTIFICATION_MODEL_SAFE_PATH = IDENTIFICATION_MODEL_SAVE_PATH
IDENTIFICATION_FINE_TUNE_MODEL_SAFE_PATH = IDENTIFICATION_FINE_TUNE_MODEL_SAVE_PATH
# ArcFace weights (separate from the triplet weights).
IDENTIFICATION_ARCFACE_MODEL_SAVE_PATH = _IDENTIFICATION_MODEL_DIR + "identification_model_51_arcface.pth"
IDENTIFICATION_ARCFACE_FINE_TUNE_MODEL_SAVE_PATH = _IDENTIFICATION_MODEL_DIR + "identification_arcface_fine_tune_model.pth"

IDENTIFICATION_POKEMON_CARD_ID_DATABASE_FILE = DATABASE_FOLDER_PATH + "pokemon_card_id_database.json"
IDENTIFICATION_POKEMON_CARD_NUMBER_DATABASE_FILE = DATABASE_FOLDER_PATH + "pokemon_card_number_database.json"
IDENTIFICATION_POKEMON_CARD_TYPE_DATABASE_FILE = DATABASE_FOLDER_PATH + "pokemon_card_type_database.json"
IDENTIFICATION_POKEMON_CARD_DATABASE_FILE = DATABASE_FOLDER_PATH + "pokemon_card_database.json"
IDENTIFICATION_TRAINER_CARD_DATABASE_FILE = DATABASE_FOLDER_PATH + "trainer_card_database.json"
IDENTIFICATION_ENERGY_CARD_DATABASE_FILE = DATABASE_FOLDER_PATH + "energy_card_database.json"
# Folder holding every named deck (.txt), including the active ``deck.txt`` that the
# recognizer reads. The deck-selection GUI saves/loads named decks here; on launch the
# union of the two player decks is written to IDENTIFICATION_FINE_TUNE_DECK_LIST_FILE.
DECK_LIST_FOLDER = _p("deck list") + os.sep
IDENTIFICATION_FINE_TUNE_DECK_LIST_FILE = DECK_LIST_FOLDER + "deck.txt"
EMBEDDING_CACHE_PATH = _p("embedding_cache") + os.sep

# The card back is a real, recognisable card (a face-down card should identify as
# this).  It is not in the Pokemon TCG API metadata, so it is handled explicitly.
BACK_CARD_ID = "back1-1"
# Cards whose stored image is actually a card back (wrong scans/downloads), detected
# with ArcFace by installation/find_wrong_scans.py.  Excluded from training/inference.
WRONG_SCAN_LIST_FILE = DATABASE_FOLDER_PATH + "wrong_scan_cards.json"

MAX_NUMBER_NATIONAL_POKEDEX = 1025
POSITIVE_DATA_NUMBER = 6
IDENTIFICATION_IMAGE_SIZE = 224
IDENTIFICATION_VALID_SPLIT = 1
IDENTIFICATION_BATCH_SIZE = 16
IDENTIFICATION_FINE_TUNE_BATCH_SIZE = 8
TRAIN_IDENTIFICATION_NUM_WORKERS = 8
VALID_IDENTIFICATION_NUM_WORKERS = 4
IDENTIFICATION_LR = 0.001
IDENTIFICATION_FINE_TUNE_LR = 0.0001
IDENTIFICATION_EPOCHS = 20
IDENTIFICATION_FINE_TUNE_EPOCHS = 10
IDENTIFICATION_OUT_FEATURES = 128

# ArcFace specific hyper-parameters.
ARCFACE_EMBEDDING_SIZE = 512
ARCFACE_SCALE = 30.0
ARCFACE_MARGIN = 0.50


def model_save_path(method=None, fine_tuned=False):
    """Return the weight path for the requested identification method."""
    method = (method or IDENTIFICATION_METHOD).lower()
    if method == "arcface":
        return IDENTIFICATION_ARCFACE_FINE_TUNE_MODEL_SAVE_PATH if fine_tuned else IDENTIFICATION_ARCFACE_MODEL_SAVE_PATH
    return IDENTIFICATION_FINE_TUNE_MODEL_SAVE_PATH if fine_tuned else IDENTIFICATION_MODEL_SAVE_PATH

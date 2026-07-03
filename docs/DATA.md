# Data & assets

Everything lives under `assets/` (overridable with `PTCG_ASSETS_ROOT`). To keep the git
repository small, only data that **cannot be downloaded or generated** is committed; the
rest is fetched with `python -m scripts.download_assets` or built with `installation.install`.

## Asset map

| Path | What | How to obtain |
|---|---|---|
| `assets/AI models/Detection model/custom_oriented_rcnn_weight.pth` | Oriented R-CNN weights (314 MB) | **download** (`download_assets.py --only models`) |
| `assets/AI models/Detection model/custom_oriented_rcnn_config.py` | detector config | **shipped** |
| `assets/AI models/Orientation model/orientation_classifier.pth` | EfficientNet-B0 weights (46 MB) | **download** |
| `assets/AI models/Identification model/identification_model_51_arcface.pth` | ArcFace weights (227 MB) | **download** |
| `assets/AI database/real/` | real evaluation set: images (169 MB) + manual annotations | annotations **shipped**, images **download** |
| `assets/AI database/background/` | background textures (~28 k) for synthetic generation | **download** (separate authors' link) |
| `assets/AI database/{detection,orientation,identification}/` | synthetic training datasets | **build** (`installation.install --*-dataset`) |
| `assets/database/card_database/` | ~20 k reference card images | **build** (`installation.install --cards`, Pokémon TCG API) |
| `assets/database/card_database/back1-1.jpg` | the card back (not in the API) | **shipped** |
| `assets/database/2D_animated_database/`, `2D_database/` | creature sprites | **build** (`installation.install --sprites`) |
| `assets/database/*.json` | card metadata (encyclopedia, id/number/type/set, …) | **build** (`installation.install --metadata`) |
| `assets/database/no_pokemon.png` | placeholder for non-creature cards | **shipped** |
| `assets/database/wrong_scan_cards.json` | list of wrongly-scanned reference cards | **shipped** (rebuild with `installation.find_wrong_scans`) |
| `assets/deck list/*.txt` | example player decks | **shipped** |

## Building the databases (Pokémon TCG API)
A free API key is required (only for building the databases): <https://dev.pokemontcg.io>.
Commands are shown for **Windows PowerShell** (Linux/macOS: replace `$env:VAR="..."` with
`export VAR=...`).
```powershell
$env:POKEMON_TCG_API_KEY = "your-key"          # or: setx POKEMON_TCG_API_KEY your-key (new shells)
python -m installation.install --metadata     # pokémon list + card encyclopedia + JSON databases
python -m installation.install --cards        # download the ~20k card images
python -m installation.install --sprites      # download the 2D / 2D-animated sprites
# or everything at once (also generates the synthetic datasets):
python -m installation.install --all
```

## Generating the synthetic datasets
Needs the card images, sprites and the **background** pack (see the map above).
```powershell
python -m installation.install --detection-dataset --orientation-dataset --identification-dataset
```
Approximate sizes: detection 18,000 / 1,000 / 1,000 (train/val/test) 1920×1080 images;
orientation 18,000 / 1,000 / 1,000 224×224 crops; identification six 224×224 crops per
card (≈122 k total). See the paper's supplementary material for the full generation and
training parameters.

## Updating download links (maintainers)
`scripts/download_assets.py` contains a `MANIFEST` list of Google Drive file IDs
(one per artifact). To update a link, replace the file ID string in that list, or
set the matching `*_URL` environment variable to any Drive share URL or bare file ID —
the script parses both forms automatically.

The background texture pack has no link yet (`None` in the manifest); set
`TCGAR_BACKGROUND_URL` or update the manifest entry when the link is available.

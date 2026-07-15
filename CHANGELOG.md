# Changelog

All notable changes to the **TCG-AR Windows installer** are documented here.
The version is the one shown in the wizard and in the `TCG-AR-Setup-<version>.exe`
filename (single-sourced from `pyproject.toml`). This project follows
[Semantic Versioning](https://semver.org): patch = fixes, minor = new features,
major = breaking changes. Each release on the
[Releases page](https://github.com/ULiege-VIULab/tcg-ar/releases) mirrors its
entry below.

## [1.2.0] — 2026-07-14

Render-pipeline performance.

- **Fixed the slow Scarlet/Violet sprite rendering** (~1.8 fps → ~150 fps for the
  same board). The renderer was re-decoding every card's GIF twice per frame; it
  now keeps decoded sprites in an **additive LRU cache** (`MODEL_CACHE_MAX`) and
  preloads every animation role once, so per-view renders and active↔idle
  switches never re-decode. SV rendering is now as fast as the default sprites.
- The Zoom slider now rescales the actually-displayed sprite when SV is enabled.

## [1.1.0] — 2026-07-14

Higher-quality animated sprites.

- **Scarlet/Violet animated sprites** (from the Scavio GIFs Tumblr): high-quality
  idle + battle animations for the SV roster (National Dex #0001–1025),
  downloaded and re-encoded (downscaled + frame-capped) during the sprite build.
- **Opt-in** via the **"Enable S&V animated sprites"** menu toggle (off by
  default, so the default look and performance are unchanged). When on, they are
  the primary sprite source.
- The **active Pokémon** (nearest the board centre on each side) plays its
  **battle** animation; benched Pokémon play the **idle** animation with the
  occasional fidget.
- SV sprites are downscaled on load to render at the same on-screen size as the
  old sprites — uniform sizing and light per-frame compositing.
- Automatic fallback per Pokémon: Scavio SV → the previous gen 1–8 animated
  sprites → the static 2D sprite (so shiny and uncovered Pokémon are unchanged).
- Building the sprites (part of the card-database step) now also fetches the SV
  set; it adds time and a few GB. "Repair" verifies the SV set and re-downloads
  only what is missing; "Update card database" refreshes it after new releases.

## [1.0.0] — 2026-07-14

First public release of the one-click Windows installer.

- Self-contained wizard (`TCG-AR-Setup-1.0.0.exe`): no Python, conda or command
  line required. Installs per-user with no admin rights.
- Automatic NVIDIA GPU detection selecting the matching stack — both run the same
  OpenMMLab 2.x code path:
  - Blackwell (RTX 50): Python 3.14 + CUDA 13.2 (torch 2.12).
  - Turing / Ampere / Ada (RTX 20/30/40): Python 3.11 + CUDA 11.8 (torch 2.3).
- Component selection (AI models, card database + sprites, precomputed embeddings)
  and an in-wizard Pokémon TCG API-key page.
- Downloads and setup run **inside the installer window** with live progress bars
  (overall + per-step tqdm), a live log view, and inline error reporting with
  **Retry**; every step is resumable.
- Start-Menu entries: launch **TCG-AR**, **Update card database** (new set
  releases, no retraining), and **Setup (repair)**.
- Maintenance mode on re-run: **repair/resume** (fast — validates existing data
  instead of re-downloading) or **update the card database**.
- In-place upgrades preserve the card database, models and settings; uninstall
  optionally keeps the downloaded data.

[1.2.0]: https://github.com/ULiege-VIULab/tcg-ar/releases/tag/v1.2.0
[1.1.0]: https://github.com/ULiege-VIULab/tcg-ar/releases/tag/v1.1.0
[1.0.0]: https://github.com/ULiege-VIULab/tcg-ar/releases/tag/v1.0.0

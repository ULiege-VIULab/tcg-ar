# Changelog

All notable changes to the **TCG-AR Windows installer** are documented here.
The version is the one shown in the wizard and in the `TCG-AR-Setup-<version>.exe`
filename (single-sourced from `pyproject.toml`). This project follows
[Semantic Versioning](https://semver.org): patch = fixes, minor = new features,
major = breaking changes. Each release on the
[Releases page](https://github.com/ULiege-VIULab/tcg-ar/releases) mirrors its
entry below.

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

[1.0.0]: https://github.com/ULiege-VIULab/tcg-ar/releases/tag/v1.0.0

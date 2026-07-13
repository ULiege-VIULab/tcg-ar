# TCG-AR Windows installer

Builds `TCG-AR-Setup-<version>.exe` — a wizard that installs TCG-AR on end-user
machines **with no prerequisites** (no Python, no conda, no manual pip). The
installer is small (~30 MB); everything heavy is downloaded during setup.

## What the installer does on a user's machine

1. **Maintenance page**: when an existing installation is detected (registry
   uninstall key + `state\install.json`), the first page offers *Install/upgrade*,
   *Repair/resume*, or *Update the card database* (new sets; skips all
   configuration pages and reuses the saved stack/API key).
2. Copies the application snapshot + a pinned [`uv`](https://github.com/astral-sh/uv)
   binary to `%LOCALAPPDATA%\Programs\TCG-AR` (per-user, **no admin rights**).
3. Detects the NVIDIA GPU (`nvidia-smi` compute capability) and picks the stack:
   - compute cap >= 10.0 (RTX 50 / Blackwell) → Python 3.14 + CUDA 13.2 + mmrotate 1.x
   - 7.5 – 9.x (RTX 20/30/40, Turing/Ampere/Ada/Hopper) → Python 3.11 + CUDA 11.8 + mmrotate 0.3.4
   The user can override the choice; no GPU → warning with abort default.
4. Wizard options: pre-trained models (required), card database + sprites
   (needs the user's free [Pokémon TCG API key](https://dev.pokemontcg.io) —
   collected on a wizard page), pre-computed embeddings.
5. After file copy, an **embedded progress page** supervises the provisioning —
   no console window. The worker (`bootstrap.ps1 -Gui`, or `update_carddb.ps1 -Gui`
   in update mode) runs **hidden** and reports through `state\progress.json`
   (step/total/name/status/message) + per-step logs under `logs\`; the wizard
   polls these (600 ms timer) to drive an overall bar, a current-step bar
   (percentages parsed from pip/gdown/tqdm output), a live log tail, and
   inline error display with a **Retry** button. Two hard-won constraints:
   - the worker is launched **via explorer.exe** (Inno's RedirectionGuard
     mitigation is inherited by children and breaks uv's junctions with
     os error 448);
   - in GUI mode external commands write to their step log through `cmd`
     redirection (a PowerShell pipeline would deliver `\r`-progress only
     line-by-line, and Start-Transcript misses native output entirely);
   - because the step logs grow to many MB and Inno can only read whole
     files, a hidden **sidecar** (`tailer.ps1`, child of the worker) seeks
     the tail of the active log every ~400 ms, strips ANSI escapes, and
     distills it into `state\live.txt` (percent + the current tqdm line,
     shown above the sub-bar) and `state\tail.txt` (last lines for the
     wizard's log view), both UTF-8-with-BOM so Inno decodes the tqdm
     block characters correctly.
   The provisioning itself: uv installs a **private CPython**, creates a venv,
   installs torch/mm*/deps, applies `scripts.patch_mmlibs` (Blackwell),
   downloads model weights, builds the card DB and the embedding cache.
   Every step is idempotent (`state\*.done` markers).
6. Isolation: nothing outside the install folder is modified except the
   `POKEMON_TCG_API_KEY` user environment variable (removed on uninstall).
   No PATH edits, no system Python, no registry Python entries.
7. Start-Menu shortcut lifecycle (consoles appear **only** via these
   shortcuts, never during the wizard):
   - **"TCG-AR Setup (repair)"** — created by the installer; always present.
   - **"TCG-AR"** (launch) — created by `bootstrap.ps1` **on success only**, so a
     half-finished setup never leaves a broken shortcut.
   - **"TCG-AR - Update card database"** — created when the card-DB component
     completes; runs `update_carddb.ps1` (metadata + missing cards + sprites +
     embedding refresh, resumable) for new-set releases. No retraining needed.

MediaMTX (RTSP server) is *not* bundled: `mediamtx-py` downloads it on the
app's first launch (one-time Windows Firewall "Allow access" prompt).

## Building a release

Prerequisites (maintainer machine only):

```powershell
winget install JRSoftware.InnoSetup     # Inno Setup 6
# git must be available; internet access the first time (fetches uv.exe)
```

Build:

```powershell
powershell -ExecutionPolicy Bypass -File installer\build.ps1
# → installer\dist\TCG-AR-Setup-<version>.exe
```

- The version is read from `pyproject.toml` (single source of truth — bump it there).
- By default the snapshot is `git archive HEAD` (**committed files only**).
  Use `-FromWorkingTree` to test uncommitted changes.
- The pinned uv release + its SHA-256 live at the top of `build.ps1`
  (`$UvVersion` / `$UvSha256`). When bumping: set the version, set
  `$UvSha256 = 'PIN-ME'`, run the build once — it prints the new hash and
  aborts; pin it and re-run.
- **mmcv wheel (Blackwell)**: mmcv 2.x has no official prebuilt wheel for
  Python 3.14 / CUDA 13.2, and end users have no compiler. Build one once in
  the `tcgar-py314` environment and drop it in `vendor\`:
  `pip wheel mmcv==2.2.0 --no-deps -w installer\vendor\`. The build embeds it
  into the installer; without it, Blackwell installs fall back to
  `TCGAR_MMCV_WHEEL_URL` or a source build (warned loudly at build time).
  Rebuild the wheel whenever the pinned torch or Python version changes.

### Releasing

Upload `dist\TCG-AR-Setup-<version>.exe` to a GitHub Release. Users re-running
a newer installer over an existing install get an **in-place upgrade**: code is
refreshed, while `app\assets` (models, card DB, embeddings), `settings.yaml`,
and the Python environment survive. So: code change → bump version →
`build.ps1` → upload. Nothing is ever rebuilt "from scratch".

### Code signing (optional)

The exe is unsigned by default, so downloads trigger Windows SmartScreen
("Windows protected your PC" → *More info* → *Run anyway*). To sign, set
`TCGAR_SIGN_CERT` (path to .pfx) and `TCGAR_SIGN_PASS` before building; an
OV/EV certificate also builds SmartScreen reputation over time.

## Testing checklist

- Fresh install on a machine with an RTX 50 (Stack A) and an RTX 20/30/40 (Stack B).
- Machine without an NVIDIA GPU → warning path.
- Kill the bootstrap console mid-download → "TCG-AR Setup (repair)" resumes.
- Upgrade: install v_N, build v_N+1, install over it → assets + venv preserved.
- Uninstall, both answers to the "delete downloaded data?" prompt.

## Files

| File | Purpose |
|---|---|
| `TCG-AR.iss` | Inno Setup script: wizard pages (GPU detect, API key), upgrade/uninstall logic |
| `bootstrap.ps1` | Post-install provisioning (Python, venv, deps, assets); also the repair tool |
| `update_carddb.ps1` | New-set card-DB refresh, launched from its Start-Menu shortcut |
| `build.ps1` | Maintainer build: stage snapshot, vendor uv, compile with ISCC |
| `assets/tcg-ar.ico` | App/installer icon (generated from the in-app pokeball drawing) |
| `stage/`, `vendor/`, `dist/` | Build artifacts (git-ignored) |

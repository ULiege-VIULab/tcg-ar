# TCG-AR post-install bootstrap.
#
# Provisions a private Python + virtual environment inside the install folder
# and downloads/builds everything TCG-AR needs. Idempotent: every step is
# guarded by a marker file under state\, so re-running (or the Start Menu
# "TCG-AR Setup (repair)" shortcut) resumes exactly where a previous run
# stopped. Nothing outside the install folder is touched, except the
# POKEMON_TCG_API_KEY user environment variable (when a key was provided).
#
# Two front-ends drive this script:
#   - the installer wizard (-Gui): runs hidden; progress is reported through
#     state\progress.json + per-step logs under logs\, which the wizard's
#     embedded progress page polls;
#   - the Start Menu "TCG-AR Setup (repair)" shortcut: a visible console.
#
# Written for Windows PowerShell 5.1 (the stock powershell.exe).

param(
    # Re-run every provisioning step even if its marker says it succeeded
    # before (package installs become fast no-ops; card download fetches
    # only missing images).
    [switch]$Repair,
    # Hidden/GUI mode: no console interaction; report through progress.json.
    [switch]$Gui,
    # Skip the final "Press Enter to close" prompt (for unattended runs/tests).
    [switch]$NoPause,
    # Install root (the folder that contains app\, tools\, installer\).
    [string]$InstallDir = (Split-Path -Parent $PSScriptRoot)
)

$ErrorActionPreference = 'Stop'

# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #
$AppDir   = Join-Path $InstallDir 'app'
$VenvDir  = Join-Path $InstallDir 'env'
$PyDir    = Join-Path $InstallDir 'python'
$UvExe    = Join-Path $InstallDir 'tools\uv.exe'
$StateDir = Join-Path $InstallDir 'state'
$LogsDir  = Join-Path $InstallDir 'logs'
$VenvPy   = Join-Path $VenvDir 'Scripts\python.exe'

foreach ($d in @($StateDir, $LogsDir)) {
    if (-not (Test-Path $d)) { New-Item -ItemType Directory -Path $d | Out-Null }
}

$LogFile = Join-Path $LogsDir ("bootstrap-{0}.log" -f (Get-Date -Format 'yyyyMMdd-HHmmss'))
Start-Transcript -Path $LogFile | Out-Null

# Keep uv fully self-contained: Pythons and download cache live in the
# install folder, never in the user profile.
$env:UV_PYTHON_INSTALL_DIR = $PyDir
$env:UV_CACHE_DIR          = Join-Path $InstallDir 'tools\uv-cache'
# Python subprocesses print unicode (arrows, progress bars); force UTF-8 so a
# legacy console code page never turns that into a UnicodeEncodeError.
$env:PYTHONUTF8 = '1'

# --------------------------------------------------------------------------- #
# Progress protocol (consumed by the installer wizard's progress page)
# --------------------------------------------------------------------------- #
$ProgressPath = Join-Path $StateDir 'progress.json'
$script:StepNum     = 0
$script:TotalSteps  = 0
$script:CurStepName = 'Starting'
$script:CurStepLog  = ''
$Utf8NoBom = New-Object System.Text.UTF8Encoding($false)

function Write-ProgressState([string]$Status, [string]$Message = '') {
    $obj = @{
        mode    = 'install'
        pid     = $PID
        step    = $script:StepNum
        total   = $script:TotalSteps
        name    = $script:CurStepName
        status  = $Status
        message = $Message
        steplog = $script:CurStepLog
    }
    $tmp = "$ProgressPath.tmp"
    [IO.File]::WriteAllText($tmp, ($obj | ConvertTo-Json -Compress), $Utf8NoBom)
    Move-Item -Force $tmp $ProgressPath
}

# GUI mode: report immediately (the wizard is waiting) and start the sidecar
# that distills the step logs into state\live.txt / state\tail.txt for the
# wizard's live progress bar and log view.
$TailerProc = $null
if ($Gui) {
    foreach ($f in @('live.txt', 'tail.txt')) {
        $p = Join-Path $StateDir $f
        if (Test-Path $p) { Remove-Item -Force $p }
    }
    $script:CurStepName = 'Preparing the setup...'
    Write-ProgressState 'running'
    $TailerProc = Start-Process -PassThru -WindowStyle Hidden `
        -FilePath "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe" `
        -ArgumentList ('-NoProfile -ExecutionPolicy Bypass -File "{0}" -InstallDir "{1}" -ParentPid {2}' -f `
            (Join-Path $PSScriptRoot 'tailer.ps1'), $InstallDir, $PID)
}

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
function Write-Banner([string]$Text) {
    Write-Host ''
    Write-Host ('=' * 70) -ForegroundColor Cyan
    Write-Host "  $Text" -ForegroundColor Cyan
    Write-Host ('=' * 70) -ForegroundColor Cyan
}

function Invoke-External {
    # Runs an external command. In GUI mode the command is executed through a
    # small batch file so its stdout/stderr stream *live* into the per-step
    # log file (tqdm/pip percentages update via \r, which a PowerShell
    # pipeline would only deliver line-by-line). Returns the exit code;
    # throws unless it is listed in $OkExitCodes.
    param([string]$Exe, [string[]]$Arguments, [string]$WorkDir, [int[]]$OkExitCodes = @(0))
    $quoted = $Arguments | ForEach-Object { if ($_ -match '[\s"]') { '"' + $_ + '"' } else { $_ } }
    $pretty = '"{0}" {1}' -f $Exe, ($quoted -join ' ')
    Write-Host "> $pretty" -ForegroundColor DarkGray
    if ($script:CurStepLog) {
        [IO.File]::AppendAllText($script:CurStepLog, "> $pretty`r`n", $Utf8NoBom)
    }
    $prev = $null
    if ($WorkDir) { $prev = Get-Location; Set-Location $WorkDir }
    try {
        if ($Gui) {
            $batch = Join-Path $StateDir 'step.cmd'
            $content = "@echo off`r`nchcp 65001 >nul`r`n" +
                       ('"{0}" {1} >> "{2}" 2>&1' -f $Exe, ($quoted -join ' '), $script:CurStepLog) + "`r`n"
            [IO.File]::WriteAllText($batch, $content, $Utf8NoBom)
            & $env:ComSpec '/c' $batch
        } else {
            & $Exe @Arguments
        }
        $code = $LASTEXITCODE
        if ($OkExitCodes -notcontains $code) {
            throw "Command failed (exit $code): $pretty"
        }
        return $code
    } finally {
        if ($prev) { Set-Location $prev }
    }
}

function New-StartMenuShortcut {
    param([string]$Name, [string]$Target, [string]$Arguments, [string]$WorkDir, [string]$Icon)
    $menuDir = Join-Path ([Environment]::GetFolderPath('Programs')) 'TCG-AR'
    if (-not (Test-Path $menuDir)) { New-Item -ItemType Directory -Path $menuDir | Out-Null }
    $shell = New-Object -ComObject WScript.Shell
    $lnk = $shell.CreateShortcut((Join-Path $menuDir "$Name.lnk"))
    $lnk.TargetPath = $Target
    $lnk.Arguments = $Arguments
    $lnk.WorkingDirectory = $WorkDir
    if ($Icon) { $lnk.IconLocation = $Icon }
    $lnk.Save()
    Write-Host "[ok] Start Menu shortcut: $Name" -ForegroundColor Green
}

function Set-Done([string]$Name) {
    Get-Date -Format o | Out-File -Encoding ascii (Join-Path $StateDir "$Name.done")
}

# Cheap data validators: is the produced data still there and plausible?
# (Presence + size/count heuristics only - integrity is the app's business.)
$script:WeightSizes = @{
    'assets\AI models\Detection model\custom_oriented_rcnn_weight.pth'          = 200MB
    'assets\AI models\Orientation model\orientation_classifier.pth'             = 20MB
    'assets\AI models\Identification model\identification_model_51_arcface.pth' = 150MB
}
function Test-CountAtLeast([string]$RelDir, [int]$Min) {
    $p = Join-Path $AppDir $RelDir
    if (-not (Test-Path $p)) { return $false }
    return (Get-ChildItem $p -File -ErrorAction SilentlyContinue | Select-Object -First ($Min + 1) |
        Measure-Object).Count -ge $Min
}
function Test-ModelsData {
    foreach ($w in $script:WeightSizes.Keys) {
        $p = Join-Path $AppDir $w
        if (-not (Test-Path $p)) { return $false }
        if ((Get-Item $p).Length -lt $script:WeightSizes[$w]) { return $false }
    }
    return $true
}
function Test-MetadataData {
    $card = Join-Path $AppDir 'assets\database\pokemon_card.json'
    if (-not (Test-Path $card)) { return $false }
    if ((Get-Item $card).Length -lt 5MB) { return $false }
    return (Get-ChildItem (Join-Path $AppDir 'assets\database') -Filter '*.json' -File |
        Measure-Object).Count -ge 5
}
function Test-SpritesData {
    return (Test-CountAtLeast 'assets\database\2D_database' 300) -and
           (Test-CountAtLeast 'assets\database\2D_animated_database' 50)
}
function Test-CardsData {
    return Test-CountAtLeast 'assets\database\card_database' 15000
}
function Test-EmbeddingsData {
    return Test-CountAtLeast 'assets\embedding_cache' 1
}

function Invoke-Step {
    # Skip rules:
    #  - normal run: skip when the marker exists AND the data validator (if
    #    any) confirms the produced data is still there - so a deleted asset
    #    re-downloads even if its marker survived.
    #  - repair run: environment steps (no validator) always re-run (fast pip
    #    no-ops); data steps (with validator) are skipped when their data
    #    verifies - re-checking the multi-GB card database would otherwise
    #    take a long time for nothing. Content refreshes are the job of the
    #    "Update the card database" mode, not of repair.
    param([string]$Name, [string]$Title, [scriptblock]$Body, [switch]$NoMarker, [scriptblock]$ValidIf)
    $script:StepNum++
    $script:CurStepName = $Title
    $script:CurStepLog = Join-Path $LogsDir ("step-{0:D2}-{1}.log" -f $script:StepNum, $Name)
    # Fresh log per attempt so the wizard's tail view starts clean.
    if (Test-Path $script:CurStepLog) { Remove-Item -Force $script:CurStepLog }
    Write-ProgressState 'running'
    $haveMarker = (-not $NoMarker) -and (Test-Path (Join-Path $StateDir "$Name.done"))
    $canSkip = $haveMarker
    if ($canSkip -and $Repair -and -not $ValidIf) { $canSkip = $false }
    if ($canSkip -and $ValidIf) { $canSkip = [bool](& $ValidIf) }
    if ($canSkip) {
        Write-Host "[skip] $Title (already done and verified)" -ForegroundColor DarkGreen
        [IO.File]::AppendAllText($script:CurStepLog, "Already done and verified - skipped.`r`n", $Utf8NoBom)
        return
    }
    Write-Banner $Title
    & $Body
    if (-not $NoMarker) { Set-Done $Name }
    Write-Host "[ok] $Title" -ForegroundColor Green
}

# --------------------------------------------------------------------------- #
# Read the choices the installer wizard recorded
# --------------------------------------------------------------------------- #
$InstallJsonPath = Join-Path $StateDir 'install.json'
if (-not (Test-Path $InstallJsonPath)) {
    if ($Gui) { Write-ProgressState 'failed' 'Missing state\install.json - run the TCG-AR installer first.' }
    throw "Missing $InstallJsonPath - run the TCG-AR installer first."
}
$Choices = Get-Content $InstallJsonPath -Raw | ConvertFrom-Json

$Stack          = $Choices.stack                      # 'blackwell' | 'cu118'
$WantModels     = [bool]$Choices.components.models
$WantCardDb     = [bool]$Choices.components.carddb
$WantEmbeddings = [bool]$Choices.components.embeddings
$ApiKey         = [string]$Choices.api_key
$DoEmbeddings   = $WantEmbeddings -and $WantModels -and $WantCardDb

# Both stacks run the SAME OpenMMLab 2.x code path (mmcv 2.2 / mmdet 3.3 /
# mmrotate 1.0.0rc1 + patch_mmlibs) - only Python/torch/CUDA differ. The
# recipe id is recorded in stack.json; when a release changes a recipe, the
# next run rebuilds the venv automatically.
switch ($Stack) {
    'blackwell' { $PyVersion = '3.14'; $RecipeId = 'mm2-torch2.12-cu132' }
    'cu118'     { $PyVersion = '3.11'; $RecipeId = 'mm2-torch2.3-cu118' }
    default     {
        if ($Gui) { Write-ProgressState 'failed' "Unknown stack '$Stack' in install.json." }
        throw "Unknown stack '$Stack' in install.json (expected 'blackwell' or 'cu118')."
    }
}

$script:TotalSteps = 5
if ($WantModels)   { $script:TotalSteps += 1 }
if ($WantCardDb)   { $script:TotalSteps += 3 }
if ($DoEmbeddings) { $script:TotalSteps += 1 }

Write-Banner "TCG-AR setup - stack: $Stack (Python $PyVersion)$(if ($Repair) { ' [repair mode]' })"
Write-Host "Install folder : $InstallDir"
Write-Host "Log file       : $LogFile"
Write-Host "Components     : models=$WantModels cardDB=$WantCardDb embeddings=$WantEmbeddings"

if ($ApiKey) {
    $env:POKEMON_TCG_API_KEY = $ApiKey
    # Persist (user scope) so manual card-DB refreshes work later.
    [Environment]::SetEnvironmentVariable('POKEMON_TCG_API_KEY', $ApiKey, 'User')
}

# --------------------------------------------------------------------------- #
# If a previous install used a different stack/recipe, rebuild the environment.
# --------------------------------------------------------------------------- #
$StackJsonPath = Join-Path $StateDir 'stack.json'
if (Test-Path $StackJsonPath) {
    $prev = Get-Content $StackJsonPath -Raw | ConvertFrom-Json
    $prevRecipe = [string]$prev.recipe
    if (-not $prevRecipe) {
        # stack.json predates recipe ids: the blackwell recipe never changed,
        # but old cu118 envs (mmrotate 0.x) must be rebuilt.
        $prevRecipe = if ($prev.stack -eq 'blackwell') { 'mm2-torch2.12-cu132' } else { 'legacy' }
    }
    if ($prevRecipe -ne $RecipeId) {
        Write-Host "Environment recipe changed ($prevRecipe -> $RecipeId): rebuilding the Python environment." -ForegroundColor Yellow
        if (Test-Path $VenvDir) { Remove-Item -Recurse -Force $VenvDir }
        Get-ChildItem $StateDir -Filter '*.done' |
            Where-Object { $_.BaseName -match '^0[1-5]-' } |
            Remove-Item -Force
        Remove-Item -Force $StackJsonPath
    }
}

$MainFailed = $false
try {

# --------------------------------------------------------------------------- #
# 1. Private Python + virtual environment
# --------------------------------------------------------------------------- #
Invoke-Step '01-python-venv' "Private Python $PyVersion + virtual environment" {
    if (-not (Test-Path $UvExe)) { throw "Missing $UvExe (broken installation - reinstall TCG-AR)." }
    Invoke-External $UvExe @('python', 'install', $PyVersion)
    if (Test-Path $VenvDir) { Remove-Item -Recurse -Force $VenvDir }
    Invoke-External $UvExe @('venv', '--python', $PyVersion, '--seed', $VenvDir)
    if (-not (Test-Path $VenvPy)) { throw "venv creation failed: $VenvPy not found." }
}

# --------------------------------------------------------------------------- #
# 2. PyTorch (the big one: ~3 GB download)
# --------------------------------------------------------------------------- #
Invoke-Step '02-torch' 'PyTorch (CUDA build, ~3 GB - the longest download)' {
    if ($Stack -eq 'blackwell') {
        Invoke-External $VenvPy @('-m', 'pip', 'install',
            'torch==2.12.1', 'torchvision',
            '--index-url', 'https://download.pytorch.org/whl/cu132')
    } else {
        # 2.3.0 is the newest torch with an official prebuilt mmcv 2.2.0
        # cu118 Windows wheel (see step 3); it still runs on Turing (sm_75).
        Invoke-External $VenvPy @('-m', 'pip', 'install',
            'torch==2.3.0', 'torchvision==0.18.0',
            '--index-url', 'https://download.pytorch.org/whl/cu118')
    }
}

# --------------------------------------------------------------------------- #
# 3. OpenMMLab detector stack
# --------------------------------------------------------------------------- #
Invoke-Step '03-mmlab' 'AI detection stack (mmcv / mmdet / mmrotate)' {
    # Same OpenMMLab 2.x stack on both GPU generations; the app's model code
    # (mmengine/mmrotate 1.x APIs) requires it. No `mim` anywhere: openmim
    # pulls in openxlab, which pins setuptools~=60.2.0 and that setuptools
    # cannot even import on Python >= 3.12.
    if ($Stack -ne 'blackwell') {
        # torch 2.3 and the cu118 mmcv wheel are built against the NumPy 1.x
        # ABI. opencv-python >= 5 hard-requires numpy>=2 and would drag it in,
        # which breaks torch at the first torch<->numpy interop
        # ("_ARRAY_API not found"). Pin both before anything can float them.
        Invoke-External $VenvPy @('-m', 'pip', 'install', 'numpy==1.26.4', 'opencv-python==4.8.1.78')
    }
    Invoke-External $VenvPy @('-m', 'pip', 'install', 'mmengine==0.10.7')
    if ($Stack -eq 'blackwell') {
        # mmcv has no official cp314/cu132 wheel; prefer the one bundled with
        # the installer, then an explicit URL, and only then a source build
        # (which requires MSVC - unlikely to exist on an end-user machine).
        $mmcvWheel = Get-ChildItem (Join-Path $InstallDir 'tools\mmcv-*.whl') -ErrorAction SilentlyContinue |
            Select-Object -First 1
        if ($mmcvWheel) {
            Invoke-External $VenvPy @('-m', 'pip', 'install', $mmcvWheel.FullName)
        } elseif ($env:TCGAR_MMCV_WHEEL_URL) {
            Invoke-External $VenvPy @('-m', 'pip', 'install', $env:TCGAR_MMCV_WHEEL_URL)
        } else {
            Write-Host 'No bundled mmcv wheel found - attempting a source build (this needs Visual Studio Build Tools and takes a long time).' -ForegroundColor Yellow
            Invoke-External $VenvPy @('-m', 'pip', 'install', 'mmcv==2.2.0')
        }
    } else {
        # Official prebuilt cp311/cu118 wheel from the OpenMMLab index.
        Invoke-External $VenvPy @('-m', 'pip', 'install', 'mmcv==2.2.0',
            '-f', 'https://download.openmmlab.com/mmcv/dist/cu118/torch2.3/index.html')
    }
    Invoke-External $VenvPy @('-m', 'pip', 'install', 'mmdet==3.3.0', 'mmrotate==1.0.0rc1')
    # Leave a modern-Python-safe setuptools in place (torch pins <82; some
    # libraries still import pkg_resources at runtime).
    Invoke-External $VenvPy @('-m', 'pip', 'install', 'setuptools==81.0.0')
    # Compatibility patches (Python 3.14 + version caps + registry wiring).
    # Idempotent; exit code 1 means "WARN: installed versions differ from
    # what the patch expects" - report it but keep going.
    $rc = Invoke-External $VenvPy @('-m', 'scripts.patch_mmlibs') -WorkDir $AppDir -OkExitCodes @(0, 1)
    if ($rc -eq 1) {
        Write-Host 'patch_mmlibs reported warnings (version drift) - continuing; check the log if the app fails to start.' -ForegroundColor Yellow
    }
    # The patches gate mmdet/mmrotate imports (version assertions); prove
    # they took effect before this step is marked done.
    Invoke-External $VenvPy @('-c', 'import mmdet, mmrotate')
}

# --------------------------------------------------------------------------- #
# 4. Remaining pure-Python dependencies
# --------------------------------------------------------------------------- #
Invoke-Step '04-requirements' 'Application dependencies (GUI, streaming, imaging)' {
    if ($Stack -eq 'blackwell') {
        Invoke-External $VenvPy @('-m', 'pip', 'install', '-r', (Join-Path $AppDir 'requirements.txt'))
    } else {
        # Constrain the resolver so nothing in requirements.txt can float
        # numpy/opencv above the torch-2.0.1-compatible pins (see step 3).
        $constraints = Join-Path $StateDir 'constraints-cu118.txt'
        @('numpy==1.26.4', 'opencv-python==4.8.1.78') | Out-File -Encoding ascii $constraints
        Invoke-External $VenvPy @('-m', 'pip', 'install',
            '-r', (Join-Path $AppDir 'requirements.txt'), '-c', $constraints)
    }
    @{ stack = $Stack; recipe = $RecipeId; python = $PyVersion; finished = (Get-Date -Format o) } |
        ConvertTo-Json | Out-File -Encoding ascii $StackJsonPath
}

# --------------------------------------------------------------------------- #
# 5. Sanity check (always runs - it is cheap)
# --------------------------------------------------------------------------- #
Invoke-Step '05-sanity' 'Checking the Python environment' -NoMarker {
    Invoke-External $VenvPy @('-c',
        "import torch, mmdet, mmrotate, PySide6, cv2; print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())")
    if ($Stack -ne 'blackwell') {
        # The cu118 torch/mmcv builds need the NumPy 1.x ABI: prove numpy
        # stayed on 1.x and that torch<->numpy interop actually works (would
        # otherwise only explode at the embeddings step, hours later).
        Invoke-External $VenvPy @('-c',
            "import numpy, torch, torchvision; assert numpy.__version__.startswith('1.'), 'numpy ' + numpy.__version__ + ' breaks the cu118 torch build'; torch.from_numpy(numpy.zeros(3)); print('numpy', numpy.__version__, '| torchvision', torchvision.__version__, '| interop OK')")
    }
}

# --------------------------------------------------------------------------- #
# 6. Pre-trained model weights
# --------------------------------------------------------------------------- #
if ($WantModels) {
    Invoke-Step '06-models' 'Pre-trained AI models (~590 MB)' -ValidIf ${function:Test-ModelsData} {
        Invoke-External $VenvPy @('-m', 'scripts.download_assets', '--only', 'models') -WorkDir $AppDir
        # download_assets prints FAILED but still exits 0 on per-file errors,
        # and Google Drive failures can leave a small HTML page (or a wrong
        # file) behind - so verify presence AND a plausible size.
        $bad = @()
        foreach ($w in $script:WeightSizes.Keys) {
            $p = Join-Path $AppDir $w
            if (-not (Test-Path $p)) {
                $bad += "$w (missing)"
            } elseif ((Get-Item $p).Length -lt $script:WeightSizes[$w]) {
                $bad += "$w (only $([int]((Get-Item $p).Length / 1MB)) MB - corrupt or wrong file on the download host)"
                Remove-Item -Force $p   # so the next retry re-downloads it
            }
        }
        if ($bad) {
            throw ("Model download incomplete:`n  " + ($bad -join "`n  ") +
                   "`nGoogle Drive may be throttling or serving a wrong file; retry later.")
        }
    }
}

# --------------------------------------------------------------------------- #
# 7-9. Card database (needs the Pokemon TCG API key; long)
# --------------------------------------------------------------------------- #
if ($WantCardDb) {
    if (-not $env:POKEMON_TCG_API_KEY) {
        throw 'The card-database component is selected but no API key was recorded. Re-run the installer and enter your key (free at https://dev.pokemontcg.io).'
    }
    Invoke-Step '07-metadata' 'Card metadata (sets, attacks, HP, ...)' -ValidIf ${function:Test-MetadataData} {
        Invoke-External $VenvPy @('-m', 'installation.install', '--metadata') -WorkDir $AppDir
    }
    Invoke-Step '08-sprites' '2D sprites (animated + static)' -ValidIf ${function:Test-SpritesData} {
        Invoke-External $VenvPy @('-m', 'installation.install', '--sprites') -WorkDir $AppDir
    }
    Invoke-Step '09-cards' 'Card images (~20,000 cards, several GB - the longest step)' -ValidIf ${function:Test-CardsData} {
        # --update fetches only missing images, so an interrupted run resumes.
        Invoke-External $VenvPy @('-m', 'installation.install', '--cards', '--update') -WorkDir $AppDir
    }
    # New-sets refresh without any command line: reuses the saved API key.
    New-StartMenuShortcut -Name 'TCG-AR - Update card database' `
        -Target "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe" `
        -Arguments "-NoProfile -ExecutionPolicy Bypass -File `"$(Join-Path $InstallDir 'installer\update_carddb.ps1')`"" `
        -WorkDir $InstallDir -Icon (Join-Path $InstallDir 'installer\tcg-ar.ico')
}

# --------------------------------------------------------------------------- #
# 10. Embedding cache (makes the first app start-up instant)
# --------------------------------------------------------------------------- #
if ($DoEmbeddings) {
    Invoke-Step '10-embeddings' 'Pre-computing card embeddings' -ValidIf ${function:Test-EmbeddingsData} {
        Invoke-External $VenvPy @('-m', 'installation.install', '--embeddings') -WorkDir $AppDir
    }
} elseif ($WantEmbeddings) {
    Write-Host 'Skipping embeddings: they require both the models and the card database components.' -ForegroundColor Yellow
}

# The app is runnable only now - create its launch shortcut at the very end
# so a half-finished install never leaves a broken "TCG-AR" entry around.
New-StartMenuShortcut -Name 'TCG-AR' `
    -Target (Join-Path $VenvDir 'Scripts\pythonw.exe') `
    -Arguments '-m inference.main' `
    -WorkDir $AppDir -Icon (Join-Path $InstallDir 'installer\tcg-ar.ico')

$script:CurStepName = 'Setup complete'
Write-ProgressState 'done'

Write-Banner 'TCG-AR setup complete!'
Write-Host 'Launch TCG-AR from the Start Menu (the shortcut was just created). Notes for the first run:' -ForegroundColor Green
Write-Host ' - The RTSP server (MediaMTX) is downloaded automatically on first launch.'
Write-Host ' - Windows Firewall will ask once to allow it; click "Allow access".'

} catch {
    $MainFailed = $true
    Write-ProgressState 'failed' $_.Exception.Message
    Write-Host ''
    Write-Host 'SETUP FAILED' -ForegroundColor Red
    Write-Host ("  {0}" -f $_.Exception.Message) -ForegroundColor Red
    Write-Host "  Full log: $LogFile" -ForegroundColor Red
    Write-Host '  Fix the issue (usually network) and retry - setup resumes where this run stopped.' -ForegroundColor Yellow
} finally {
    Stop-Transcript | Out-Null
    if ($TailerProc) {
        # Let the sidecar do its final flush (it self-exits on done/failed),
        # then make sure it is gone.
        Start-Sleep -Milliseconds 1200
        if (-not $TailerProc.HasExited) {
            Stop-Process -Id $TailerProc.Id -Force -Confirm:$false -ErrorAction SilentlyContinue
        }
    }
}

if (-not $NoPause -and -not $Gui) {
    Write-Host ''
    Read-Host 'Press Enter to close this window'
}
if ($MainFailed) { exit 1 }
exit 0

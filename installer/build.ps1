# Build TCG-AR-Setup-<version>.exe.
#
# Usage (from anywhere):
#   powershell -ExecutionPolicy Bypass -File installer\build.ps1
#   ... -FromWorkingTree     # stage the working tree instead of git HEAD
#                            # (for testing uncommitted changes)
#
# Prerequisites: git, Inno Setup 6 (winget install JRSoftware.InnoSetup),
# internet access the first time (downloads the pinned uv.exe).
#
# Output: installer\dist\TCG-AR-Setup-<version>.exe
# Optional signing: set TCGAR_SIGN_CERT (path to .pfx) and TCGAR_SIGN_PASS.

param(
    [switch]$FromWorkingTree
)

$ErrorActionPreference = 'Stop'

# Pinned uv release bundled inside the installer. When bumping the version,
# set the new SHA-256 (build fails loudly on mismatch).
$UvVersion = '0.11.28'
$UvSha256  = '533FE4044BC50B05AC89F4D07925597FDB5285369724E8986ECAB356818F09EE'   # uv.exe from uv-x86_64-pc-windows-msvc.zip

$InstallerDir = $PSScriptRoot
$RepoRoot     = Split-Path -Parent $InstallerDir
$StageDir     = Join-Path $InstallerDir 'stage'
$StageApp     = Join-Path $StageDir 'app'
$VendorDir    = Join-Path $InstallerDir 'vendor'
$DistDir      = Join-Path $InstallerDir 'dist'
$IssFile      = Join-Path $InstallerDir 'TCG-AR.iss'

# --------------------------------------------------------------------------- #
# 1. Version from pyproject.toml (single source of truth)
# --------------------------------------------------------------------------- #
$pyproject = Get-Content (Join-Path $RepoRoot 'pyproject.toml') -Raw
if ($pyproject -notmatch '(?m)^version\s*=\s*"([^"]+)"') {
    throw 'Could not parse version from pyproject.toml'
}
$Version = $Matches[1]
Write-Host "Building TCG-AR installer v$Version" -ForegroundColor Cyan

# --------------------------------------------------------------------------- #
# 2. Stage the application snapshot
# --------------------------------------------------------------------------- #
if (Test-Path $StageApp) { Remove-Item -Recurse -Force $StageApp }
New-Item -ItemType Directory -Force -Path $StageApp | Out-Null

if ($FromWorkingTree) {
    Write-Host 'Staging from WORKING TREE (uncommitted changes included)' -ForegroundColor Yellow
    $exclDirs = @('.git', 'work_dirs', 'mediamtx',
                  (Join-Path 'installer' 'stage'), (Join-Path 'installer' 'dist'),
                  (Join-Path 'installer' 'vendor'))
    $xd = $exclDirs | ForEach-Object { Join-Path $RepoRoot $_ }
    $xd += '__pycache__'   # bare name = matched at any depth
    # Exclude the multi-GB downloadable asset bulk but keep tracked seeds
    # (annotations, .gitkeep, card back). Simplest robust rule: mirror the
    # repo except the big generated trees.
    $xd += @((Join-Path $RepoRoot 'assets\AI database\background'),
             (Join-Path $RepoRoot 'assets\AI database\detection'),
             (Join-Path $RepoRoot 'assets\AI database\orientation'),
             (Join-Path $RepoRoot 'assets\AI database\identification'),
             (Join-Path $RepoRoot 'assets\AI database\real\images'),
             (Join-Path $RepoRoot 'assets\AI database\real\orientation'),
             (Join-Path $RepoRoot 'assets\database\card_database'),
             (Join-Path $RepoRoot 'assets\database\2D_database'),
             (Join-Path $RepoRoot 'assets\database\2D_animated_database'),
             (Join-Path $RepoRoot 'assets\embedding_cache'))
    robocopy $RepoRoot $StageApp /MIR /XD @xd /XF *.pth *.zip *.log *.pyc settings.yaml auto.crt auto.key /NFL /NDL /NJH | Out-Null
    if ($LASTEXITCODE -ge 8) { throw "robocopy failed (exit $LASTEXITCODE)" }
    # Drop files the working tree accumulated inside otherwise-tracked dirs,
    # and restore the tracked seed files living inside excluded dirs.
    Remove-Item -Force -ErrorAction SilentlyContinue (Join-Path $StageApp 'tests\*.png')
    Remove-Item -Force -ErrorAction SilentlyContinue (Join-Path $StageApp 'assets\database\*.json') -Exclude 'wrong_scan_cards.json'
    New-Item -ItemType Directory -Force -Path (Join-Path $StageApp 'assets\database\card_database') | Out-Null
    Copy-Item (Join-Path $RepoRoot 'assets\database\card_database\back1-1.jpg') `
              (Join-Path $StageApp 'assets\database\card_database\back1-1.jpg')
    # Recreate the excluded-but-expected empty folders (git archive keeps them
    # via .gitkeep; mirror that here).
    foreach ($d in @('assets\AI database\background', 'assets\AI database\detection',
                     'assets\AI database\orientation', 'assets\AI database\identification',
                     'assets\database\2D_database', 'assets\database\2D_animated_database')) {
        New-Item -ItemType Directory -Force -Path (Join-Path $StageApp $d) | Out-Null
    }
} else {
    Write-Host 'Staging from git HEAD (committed files only)'
    $zip = Join-Path $StageDir 'app.zip'
    git -C $RepoRoot archive --format=zip -o $zip HEAD
    if ($LASTEXITCODE -ne 0) { throw 'git archive failed' }
    Expand-Archive -Path $zip -DestinationPath $StageApp -Force
    Remove-Item $zip
}

$stageSize = (Get-ChildItem $StageApp -Recurse -File | Measure-Object Length -Sum).Sum / 1MB
Write-Host ("Staged app snapshot: {0:N1} MB" -f $stageSize)
if ($stageSize -gt 200) {
    throw "Staged snapshot is suspiciously large ($([int]$stageSize) MB) - a generated asset tree leaked into the stage."
}

# --------------------------------------------------------------------------- #
# 3. Vendor the pinned uv.exe
# --------------------------------------------------------------------------- #
$UvExe = Join-Path $VendorDir 'uv.exe'
$needUv = $true
if (Test-Path $UvExe) {
    $have = (Get-FileHash $UvExe -Algorithm SHA256).Hash
    if ($UvSha256 -ne 'PIN-ME' -and $have -eq $UvSha256) { $needUv = $false }
}
if ($needUv) {
    New-Item -ItemType Directory -Force -Path $VendorDir | Out-Null
    $url = "https://github.com/astral-sh/uv/releases/download/$UvVersion/uv-x86_64-pc-windows-msvc.zip"
    $tmp = Join-Path $VendorDir 'uv.zip'
    Write-Host "Downloading uv $UvVersion ..."
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    Invoke-WebRequest -Uri $url -OutFile $tmp -UseBasicParsing
    Expand-Archive -Path $tmp -DestinationPath $VendorDir -Force
    Remove-Item $tmp
    if (-not (Test-Path $UvExe)) { throw 'uv.zip did not contain uv.exe' }
    $have = (Get-FileHash $UvExe -Algorithm SHA256).Hash
    if ($UvSha256 -eq 'PIN-ME') {
        Write-Host "uv.exe SHA-256 = $have" -ForegroundColor Yellow
        Write-Host 'Pin this hash in build.ps1 ($UvSha256) and re-run.' -ForegroundColor Yellow
        throw 'uv hash not pinned yet (supply-chain guard).'
    }
    if ($have -ne $UvSha256) {
        Remove-Item $UvExe
        throw "uv.exe SHA-256 mismatch (expected $UvSha256, got $have)."
    }
}
Write-Host "uv $UvVersion vendored OK"

# --------------------------------------------------------------------------- #
# 3b. Vendor the pre-built mmcv wheel (Blackwell stack)
# --------------------------------------------------------------------------- #
# mmcv 2.x has no official prebuilt wheel for Python 3.14 / torch 2.12 / cu132;
# end users have no compiler, so we ship one built on the maintainer machine
# (mmcv is Apache-2.0; redistribution is fine). Build it once with:
#   pip wheel mmcv==2.2.0 --no-deps -w installer\vendor\
# inside the tcgar-py314 environment (or copy it from the pip wheel cache).
$MmcvWheel = Get-ChildItem (Join-Path $VendorDir 'mmcv-*-cp314-*win_amd64.whl') -ErrorAction SilentlyContinue |
    Select-Object -First 1
if ($MmcvWheel) {
    Write-Host "mmcv wheel vendored: $($MmcvWheel.Name)"
} else {
    Write-Host 'WARNING: no mmcv cp314 wheel in vendor\ - Blackwell installs will fall back to a source build (needs MSVC on the user machine!).' -ForegroundColor Yellow
}

# --------------------------------------------------------------------------- #
# 4. Compile with Inno Setup
# --------------------------------------------------------------------------- #
$iscc = @(
    (Join-Path ${env:ProgramFiles(x86)} 'Inno Setup 6\ISCC.exe'),
    (Join-Path $env:ProgramFiles 'Inno Setup 6\ISCC.exe'),
    (Join-Path $env:LOCALAPPDATA 'Programs\Inno Setup 6\ISCC.exe')
) | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $iscc) {
    $cmd = Get-Command ISCC.exe -ErrorAction SilentlyContinue
    if ($cmd) { $iscc = $cmd.Source }
}
if (-not $iscc) { throw 'Inno Setup 6 not found. Install it: winget install JRSoftware.InnoSetup' }

New-Item -ItemType Directory -Force -Path $DistDir | Out-Null
$isccArgs = @("/DAppVersion=$Version", "/DStageDir=$StageApp", "/DVendorDir=$VendorDir", "/O$DistDir")
if ($MmcvWheel) { $isccArgs += "/DMmcvWheel=$($MmcvWheel.FullName)" }
& $iscc @isccArgs $IssFile
if ($LASTEXITCODE -ne 0) { throw "ISCC failed (exit $LASTEXITCODE)" }

$SetupExe = Join-Path $DistDir "TCG-AR-Setup-$Version.exe"
if (-not (Test-Path $SetupExe)) { throw "Expected output not found: $SetupExe" }

# --------------------------------------------------------------------------- #
# 5. Optional code signing
# --------------------------------------------------------------------------- #
if ($env:TCGAR_SIGN_CERT) {
    Write-Host 'Signing installer...'
    & signtool sign /f $env:TCGAR_SIGN_CERT /p $env:TCGAR_SIGN_PASS `
        /tr http://timestamp.digicert.com /td sha256 /fd sha256 $SetupExe
    if ($LASTEXITCODE -ne 0) { throw 'signtool failed' }
}

$exeSize = (Get-Item $SetupExe).Length / 1MB
Write-Host ("Done: {0} ({1:N1} MB)" -f $SetupExe, $exeSize) -ForegroundColor Green

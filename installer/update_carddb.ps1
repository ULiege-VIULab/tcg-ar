# TCG-AR card-database update.
#
# Refreshes the card database after new Pokemon TCG sets are released.
# No retraining is needed: identification is open-set metric learning, so
# fetching the new reference images is enough. Safe to re-run any time;
# only missing data is downloaded.
#
# Driven either by the installer wizard ("Update the card database" mode,
# -Gui: hidden, progress via state\progress.json) or by the Start Menu
# shortcut "TCG-AR - Update card database" (visible console).
#
# Written for Windows PowerShell 5.1 (the stock powershell.exe).

param(
    # Hidden/GUI mode: no console interaction; report through progress.json.
    [switch]$Gui,
    [switch]$NoPause,
    [string]$InstallDir = (Split-Path -Parent $PSScriptRoot)
)

$ErrorActionPreference = 'Stop'

$AppDir   = Join-Path $InstallDir 'app'
$VenvPy   = Join-Path $InstallDir 'env\Scripts\python.exe'
$StateDir = Join-Path $InstallDir 'state'
$LogsDir  = Join-Path $InstallDir 'logs'
foreach ($d in @($StateDir, $LogsDir)) {
    if (-not (Test-Path $d)) { New-Item -ItemType Directory -Path $d | Out-Null }
}
$LogFile = Join-Path $LogsDir ("update-carddb-{0}.log" -f (Get-Date -Format 'yyyyMMdd-HHmmss'))
Start-Transcript -Path $LogFile | Out-Null

# Python subprocesses print unicode; never let a legacy code page crash them.
$env:PYTHONUTF8 = '1'

# --------------------------------------------------------------------------- #
# Progress protocol (same shape as bootstrap.ps1; wizard-agnostic)
# --------------------------------------------------------------------------- #
$ProgressPath = Join-Path $StateDir 'progress.json'
$script:StepNum     = 0
$script:TotalSteps  = 4
$script:CurStepName = 'Starting'
$script:CurStepLog  = ''
$Utf8NoBom = New-Object System.Text.UTF8Encoding($false)

function Write-ProgressState([string]$Status, [string]$Message = '') {
    $obj = @{
        mode    = 'update'
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

# GUI mode: report immediately and start the log-tail sidecar (see tailer.ps1).
$TailerProc = $null
if ($Gui) {
    foreach ($f in @('live.txt', 'tail.txt')) {
        $p = Join-Path $StateDir $f
        if (Test-Path $p) { Remove-Item -Force $p }
    }
    $script:CurStepName = 'Preparing the update...'
    Write-ProgressState 'running'
    $TailerProc = Start-Process -PassThru -WindowStyle Hidden `
        -FilePath "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe" `
        -ArgumentList ('-NoProfile -ExecutionPolicy Bypass -File "{0}" -InstallDir "{1}" -ParentPid {2}' -f `
            (Join-Path $PSScriptRoot 'tailer.ps1'), $InstallDir, $PID)
}

function Write-Banner([string]$Text) {
    Write-Host ''
    Write-Host ('=' * 70) -ForegroundColor Cyan
    Write-Host "  $Text" -ForegroundColor Cyan
    Write-Host ('=' * 70) -ForegroundColor Cyan
}

function Invoke-External {
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

function Invoke-UpdateStep {
    param([string]$Name, [string]$Title, [scriptblock]$Body)
    $script:StepNum++
    $script:CurStepName = $Title
    $script:CurStepLog = Join-Path $LogsDir ("update-{0:D2}-{1}.log" -f $script:StepNum, $Name)
    if (Test-Path $script:CurStepLog) { Remove-Item -Force $script:CurStepLog }
    Write-ProgressState 'running'
    Write-Banner "$($script:StepNum)/$($script:TotalSteps)  $Title"
    & $Body
    Write-Host "[ok] $Title" -ForegroundColor Green
}

$Failed = $false
try {
    if (-not (Test-Path $VenvPy)) {
        throw 'TCG-AR is not fully set up yet - run the installer (or "TCG-AR Setup (repair)" from the Start Menu) first.'
    }
    if (-not $env:POKEMON_TCG_API_KEY) {
        throw ('No Pokemon TCG API key is configured. Re-run the TCG-AR installer and enter your key ' +
               '(free at https://dev.pokemontcg.io) on the API key page.')
    }

    Write-Banner 'Updating the TCG-AR card database'
    Write-Host 'Only missing cards/sprites are downloaded; this is quick unless many new sets were released.'

    Invoke-UpdateStep 'metadata' 'Card metadata (sets, attacks, HP, ...)' {
        Invoke-External $VenvPy @('-m', 'installation.install', '--metadata') -WorkDir $AppDir
    }
    Invoke-UpdateStep 'cards' 'New card images' {
        Invoke-External $VenvPy @('-m', 'installation.install', '--cards', '--update') -WorkDir $AppDir
    }
    Invoke-UpdateStep 'sprites' 'New sprites' {
        Invoke-External $VenvPy @('-m', 'installation.install', '--sprites') -WorkDir $AppDir
    }
    Invoke-UpdateStep 'embeddings' 'Refreshing the embedding cache' {
        Invoke-External $VenvPy @('-m', 'installation.install', '--embeddings') -WorkDir $AppDir
    }

    $script:CurStepName = 'Card database is up to date'
    Write-ProgressState 'done'
    Write-Banner 'Card database is up to date!'
    Write-Host 'The new cards are picked up automatically the next time TCG-AR starts.' -ForegroundColor Green
} catch {
    $Failed = $true
    Write-ProgressState 'failed' $_.Exception.Message
    Write-Host ''
    Write-Host 'UPDATE FAILED' -ForegroundColor Red
    Write-Host ("  {0}" -f $_.Exception.Message) -ForegroundColor Red
    Write-Host "  Full log: $LogFile" -ForegroundColor Red
    Write-Host '  This is usually a network hiccup - simply run the update again; it resumes where it stopped.' -ForegroundColor Yellow
} finally {
    Stop-Transcript | Out-Null
    if ($TailerProc) {
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
if ($Failed) { exit 1 }
exit 0

# TCG-AR setup sidecar: distills the active worker step log into two tiny
# files that the installer wizard's progress page polls:
#   state\live.txt  - line 1: last percentage seen (or -1), line 2: the
#                     current progress line (e.g. a live tqdm bar)
#   state\tail.txt  - the last ~14 log lines for the wizard's log view
#
# Why a sidecar: the step logs grow to many MB (pip/tqdm rewrite progress via
# \r), and Inno's Pascal can only read whole files. This process seeks to the
# tail with a shared-read FileStream, strips ANSI escapes, and rewrites the
# small files every ~400 ms. Started by bootstrap.ps1 / update_carddb.ps1 in
# -Gui mode; exits by itself when the run ends or the worker dies.

param(
    [Parameter(Mandatory = $true)][string]$InstallDir,
    [Parameter(Mandatory = $true)][int]$ParentPid
)

$ErrorActionPreference = 'Continue'

$StateDir     = Join-Path $InstallDir 'state'
$ProgressPath = Join-Path $StateDir 'progress.json'
$LivePath     = Join-Path $StateDir 'live.txt'
$TailPath     = Join-Path $StateDir 'tail.txt'
$Utf8Bom      = New-Object System.Text.UTF8Encoding($true)   # BOM: Inno's LoadStringsFromFile needs it to decode UTF-8
$AnsiRe       = [regex]"`e\[[0-9;?]*[A-Za-z]"
$PctRe        = [regex]'(\d{1,3})\s?%'

function Write-Atomic([string]$Path, [string]$Content) {
    $tmp = "$Path.tmp"
    [IO.File]::WriteAllText($tmp, $Content, $Utf8Bom)
    Move-Item -Force $tmp $Path
}

function Read-LogTail([string]$Path, [int]$Bytes) {
    try {
        $fs = [IO.File]::Open($Path, 'Open', 'Read', 'ReadWrite')
        try {
            $start = [Math]::Max(0, $fs.Length - $Bytes)
            $fs.Seek($start, 'Begin') | Out-Null
            $buf = New-Object byte[] ($fs.Length - $start)
            $n = $fs.Read($buf, 0, $buf.Length)
            return [Text.Encoding]::UTF8.GetString($buf, 0, $n)
        } finally { $fs.Close() }
    } catch { return $null }
}

$lastWrite = ''
while ($true) {
    Start-Sleep -Milliseconds 400

    # Stop when the worker is gone or the run has ended (one final pass first).
    $final = $false
    if (-not (Get-Process -Id $ParentPid -ErrorAction SilentlyContinue)) { $final = $true }
    $stepLog = ''
    try {
        $j = Get-Content $ProgressPath -Raw -ErrorAction Stop | ConvertFrom-Json
        $stepLog = [string]$j.steplog
        if ($j.status -in 'done', 'failed') { $final = $true }
    } catch { }

    if ($stepLog -and (Test-Path $stepLog)) {
        $raw = Read-LogTail $stepLog 8192
        if ($null -ne $raw) {
            $txt = $AnsiRe.Replace($raw, '')
            $txt = $txt -replace "`r`n", "`n" -replace "`r", "`n"
            $lines = @($txt -split "`n" | Where-Object { $_.Trim() -ne '' })
            if ($lines.Count -gt 0) {
                $current = $lines[-1].TrimEnd()
                $pct = -1
                $m = $PctRe.Matches($current)
                if ($m.Count -gt 0) {
                    $v = [int]$m[$m.Count - 1].Groups[1].Value
                    if ($v -ge 0 -and $v -le 100) { $pct = $v }
                }
                $tailLines = $lines | Select-Object -Last 14
                $stamp = "$pct|$current|$($tailLines -join '~')"
                if ($stamp -ne $lastWrite) {
                    $lastWrite = $stamp
                    Write-Atomic $LivePath ("{0}`r`n{1}`r`n" -f $pct, $current)
                    Write-Atomic $TailPath (($tailLines -join "`r`n") + "`r`n")
                }
            }
        }
    }

    if ($final) { break }
}
exit 0

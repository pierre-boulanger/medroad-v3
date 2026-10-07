<#
.SYNOPSIS
    End-to-end retraining for MedROAD V3 on Windows.

.DESCRIPTION
    PowerShell wrapper around scripts/retrain.py. Resolves the interpreter,
    checks the extracts exist before starting, and quotes paths so that
    directories containing spaces work, which matters because the usual
    OneDrive working directory has several.

.EXAMPLE
    .\scripts\retrain.ps1 -MimicDir .\mimic_extract -Quick
    .\scripts\retrain.ps1 -MimicDir .\mimic_extract
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$MimicDir,

    [string]$Windows = "mimic_windows.csv",
    [string]$Models  = "models_saved",
    [string]$Results = "results",
    [double]$HorizonHours = 12.0,

    [switch]$Quick,             # 50 stays, to verify the pipeline
    [switch]$Lenient,           # warn instead of stopping on plausibility gates
    [switch]$SkipEtl,
    [switch]$SkipExperiments
)

$ErrorActionPreference = "Stop"

function Write-Stage($text) {
    Write-Host ""
    Write-Host ("=" * 66) -ForegroundColor Cyan
    Write-Host "  $text" -ForegroundColor Cyan
    Write-Host ("=" * 66) -ForegroundColor Cyan
}

# ── Interpreter ──────────────────────────────────────────────────────────
# Windows installs expose "python"; "python3" usually resolves to the Store
# stub, which silently does nothing useful.
$py = $null
foreach ($cand in @("python", "py")) {
    $cmd = Get-Command $cand -ErrorAction SilentlyContinue
    if ($cmd) { $py = $cand; break }
}
if (-not $py) {
    throw "No Python interpreter found on PATH. Activate your environment first, e.g. conda activate ndib"
}

$ver = & $py -c "import sys; print('.'.join(map(str, sys.version_info[:2])))"
Write-Host "Interpreter : $py (Python $ver)"
if ([version]$ver -lt [version]"3.11") {
    throw "Python 3.11 or newer is required; found $ver"
}

# ── Repository root ──────────────────────────────────────────────────────
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
Write-Host "Repository  : $repo"

# ── Package import check ─────────────────────────────────────────────────
& $py -c "import medroad_v3" 2>$null
if ($LASTEXITCODE -ne 0) {
    throw "medroad_v3 is not importable. Run: pip install -e `".[all]`""
}

# ── Extracts ─────────────────────────────────────────────────────────────
Write-Stage "Checking extracts"
$MimicDir = (Resolve-Path $MimicDir).Path
$required = @("chartevents.csv", "labevents.csv", "stays.csv", "outcomes.csv")
$missing  = @()
foreach ($f in $required) {
    $p = Join-Path $MimicDir $f
    if (Test-Path $p) {
        $mb = [math]::Round((Get-Item $p).Length / 1MB, 1)
        Write-Host ("  {0,-20} {1,10} MB" -f $f, $mb)
    } else {
        $missing += $f
    }
}
if ($missing.Count -gt 0) {
    throw "Missing extracts: $($missing -join ', '). Run sql\extract_mimic.sql first."
}

# ── Build the argument list ──────────────────────────────────────────────
$pyArgs = @(
    "scripts/retrain.py",
    "--mimic-dir",     $MimicDir,
    "--windows",       $Windows,
    "--models",        $Models,
    "--results",       $Results,
    "--horizon-hours", $HorizonHours
)
if ($Quick)           { $pyArgs += "--quick" }
if ($Lenient)         { $pyArgs += "--lenient" }
if ($SkipEtl)         { $pyArgs += "--skip-etl" }
if ($SkipExperiments) { $pyArgs += "--skip-experiments" }

Write-Stage "Running retraining"
Write-Host "  $py $($pyArgs -join ' ')"
Write-Host ""

$sw = [System.Diagnostics.Stopwatch]::StartNew()
& $py @pyArgs
$code = $LASTEXITCODE
$sw.Stop()

Write-Host ""
if ($code -eq 0) {
    Write-Host ("Completed in {0:hh\:mm\:ss}" -f $sw.Elapsed) -ForegroundColor Green
    Write-Host "Models  : $(Join-Path $repo $Models)"
    Write-Host "Results : $(Join-Path $repo $Results)"
} else {
    Write-Host ("Stopped after {0:hh\:mm\:ss}" -f $sw.Elapsed) -ForegroundColor Red
    Write-Host "See $(Join-Path $repo (Join-Path $Results 'retrain_report.json'))"
}
exit $code

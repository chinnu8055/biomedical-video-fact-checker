param(
    [ValidateSet("3.11", "3.12")]
    [string]$PreferredVersion = "3.11"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot

$pythonCommand = $null
$pythonPrefixArguments = @()
$launcher = Get-Command py -ErrorAction SilentlyContinue
if ($null -ne $launcher) {
    foreach ($version in @($PreferredVersion, "3.11", "3.12") | Select-Object -Unique) {
        & $launcher.Source "-$version" -c "import sys; assert sys.version_info[:2] == tuple(map(int, '$version'.split('.')))" 2>$null
        if ($LASTEXITCODE -eq 0) {
            $pythonCommand = $launcher.Source
            $pythonPrefixArguments = @("-$version")
            break
        }
    }
}

if ($null -eq $pythonCommand) {
    $candidatePaths = @(
        "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe",
        "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe",
        "C:\Python311\python.exe",
        "C:\Python312\python.exe"
    )
    foreach ($candidate in $candidatePaths) {
        if (Test-Path -LiteralPath $candidate) {
            $version = & $candidate -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
            if ($version -in @("3.11", "3.12")) {
                $pythonCommand = $candidate
                break
            }
        }
    }
}

if ($null -eq $pythonCommand) {
    throw @"
Python 3.11 or 3.12 was not found. Python 3.13 is not supported by the ML stack.

Install Python 3.11, then close and reopen PowerShell:
  winget install -e --id Python.Python.3.11

After reopening PowerShell, run:
  powershell -ExecutionPolicy Bypass -File .\scripts\setup_windows.ps1
"@
}

$venvPath = Join-Path $projectRoot ".venv"
$venvPython = Join-Path $venvPath "Scripts\python.exe"
$reuseEnvironment = $false

if (Test-Path -LiteralPath $venvPath) {
    if (Test-Path -LiteralPath $venvPython) {
        try {
            $existingVersion = & $venvPython -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
            $reuseEnvironment = $LASTEXITCODE -eq 0 -and $existingVersion -in @("3.11", "3.12")
        }
        catch {
            $reuseEnvironment = $false
        }
    }

    if (-not $reuseEnvironment) {
        $projectPrefix = [System.IO.Path]::GetFullPath($projectRoot) + [System.IO.Path]::DirectorySeparatorChar
        $resolvedVenv = [System.IO.Path]::GetFullPath($venvPath)
        if (-not $resolvedVenv.StartsWith($projectPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
            throw "Refusing to move a virtual environment outside the project folder: $resolvedVenv"
        }

        $timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
        $backupPath = Join-Path $projectRoot ".venv-backup-$timestamp"
        if (Test-Path -LiteralPath $backupPath) {
            $backupPath = Join-Path $projectRoot ".venv-backup-$timestamp-$([guid]::NewGuid().ToString('N').Substring(0, 6))"
        }
        Write-Warning "The existing .venv is unusable. Preserving it as $(Split-Path -Leaf $backupPath)."
        try {
            Move-Item -LiteralPath $venvPath -Destination $backupPath
        }
        catch {
            throw "Could not preserve the old .venv. Close any Python or Streamlit process using it, then retry. $($_.Exception.Message)"
        }
    }
}

if ($reuseEnvironment) {
    Write-Host "Reusing the compatible .venv..." -ForegroundColor Cyan
}
else {
    Write-Host "Creating .venv with a compatible Python..." -ForegroundColor Cyan
    & $pythonCommand @pythonPrefixArguments -m venv $venvPath
    if ($LASTEXITCODE -ne 0) { throw "Virtual environment creation failed." }
}

& $venvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed." }

& $venvPython -m pip install -e ".[all,dev]"
if ($LASTEXITCODE -ne 0) { throw "Project installation failed." }

& $venvPython -c "import biomed_video_fc; print('Installed biomedical-video-fact-checker', biomed_video_fc.__version__)"
if ($LASTEXITCODE -ne 0) { throw "Project import verification failed." }

if ($null -eq (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    $bundledFFmpeg = & $venvPython -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())"
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Using bundled FFmpeg: $bundledFFmpeg" -ForegroundColor Green
    }
    else {
        Write-Warning "Neither system FFmpeg nor the bundled imageio-ffmpeg binary is available."
    }
}

Write-Host "" 
Write-Host "Setup complete. Use these commands from the project folder:" -ForegroundColor Green
Write-Host ".\.venv\Scripts\biofc-prepare-datasets.exe --data-dir data --output artifacts\prepared_datasets"
Write-Host ".\.venv\Scripts\biofc-train-multisource.exe --prepared-dir artifacts\prepared_datasets --output artifacts\multisource_classifier --epochs 5 --batch-size 8 --hard-negative-ratio 0.25 --seed 42"
Write-Host ".\.venv\Scripts\biofc-evaluate-challenge.exe --model-dir artifacts\multisource_classifier --challenge data\challenge_sets\vitamin_c_precision.jsonl --output artifacts\evaluation\vitamin_c_precision_report.json"
Write-Host '$env:BIOFC_CLASSIFIER_DIR = "artifacts/multisource_classifier"'
Write-Host ""
Write-Host "Set your contact email for the official PubMed API (no baseline download is needed):"
Write-Host '$env:BIOFC_NCBI_EMAIL = "your-real-email@example.com"'
Write-Host "Optional, if you already have an NCBI API key:"
Write-Host '$env:BIOFC_NCBI_API_KEY = "your-key"'
Write-Host ""
Write-Host "powershell -ExecutionPolicy Bypass -File .\scripts\run_dashboard.ps1"

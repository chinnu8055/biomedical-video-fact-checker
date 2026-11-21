param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$StreamlitArguments
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
$appPath = Join-Path $projectRoot "app.py"

if (-not (Test-Path -LiteralPath $venvPython)) {
    throw "The project environment is missing. Run .\scripts\setup_windows.ps1 first."
}

Set-Location -LiteralPath $projectRoot
& $venvPython -m streamlit run $appPath `
    --server.fileWatcherType none `
    --server.runOnSave false `
    @StreamlitArguments
exit $LASTEXITCODE

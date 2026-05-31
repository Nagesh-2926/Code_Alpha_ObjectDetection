$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

$venvPath = Join-Path $PSScriptRoot ".venv"
if (-not (Test-Path $venvPath)) {
    python -m venv $venvPath
}

$pythonExe = Join-Path $venvPath "Scripts\\python.exe"

& $pythonExe -m pip install --upgrade pip
& $pythonExe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
& $pythonExe -m pip install ".[dev]"

Write-Host "Setup complete. Activate the environment with .\\.venv\\Scripts\\Activate.ps1"

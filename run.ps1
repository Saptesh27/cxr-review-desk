$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
    Write-Host "No virtual environment yet. Follow STEPS.md first."
    exit 1
}
& ".\.venv\Scripts\python.exe" -m uvicorn cxr.server:app --host 127.0.0.1 --port 7860

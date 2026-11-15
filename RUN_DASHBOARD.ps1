# Launches the Streamlit dashboard in an isolated Python environment.
# Run from PowerShell: powershell -ExecutionPolicy Bypass -File .\RUN_DASHBOARD.ps1
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$py = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $py)) {
    Write-Host '[VEXO] Creating isolated .venv ...'
    python -m venv .venv
}
& $py -c 'import streamlit, plotly, reportlab' 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host '[VEXO] Installing dashboard dependencies ...'
    & $py -m pip install -r .\requirements_demo.txt
    if ($LASTEXITCODE -ne 0) { throw 'Package installation failed.' }
}
Write-Host '[VEXO] Starting dashboard. Use a separate terminal for the Vexo Bluetooth collector.'
& $py -m streamlit run .\app\vexo_demo.py

# Vexo dashboard launcher -- robust for Windows PowerShell 5.1
# Save this script in the vexo-pain-monitor project root.
$ErrorActionPreference = 'Continue'
Set-Location $PSScriptRoot

$py = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $py)) {
    Write-Host '[VEXO] Creating isolated .venv ...'
    python -m venv .venv
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $py)) {
        Write-Error 'Could not create the Python virtual environment. Check that python is installed.'
        exit 1
    }
}

Write-Host '[VEXO] Installing/checking dashboard dependencies ...'
& $py -m pip install -r (Join-Path $PSScriptRoot 'requirements_demo.txt')
if ($LASTEXITCODE -ne 0) {
    Write-Error 'Package installation failed. Review the pip output above.'
    exit 1
}

Write-Host '[VEXO] Starting dashboard; use a separate terminal for the Bluetooth collector.'
& $py -m streamlit run (Join-Path $PSScriptRoot 'app\vexo_demo.py')
if ($LASTEXITCODE -ne 0) {
    Write-Error 'Streamlit stopped with an error. Review the output above.'
    exit 1
}

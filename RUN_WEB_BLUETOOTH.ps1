$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
Write-Host 'Vexo Web Bluetooth connector (Chrome / Edge)' -ForegroundColor Cyan
Write-Host 'Open http://localhost:8765/ble_bridge/index.html and click Connect Vexo 79'
python .\ble_bridge\server.py

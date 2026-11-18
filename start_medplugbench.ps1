$ErrorActionPreference = 'Stop'

Write-Host "MedPlugBench" -ForegroundColor Cyan
Write-Host "Project: $((Get-Location).Path)"
Write-Host ''
Write-Host 'Optional: set/validate the SIIM-ACR dataset root first:' -ForegroundColor Cyan
Write-Host '  . .\tools\set_dataset_root.ps1 -DatasetRoot "C:\MedPlugBench\nothing"'
Write-Host ''

$answer = Read-Host "Start MedPlugBench app.py (configuration/results UI only)? [Y/N]"
if ($answer -notmatch '^[Yy]$') {
    Write-Host "Cancelled."
    exit 0
}

python app.py

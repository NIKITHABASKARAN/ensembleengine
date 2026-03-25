# Start the unified FastAPI API (dashboard + /simulate) on http://127.0.0.1:8000
# Requires Python 3.11+ so login_risk_model.pkl unpickles correctly.
#
# Usage:
#   .\run_backend.ps1              # warns if port 8000 busy
#   .\run_backend.ps1 -FreePort    # stops python.exe listening on 8000, then starts
param(
    [switch] $FreePort
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$venvPy = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPy)) {
    Write-Host "No .venv found. Creating with Python 3.11..." -ForegroundColor Yellow
    py -3.11 -m venv .venv
    & .\.venv\Scripts\pip.exe install -r requirements.txt
}
$port = 8000
$inUse = @(Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue)
if ($inUse.Count -gt 0) {
    $pidListen = $inUse[0].OwningProcess
    $proc = Get-Process -Id $pidListen -ErrorAction SilentlyContinue
    if ($FreePort -and $proc -and $proc.ProcessName -eq "python") {
        Write-Host "Stopping python.exe PID $pidListen on port $port..." -ForegroundColor Yellow
        Stop-Process -Id $pidListen -Force
        Start-Sleep -Seconds 2
    } else {
        Write-Host "WARNING: Port $port is in use (PID $pidListen). You may see DEGRADED /health if that is old Python 3.10." -ForegroundColor Yellow
        Write-Host "Run: .\run_backend.ps1 -FreePort   OR   Stop-Process -Id $pidListen -Force" -ForegroundColor Gray
    }
}
Write-Host "Starting app_unified on http://127.0.0.1:$port (Ctrl+C to stop)..." -ForegroundColor Cyan
& $venvPy -m uvicorn app_unified:fastapi_app --host 127.0.0.1 --port $port

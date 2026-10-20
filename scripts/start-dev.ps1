$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'Create .venv and install requirements-dev.txt first. See README.md.' }
$ownedServer = $null
try {
    try { $health = Invoke-RestMethod -Uri 'http://127.0.0.1:8000/api/health' -TimeoutSec 2 } catch { $health = $null }
    if ($health.status -ne 'ok') {
        New-Item -ItemType Directory -Force (Join-Path $projectRoot 'runtime') | Out-Null
        $ownedServer = Start-Process -FilePath $pythonPath -ArgumentList @('run_dashboard.py','--host','0.0.0.0','--port','8000') -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $projectRoot 'runtime/server.out.log') -RedirectStandardError (Join-Path $projectRoot 'runtime/server.err.log')
    }
    Write-Host 'Dashboard: http://127.0.0.1:8000'
    Write-Host 'Keep the phone and computer on the same Wi-Fi. Scan the Expo QR below.'
    Push-Location (Join-Path $projectRoot 'mobile-app')
    try { npm.cmd start } finally { Pop-Location }
} finally {
    if ($ownedServer -and -not $ownedServer.HasExited) { Stop-Process -Id $ownedServer.Id }
}

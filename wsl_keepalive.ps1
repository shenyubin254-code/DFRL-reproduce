# Invoked by Windows Task Scheduler, under the Windows owner of Ubuntu-Shen.
# Does not launch experiments. Stop the task before intentionally shutting WSL down.
$ErrorActionPreference = 'Continue'
$log = Join-Path $PSScriptRoot 'wsl_keepalive.log'
while ($true) {
    Add-Content -LiteralPath $log -Value "$(Get-Date -Format o) Starting Ubuntu-Shen lifetime anchor"
    & "$env:WINDIR\System32\wsl.exe" -d Ubuntu-Shen -u Shen --exec /bin/sleep infinity
    Add-Content -LiteralPath $log -Value "$(Get-Date -Format o) wsl.exe exited: $LASTEXITCODE; retry in 10 seconds"
    Start-Sleep -Seconds 10
}

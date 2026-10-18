# OPTIONAL HOST CONFIGURATION: review and run on SIAT-RTX3080 only.
# This task keeps WSL open; it NEVER starts or resumes a benchmark.
# It has not been installed by the audit.
$ErrorActionPreference = 'Stop'
if ($env:COMPUTERNAME -ne 'SIAT-RTX3080') {
    throw 'Run this on the Windows server SIAT-RTX3080, not on your laptop.'
}
$taskName = 'DFRL-WSL-KeepAlive'
if (Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue) {
    throw "Task $taskName already exists; inspect it before changing anything."
}
$account = "$env:COMPUTERNAME\Shen"
$credential = Get-Credential -UserName $account -Message 'Windows account that owns Ubuntu-Shen (not the Linux password)'
if ($credential.UserName -ne $account) { throw "Expected account $account" }
$ops = Join-Path $env:LOCALAPPDATA 'DFRL-ops'
$runner = Join-Path $ops 'wsl_keepalive.ps1'
if (Test-Path -LiteralPath $runner) { throw "Existing runner: $runner; inspect before replacing." }
New-Item -ItemType Directory -Path $ops -Force | Out-Null
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'wsl_keepalive.ps1') -Destination $runner
$action = New-ScheduledTaskAction -Execute "$env:WINDIR\System32\WindowsPowerShell\v1.0\powershell.exe" `
    -Argument "-NoProfile -NonInteractive -File `"$runner`""
$trigger = New-ScheduledTaskTrigger -AtStartup
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1) `
    -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -MultipleInstances IgnoreNew
Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger `
    -Settings $settings -User $account -Password $credential.GetNetworkCredential().Password `
    -Description 'Keep Ubuntu-Shen running independently of SSH. No benchmark autorun.' | Out-Null
Start-ScheduledTask -TaskName $taskName
Get-ScheduledTask -TaskName $taskName | Select-Object TaskName, State

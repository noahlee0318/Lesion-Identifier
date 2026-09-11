<#
.SYNOPSIS
  Register the Lesion Atlas ingest server to start hidden at logon.

.DESCRIPTION
  "Go start the server first" is exactly the daily friction that kills
  adherence, and adherence is the #1 risk on this project. This makes the
  server a thing that is simply always running.

  Creates a Task Scheduler task that:
    - fires at logon for the current user
    - runs the venv's pythonw.exe (no console window ever appears)
    - pins the working directory to the repo
    - restarts on failure, indefinitely
    - has no execution time limit and is not stopped on battery or idle

  RUN THIS ONCE, AS ADMINISTRATOR:
      powershell -ExecutionPolicy Bypass -File .\scripts\install_autostart.ps1

  Add -WithFirewall to also open the port inbound on PRIVATE networks only.

.PARAMETER Port
  Port to bind. Default 8000.

.PARAMETER WithFirewall
  Also create the inbound firewall rule (Private profile only).
#>
[CmdletBinding()]
param(
  [int]$Port = 8000,
  [switch]$WithFirewall
)

$ErrorActionPreference = "Stop"
$TaskName = "LesionAtlasIngest"

# --- must be admin -------------------------------------------------------
$isAdmin = ([Security.Principal.WindowsPrincipal] `
  [Security.Principal.WindowsIdentity]::GetCurrent()
).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
  Write-Error "Run this in an ADMINISTRATOR PowerShell. Right-click PowerShell -> Run as administrator."
}

$repo   = Split-Path -Parent $PSScriptRoot
$pythonw = Join-Path $repo ".venv\Scripts\pythonw.exe"
$python  = Join-Path $repo ".venv\Scripts\python.exe"

if (-not (Test-Path $pythonw)) {
  if (Test-Path $python) {
    Write-Warning "pythonw.exe missing; falling back to python.exe (a console window WILL appear)."
    $pythonw = $python
  } else {
    Write-Error "No interpreter at $python - create the venv and run scripts\install_deps.ps1 first."
  }
}

Write-Host "repo        : $repo"
Write-Host "interpreter : $pythonw"
Write-Host "port        : $Port"

# --- build the task ------------------------------------------------------
$action = New-ScheduledTaskAction `
  -Execute  $pythonw `
  -Argument "-m src.server.main --port $Port --no-banner" `
  -WorkingDirectory $repo

$trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME

# Interactive token, ordinary privileges. The server binds 8000 and writes
# under the data root; it has no reason to run elevated.
$principal = New-ScheduledTaskPrincipal `
  -UserId ("{0}\{1}" -f $env:USERDOMAIN, $env:USERNAME) `
  -LogonType Interactive `
  -RunLevel Limited

$settings = New-ScheduledTaskSettingsSet `
  -AllowStartIfOnBatteries `
  -DontStopIfGoingOnBatteries `
  -DontStopOnIdleEnd `
  -StartWhenAvailable `
  -RestartCount 999 `
  -RestartInterval (New-TimeSpan -Minutes 1) `
  -ExecutionTimeLimit (New-TimeSpan -Seconds 0) `
  -MultipleInstances IgnoreNew

if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
  Write-Host "existing task found - replacing it"
  Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

Register-ScheduledTask `
  -TaskName  $TaskName `
  -Action    $action `
  -Trigger   $trigger `
  -Principal $principal `
  -Settings  $settings `
  -Description "Lesion Atlas phase 0 ingest server (hidden, restarts on failure)" | Out-Null

Write-Host ""
Write-Host "registered scheduled task '$TaskName'" -ForegroundColor Green

# --- optional firewall rule ---------------------------------------------
if ($WithFirewall) {
  $ruleName = "Lesion Atlas ingest $Port (Private)"
  Get-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue |
    Remove-NetFirewallRule -ErrorAction SilentlyContinue
  New-NetFirewallRule `
    -DisplayName $ruleName `
    -Direction Inbound -Action Allow -Protocol TCP -LocalPort $Port `
    -Profile Private `
    -Description "Phone -> laptop photo upload. PRIVATE networks only." | Out-Null
  Write-Host "firewall rule '$ruleName' created (Private profile only)" -ForegroundColor Green
}

# --- start it now --------------------------------------------------------
Start-ScheduledTask -TaskName $TaskName
Start-Sleep -Seconds 4

try {
  $h = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 8
  Write-Host ""
  Write-Host "server is up. images in db: $($h.images)  data root: $($h.data_root)" -ForegroundColor Green
} catch {
  Write-Warning "task registered but /health did not answer yet."
  Write-Warning "check logs\server.log, then: Get-ScheduledTaskInfo -TaskName $TaskName"
}

Write-Host ""
Write-Host "To verify after a reboot, from your phone browse to:"
Write-Host "    http://<laptop-ip>:$Port/"
Write-Host "To remove:  .\scripts\uninstall_autostart.ps1"

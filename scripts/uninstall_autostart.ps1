<#
.SYNOPSIS
  Remove the Lesion Atlas autostart task (and optionally its firewall rule).

.DESCRIPTION
  RUN AS ADMINISTRATOR:
      powershell -ExecutionPolicy Bypass -File .\scripts\uninstall_autostart.ps1

  Removes only the scheduled task and, with -WithFirewall, the inbound rule.
  Touches no data, no database, no images.
#>
[CmdletBinding()]
param(
  [int]$Port = 8000,
  [switch]$WithFirewall
)

$ErrorActionPreference = "Stop"
$TaskName = "LesionAtlasIngest"

$isAdmin = ([Security.Principal.WindowsPrincipal] `
  [Security.Principal.WindowsIdentity]::GetCurrent()
).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
  Write-Error "Run this in an ADMINISTRATOR PowerShell."
}

$t = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($t) {
  if ($t.State -eq "Running") { Stop-ScheduledTask -TaskName $TaskName }
  Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
  Write-Host "removed scheduled task '$TaskName'" -ForegroundColor Green
} else {
  Write-Host "no task named '$TaskName' - nothing to remove"
}

if ($WithFirewall) {
  $ruleName = "Lesion Atlas ingest $Port (Private)"
  $r = Get-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue
  if ($r) {
    $r | Remove-NetFirewallRule
    Write-Host "removed firewall rule '$ruleName'" -ForegroundColor Green
  } else {
    Write-Host "no firewall rule named '$ruleName'"
  }
}

Write-Host ""
Write-Host "Data, database and images were not touched."

<#
.SYNOPSIS
  Mirror the data root to an external drive or local folder.

.DESCRIPTION
  The code is a weekend. The images cannot be recreated - a session you did
  not shoot is gone permanently, and so is one you shot and then lost.

  Deliberately NOT a cloud service and NOT a git host, per CLAUDE.md. Target
  an external drive or a plain local folder on another physical disk.

  Uses robocopy /MIR with /XO so it is incremental after the first run.

  ONE-OFF:
      .\scripts\backup_data.ps1 -Destination E:\LesionAtlasBackup

  SCHEDULE IT (run as administrator, once):
      .\scripts\backup_data.ps1 -Destination E:\LesionAtlasBackup -Install

  The scheduled job runs daily at 21:00 and also at logon. If the drive is
  not connected it exits quietly rather than failing loudly - a missing
  external drive is normal, not an error.

.PARAMETER Destination
  Backup root. An external drive is the point; a second folder on the same
  physical disk protects against mistakes but not against disk failure.

.PARAMETER Install
  Register a daily scheduled task instead of backing up now.

.PARAMETER Prune
  Allow deletions in the mirror. OFF by default: without it, files removed
  from the source are KEPT in the backup, which is the safer failure mode
  for irreplaceable data.
#>
[CmdletBinding()]
param(
  [Parameter(Mandatory=$true)][string]$Destination,
  [switch]$Install,
  [switch]$Prune
)

$ErrorActionPreference = "Stop"
$TaskName = "LesionAtlasBackup"
$repo = Split-Path -Parent $PSScriptRoot

# Source = the data root config.py resolves to.
$source = $env:LESION_ATLAS_DATA
if (-not $source) { $source = "C:\LesionAtlas\data" }

if ($Install) {
  $isAdmin = ([Security.Principal.WindowsPrincipal] `
    [Security.Principal.WindowsIdentity]::GetCurrent()
  ).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
  if (-not $isAdmin) { Write-Error "Run as administrator to install the scheduled task." }

  $self = Join-Path $PSScriptRoot "backup_data.ps1"
  $argline = "-NoProfile -ExecutionPolicy Bypass -File `"$self`" -Destination `"$Destination`""
  $action  = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $argline -WorkingDirectory $repo
  $t1 = New-ScheduledTaskTrigger -Daily -At 9:00PM
  $t2 = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
  $principal = New-ScheduledTaskPrincipal `
    -UserId ("{0}\{1}" -f $env:USERDOMAIN, $env:USERNAME) -LogonType Interactive -RunLevel Limited
  $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -StartWhenAvailable -MultipleInstances IgnoreNew

  if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
  }
  Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger @($t1,$t2) `
    -Principal $principal -Settings $settings `
    -Description "Mirror Lesion Atlas data root to $Destination" | Out-Null
  Write-Host "registered '$TaskName' -> $Destination (daily 21:00 + at logon)" -ForegroundColor Green
  return
}

# --- run a backup --------------------------------------------------------
if (-not (Test-Path $source)) {
  Write-Warning "source $source does not exist yet - nothing to back up"
  exit 0
}

$destRoot = Split-Path -Qualifier $Destination
if ($destRoot -and -not (Test-Path $destRoot)) {
  Write-Host "drive $destRoot not connected - skipping (this is normal)"
  exit 0
}

New-Item -ItemType Directory -Force -Path $Destination | Out-Null
$logDir = Join-Path $repo "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir "backup.log"

Write-Host "mirroring"
Write-Host "  from : $source"
Write-Host "  to   : $Destination"
Write-Host "  prune: $($Prune.IsPresent)"

# /MIR mirrors; without /PURGE-equivalent behaviour we use /E instead so that
# deletions in the source never propagate unless -Prune was asked for.
$mode = if ($Prune) { "/MIR" } else { "/E" }
$args = @($source, $Destination, $mode, "/R:2", "/W:3", "/MT:8", "/NFL", "/NDL", "/NP",
          "/LOG+:$log", "/TEE")
& robocopy.exe @args
$rc = $LASTEXITCODE

# robocopy: 0-7 are success-ish, 8+ are real failures.
if ($rc -ge 8) {
  Write-Error "robocopy failed with exit code $rc - see $log"
} else {
  $n = (Get-ChildItem $Destination -Recurse -File -ErrorAction SilentlyContinue).Count
  Write-Host "backup ok (robocopy code $rc). $n files at destination." -ForegroundColor Green
  Write-Host "log: $log"
}
exit 0

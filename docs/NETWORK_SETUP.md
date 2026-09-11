# Network setup — phone to laptop

The goal: typing one address on your phone works every morning without
thinking about it. Three things have to hold — the server is running, the
firewall lets the phone in, and the laptop's address does not change.

Plain **HTTP**, no TLS. A `<input type="file">` needs no secure context.
HTTPS (mkcert or Tailscale) is a phase 1 problem, for `getUserMedia` only.

---

## 1 · Firewall — Private networks only

The server binds `0.0.0.0:8000`. Windows Firewall blocks inbound connections
by default, so the phone gets a hang and then a timeout.

Run **as administrator**, once:

```powershell
New-NetFirewallRule `
  -DisplayName "Lesion Atlas ingest 8000 (Private)" `
  -Direction Inbound -Action Allow -Protocol TCP -LocalPort 8000 `
  -Profile Private
```

`-Profile Private` is not optional. It means the rule applies on your home
Wi-Fi and **not** on coffee-shop or campus Wi-Fi. Without it you are serving
several hundred photos of your face to whatever network you happen to join.

`scripts\install_autostart.ps1 -WithFirewall` does exactly this rule for you.

Confirm your home Wi-Fi is actually classified Private:

```powershell
Get-NetConnectionProfile | Select-Object Name, NetworkCategory
```

If it says `Public`, fix it:

```powershell
Set-NetConnectionProfile -Name "<your wifi name>" -NetworkCategory Private
```

To check or remove the rule later:

```powershell
Get-NetFirewallRule -DisplayName "Lesion Atlas*" | Format-List DisplayName, Enabled, Profile
Get-NetFirewallRule -DisplayName "Lesion Atlas*" | Remove-NetFirewallRule
```

---

## 2 · Pin the laptop's IP (DHCP reservation)

Find the current address and the MAC of the Wi-Fi adapter:

```powershell
Get-NetIPAddress -AddressFamily IPv4 |
  Where-Object { $_.InterfaceAlias -like "*Wi-Fi*" } |
  Select-Object IPAddress, InterfaceAlias
Get-NetAdapter -Name "Wi-Fi" | Select-Object Name, MacAddress
```

Then in the router:

1. Open the router admin page — usually `http://192.168.1.1` or
   `http://192.168.0.1`. `ipconfig | findstr Gateway` tells you which.
2. Find **DHCP Reservation** / **Static DHCP** / **Address Reservation**
   (vendors all name it differently; often under LAN or Advanced).
3. Bind the laptop's MAC to its current IP.
4. Reboot the laptop's Wi-Fi and confirm the address did not move.

Without this the laptop gets a new address after a router reboot, the
bookmark on your phone breaks, and the friction that kills adherence is back.

**Alternative if the router will not cooperate:** use the hostname. Windows
advertises `<computername>.local` over mDNS and iOS resolves it natively:

```
http://<computername>.local:8000/
```

Get the name with `hostname`. This survives IP changes entirely. Try it
first — if it works, skip the reservation.

---

## 3 · Confirm the phone can reach the laptop

1. Start the server. It prints the LAN URL and a QR code:
   ```powershell
   python -m src.server.main
   ```
2. Scan the QR with the phone camera, or type the URL.
3. The page should load and show **no red bar**. The red
   "server unreachable" bar means the health check is failing.

If it does not load, in order:

| Check | How |
|---|---|
| Same Wi-Fi? | Phone on the 5 GHz band and laptop on 2.4 GHz of the same router is still the same network — but a **guest** network is isolated and will never work. |
| Server actually up? | On the laptop: `curl http://127.0.0.1:8000/health` |
| Firewall rule present? | `Get-NetFirewallRule -DisplayName "Lesion Atlas*"` |
| Profile is Private? | `Get-NetConnectionProfile` |
| Right IP? | The banner's IP is a best guess. `ipconfig` shows all of them; try the one on the same subnet as the phone. |
| AP isolation | Some routers block client-to-client traffic. Look for "AP Isolation" / "Client Isolation" and turn it off. |

---

## 4 · The laptop must be awake

The server cannot receive an upload while the laptop is asleep, and Task
Scheduler will not wake it for this.

**This is fine and costs you nothing.** If the laptop is asleep when you
shoot, the photos stay in your camera roll and upload later — the upload page
dates the session from the **date field**, not from when the bytes arrived.
Uploading Tuesday's session on Tuesday evening is identical, in the data, to
uploading it Tuesday morning.

What you should not do is skip the shot because the laptop is off. Shoot it,
upload it whenever.

If you want the laptop reliably awake at your capture time:
`Settings → System → Power & battery → Screen and sleep`, or
`powercfg /change standby-timeout-ac 0` while plugged in.

---

## 5 · Autostart

```powershell
# run once, as administrator
powershell -ExecutionPolicy Bypass -File .\scripts\install_autostart.ps1 -WithFirewall
```

Registers a Task Scheduler job that starts the server hidden at logon, pins
the working directory to the repo, and restarts it on failure. No console
window appears.

Check on it:

```powershell
Get-ScheduledTaskInfo -TaskName LesionAtlasIngest
Get-Content .\logs\server.log -Tail 40
```

Remove it:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\uninstall_autostart.ps1 -WithFirewall
```

Logs rotate at 5 MB, 5 files kept — a failure at 7am is still diagnosable at
7pm.

---

## What is deliberately NOT here

- **No HTTPS.** Phase 1, for `getUserMedia` only.
- **No port forwarding, no dynamic DNS, no tunnel.** This server must never
  be reachable from the internet. It has no authentication because it is not
  supposed to need any.
- **No cloud sync of `data/`.** The data root lives at `C:\LesionAtlas\data`,
  deliberately outside the OneDrive-synced repo folder.

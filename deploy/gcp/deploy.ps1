<#
.SYNOPSIS
  Create, use and delete a GPU VM (NVIDIA L4) on Google Cloud that serves the
  roundabout analysis GUI behind HTTPS and a password.

.DESCRIPTION
  Actions:
    up        Create the VM, install everything, print the URL and password.
    status    Show the VM state, URL and service health.
    start     Start a stopped VM (it stops itself when idle) and print the new URL.
    repair    Re-run the install on the existing VM (after a failed `up`).
    firewall  Allow your current public IP again (use after your IP changes).
    password  Set a new password.
    upload    Copy a video to the VM:  -Action upload -File C:\videos\a.mp4
    ssh       Open a shell on the VM.
    down      Delete the VM and the firewall rules (results on the VM are lost).

  Nothing here touches any project but -Project. Every gcloud call passes it.

.EXAMPLE
  .\deploy.ps1 -Action up
  .\deploy.ps1 -Action upload -File D:\video\DJI_0001.MP4
  .\deploy.ps1 -Action down
#>
[CmdletBinding()]
param(
    [ValidateSet('up', 'status', 'start', 'firewall', 'password', 'upload', 'ssh', 'down', 'repair')]
    [string]$Action = 'up',
    [string]$Project = 'dev-project-35318',
    [string]$Region = 'us-central1',
    [string[]]$Zones = @('us-central1-a', 'us-central1-b', 'us-central1-c'),
    [string]$Name = 'traffic-gui',
    [string]$Tag = 'traffic-analysis',
    [string]$HttpsRule = 'ta-allow-https',
    [string]$SshRule = 'ta-allow-ssh',
    [string]$MachineType = 'g2-standard-4',
    [ValidateSet('L4', 'T4')]
    [string]$GpuType = 'L4',
    [int]$DiskGb = 150,
    [string]$Network = 'default',
    [string]$ImageFamily = '',
    [int]$MaxHours = 6,
    [int]$IdleMinutes = 60,
    [string]$UserName = 'analyst',
    [string]$AllowedIp = '',
    [string]$Account = '',
    [string]$File = '',
    [switch]$Spot,
    [switch]$Yes
)

$ErrorActionPreference = 'Stop'
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path

function Invoke-Gcloud {
    param([string[]]$GcloudArgs, [switch]$AllowFail)
    $all = @($GcloudArgs)
    if ($Account) { $all += @('--account', $Account) }
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try { $raw = @(& gcloud @all 2>&1) }
    finally { $ErrorActionPreference = $previous }
    $code = $LASTEXITCODE
    $isErr = { $_ -is [System.Management.Automation.ErrorRecord] }
    $stdout = @($raw | Where-Object { -not (& $isErr) } | ForEach-Object { "$_" })
    $stderr = @($raw | Where-Object { & $isErr } | ForEach-Object { "$_" })
    $text = ($stdout -join "`n")
    $all2 = (($stdout + $stderr) -join "`n")
    if ($code -ne 0 -and -not $AllowFail) {
        throw "gcloud $($GcloudArgs -join ' ') failed (exit $code):`n$all2"
    }
    [pscustomobject]@{ Code = $code; Out = $text; All = $all2 }
}

function Get-Instance {
    $r = Invoke-Gcloud @('compute', 'instances', 'list', '--project', $Project, "--filter=name=$Name", '--format=json')
    if (-not $r.Out.Trim()) { return $null }
    $items = @($r.Out | ConvertFrom-Json)
    if ($items.Count -eq 0) { return $null }
    $vm = $items[0]
    [pscustomobject]@{
        Zone   = ($vm.zone -split '/')[-1]
        Status = $vm.status
        Ip     = $vm.networkInterfaces[0].accessConfigs[0].natIP
    }
}

function Confirm-Step([string]$Question) {
    if ($Yes) { return }
    $answer = Read-Host "$Question [y/N]"
    if ($answer -notmatch '^(y|yes)$') { throw 'Cancelled.' }
}

function Test-Preflight {
    if (-not (Get-Command gcloud -ErrorAction SilentlyContinue)) {
        throw 'gcloud is not installed. Install the Google Cloud CLI first.'
    }
    $active = $Account
    if (-not $active) {
        $r = Invoke-Gcloud @('auth', 'list', '--filter=status:ACTIVE', '--format=value(account)')
        $active = $r.Out.Trim()
    }
    if (-not $active) { throw 'No gcloud account. Run: gcloud auth login' }
    Write-Host "Account: $active"
    Write-Host "Project: $Project   Region: $Region"
    $p = Invoke-Gcloud @('projects', 'describe', $Project, '--format=value(projectId)') -AllowFail
    if ($p.Code -ne 0) {
        throw ("Account '$active' cannot see project '$Project'.`n" +
            "Run 'gcloud auth login' with the account that owns it, or pass -Account.`n$($p.All)")
    }
}

function Test-Quota {
    $r = Invoke-Gcloud @('compute', 'regions', 'describe', $Region, '--project', $Project, '--format=json')
    $metric = "NVIDIA_${GpuType}_GPUS"
    if ($Spot) { $metric = "PREEMPTIBLE_NVIDIA_${GpuType}_GPUS" }
    $quota = @(($r.Out | ConvertFrom-Json).quotas) | Where-Object { $_.metric -eq $metric }
    if (-not $quota) {
        Write-Warning "Could not read the $metric quota; continuing. If VM creation fails, request quota in the console."
        return
    }
    if ($quota.limit -lt 1) {
        throw ("This project has no $metric quota in $Region (limit 0).`n" +
            "Request 1 at: https://console.cloud.google.com/iam-admin/quotas?project=$Project`n" +
            "Filter on '$metric' and region '$Region'. Approval can take hours to days.")
    }
}

function Get-MyIp {
    if ($AllowedIp) { return $AllowedIp }
    $ip = (Invoke-RestMethod -Uri 'https://api.ipify.org' -TimeoutSec 15).ToString().Trim()
    if ($ip -notmatch '^\d{1,3}(\.\d{1,3}){3}$') { throw "Could not detect your public IP (got '$ip'). Pass -AllowedIp." }
    return $ip
}

function Set-FirewallRule([string]$Rule, [string]$Port, [string]$Ip) {
    $exists = Invoke-Gcloud @('compute', 'firewall-rules', 'describe', $Rule, '--project', $Project, '--format=value(name)') -AllowFail
    if ($exists.Code -eq 0) {
        Invoke-Gcloud @('compute', 'firewall-rules', 'update', $Rule, '--project', $Project, "--source-ranges=$Ip/32") | Out-Null
    }
    else {
        Invoke-Gcloud @('compute', 'firewall-rules', 'create', $Rule, '--project', $Project,
            '--network', $Network, '--direction=INGRESS', '--action=ALLOW', "--rules=tcp:$Port",
            "--source-ranges=$Ip/32", "--target-tags=$Tag") | Out-Null
    }
}

function Update-Firewall {
    $ip = Get-MyIp
    Write-Host "Allowing $ip to reach the VM (HTTPS and SSH only)"
    Set-FirewallRule $HttpsRule '443' $ip
    Set-FirewallRule $SshRule '22' $ip
}

function Resolve-ImageFamily {
    if ($ImageFamily) { return $ImageFamily }
    $r = Invoke-Gcloud @('compute', 'images', 'list', '--project', 'deeplearning-platform-release',
        '--no-standard-images', '--filter=family~common-cu', '--sort-by=~creationTimestamp',
        '--limit=300', '--format=json')
    $best = $null
    $bestCuda = 0
    foreach ($image in @($r.Out | ConvertFrom-Json)) {
        if ($image.family -match '^common-cu(\d+)-ubuntu-2204.*nvidia' -and -not $image.deprecated) {
            $cuda = [int]$Matches[1]
            if ($cuda -ge 128 -and $cuda -gt $bestCuda) { $best = $image.family; $bestCuda = $cuda }
        }
    }
    if (-not $best) { throw 'No CUDA 12.8+ Deep Learning VM image family found. Pass -ImageFamily.' }
    return $best
}

function New-Password {
    $alphabet = 'abcdefghijkmnpqrstuvwxyzABCDEFGHJKLMNPQRSTUVWXYZ23456789'
    $bytes = New-Object byte[] 24
    $rng = [System.Security.Cryptography.RandomNumberGenerator]::Create()
    $rng.GetBytes($bytes)
    -join ($bytes | ForEach-Object { $alphabet[$_ % $alphabet.Length] })
}

function Write-LfFile([string]$Path, [string]$Content) {
    [System.IO.File]::WriteAllText($Path, ($Content -replace "`r`n", "`n"), (New-Object System.Text.UTF8Encoding($false)))
}

function Invoke-Ssh([string]$Zone, [string]$Command, [switch]$AllowFail) {
    Invoke-Gcloud @('compute', 'ssh', $Name, '--project', $Project, '--zone', $Zone, '--quiet',
        '--strict-host-key-checking=no', "--command=$Command") -AllowFail:$AllowFail
}

function Copy-ToVm([string]$Zone, [string[]]$Paths, [string]$Destination) {
    $args2 = @('compute', 'scp') + $Paths + @("${Name}:$Destination", '--project', $Project,
        '--zone', $Zone, '--quiet', '--strict-host-key-checking=no')
    Invoke-Gcloud $args2 | Out-Null
}

function Wait-Ssh([string]$Zone) {
    Write-Host 'Waiting for SSH (the first connection also creates an SSH key)...'
    for ($i = 1; $i -le 40; $i++) {
        $r = Invoke-Ssh $Zone 'echo ok' -AllowFail
        if ($r.Code -eq 0) { return }
        Start-Sleep -Seconds 15
    }
    throw 'SSH never became ready.'
}

function Wait-Setup([string]$Zone) {
    Write-Host 'Installing on the VM (GPU driver, packages, Python environment). Usually 10-20 minutes.'
    $deadline = (Get-Date).AddMinutes(45)
    while ((Get-Date) -lt $deadline) {
        $r = Invoke-Ssh $Zone 'cat /var/lib/ta-setup.state 2>/dev/null; sudo journalctl -u ta-setup -n 1 --no-pager -o cat' -AllowFail
        if ($r.Code -eq 0) {
            $lines = $r.Out -split "`n"
            $state = $lines[0].Trim()
            if ($state -eq 'ready') { return }
            if ($state -eq 'failed') {
                $log = Invoke-Ssh $Zone 'sudo journalctl -u ta-setup -n 30 --no-pager -o cat' -AllowFail
                throw "Setup failed on the VM. Last log lines:`n$($log.Out)"
            }
            if ($lines.Count -gt 1) { Write-Host "  $($lines[-1])" }
        }
        Start-Sleep -Seconds 20
    }
    throw 'Setup did not finish in 45 minutes. Check: .\deploy.ps1 -Action ssh, then: sudo journalctl -u ta-setup'
}

function Show-Access([string]$Zone, [string]$Ip, [string]$Password) {
    $fp = Invoke-Ssh $Zone 'cat /var/lib/ta-cert.fingerprint' -AllowFail
    Write-Host ''
    Write-Host '================ ACCESS ================' -ForegroundColor Green
    Write-Host "URL:       https://$Ip/"
    Write-Host "User:      $UserName"
    if ($Password) { Write-Host "Password:  $Password   (shown once, not stored anywhere)" }
    if ($fp.Code -eq 0) {
        Write-Host "The browser warns about the certificate (it is self-signed). Compare its SHA-256"
        Write-Host "fingerprint with:  $($fp.Out.Trim())"
    }
    Write-Host "Only your IP can connect. If it changes: .\deploy.ps1 -Action firewall"
    Write-Host "The VM stops itself after $IdleMinutes idle minutes or $MaxHours hours. Delete it with: .\deploy.ps1 -Action down"
    Write-Host '========================================' -ForegroundColor Green
}

function Send-Password([string]$Zone, [string]$Password) {
    $env1 = Join-Path ([System.IO.Path]::GetTempPath()) "ta_env_$PID"
    $pass1 = Join-Path ([System.IO.Path]::GetTempPath()) "ta_pass_$PID"
    try {
        Write-LfFile $env1 "TA_USER=$UserName`nTA_MAX_HOURS=$MaxHours`nTA_IDLE_MIN=$IdleMinutes`n"
        Write-LfFile $pass1 $Password
        Copy-ToVm $Zone @($env1) '/tmp/ta.env'
        Copy-ToVm $Zone @($pass1) '/tmp/ta_pass'
        Invoke-Ssh $Zone 'chmod 600 /tmp/ta_pass' | Out-Null
    }
    finally {
        Remove-Item -Force -ErrorAction SilentlyContinue $env1, $pass1
    }
}

function Invoke-Up {
    $existing = Get-Instance
    if ($existing) {
        throw "VM '$Name' already exists in $($existing.Zone) ($($existing.Status)). Use -Action start, status or down."
    }
    Test-Quota
    $family = Resolve-ImageFamily
    Write-Host "Image family: $family"
    $kind = 'on-demand'
    if ($Spot) { $kind = 'Spot (cheaper, can be interrupted)' }
    Confirm-Step "Create a $MachineType VM with 1 NVIDIA $GpuType ($kind) in $Project?"

    Update-Firewall

    $zoneUsed = $null
    foreach ($zone in $Zones) {
        Write-Host "Creating the VM in $zone ..."
        $create = @('compute', 'instances', 'create', $Name, '--project', $Project, '--zone', $zone,
            '--machine-type', $MachineType, '--image-family', $family,
            '--image-project', 'deeplearning-platform-release',
            '--boot-disk-size', "${DiskGb}GB", '--boot-disk-type', 'pd-balanced',
            '--maintenance-policy', 'TERMINATE', '--network', $Network, '--tags', $Tag,
            '--metadata', 'install-nvidia-driver=True', '--labels', 'app=traffic-analysis',
            '--no-service-account', '--no-scopes')
        if ($GpuType -eq 'T4') { $create += '--accelerator=type=nvidia-tesla-t4,count=1' }
        if ($Spot) { $create += @('--provisioning-model', 'SPOT', '--instance-termination-action', 'STOP') }
        $r = Invoke-Gcloud $create -AllowFail
        if ($r.Code -eq 0) { $zoneUsed = $zone; break }
        if ($r.All -match 'does not have enough resources|ZONE_RESOURCE_POOL_EXHAUSTED|STOCKOUT|currently unavailable') {
            $why = ($r.All -split "`n" | Where-Object { $_ -match 'ERROR|exhausted|enough resources' } | Select-Object -First 2) -join ' '
            Write-Warning "No $GpuType capacity in $zone, trying the next zone. ($why)"
            continue
        }
        throw "Could not create the VM:`n$($r.All)"
    }
    if (-not $zoneUsed) { throw "No $GpuType capacity in any of: $($Zones -join ', '). Try again later or pass other -Zones." }

    Install-OnVm $zoneUsed
}

function Install-OnVm([string]$Zone) {
    Wait-Ssh $Zone

    $tar = Join-Path ([System.IO.Path]::GetTempPath()) "ta_app_$PID.tar.gz"
    try {
        Write-Host 'Packing the project...'
        & tar -czf $tar -C $RepoRoot --exclude=__pycache__ pyproject.toml uv.lock README.md src configs yolo11m.pt
        if ($LASTEXITCODE -ne 0) { throw 'tar failed' }
        $password = New-Password
        Write-Host 'Uploading the project...'
        Copy-ToVm $Zone @($tar) '/tmp/app.tar.gz'
        $setup = Join-Path ([System.IO.Path]::GetTempPath()) "ta_setup_$PID.sh"
        Write-LfFile $setup (Get-Content -Raw -LiteralPath (Join-Path $PSScriptRoot 'vm_setup.sh'))
        Copy-ToVm $Zone @($setup) '/tmp/vm_setup.sh'
        Remove-Item -Force -ErrorAction SilentlyContinue $setup
        Send-Password $Zone $password
    }
    finally {
        Remove-Item -Force -ErrorAction SilentlyContinue $tar
    }

    Invoke-Ssh $Zone 'sudo systemd-run --unit=ta-setup --collect bash /tmp/vm_setup.sh' | Out-Null
    Wait-Setup $Zone
    $vm = Get-Instance
    Show-Access $Zone $vm.Ip $password
}

function Invoke-Repair {
    $vm = Get-Instance
    if (-not $vm -or $vm.Status -ne 'RUNNING') { throw "VM '$Name' is not running. Use -Action up or start." }
    Update-Firewall
    Install-OnVm $vm.Zone
}

function Invoke-Start {
    $vm = Get-Instance
    if (-not $vm) { throw "VM '$Name' does not exist. Use -Action up." }
    if ($vm.Status -ne 'RUNNING') {
        Invoke-Gcloud @('compute', 'instances', 'start', $Name, '--project', $Project, '--zone', $vm.Zone) | Out-Null
    }
    Update-Firewall
    Wait-Ssh $vm.Zone
    $vm = Get-Instance
    Show-Access $vm.Zone $vm.Ip ''
}

function Invoke-Status {
    $vm = Get-Instance
    if (-not $vm) { Write-Host "VM '$Name' does not exist."; return }
    Write-Host "VM $Name in $($vm.Zone): $($vm.Status)   IP: $($vm.Ip)"
    if ($vm.Status -ne 'RUNNING') { Write-Host 'Start it with: .\deploy.ps1 -Action start'; return }
    $r = Invoke-Ssh $vm.Zone 'systemctl is-active traffic-gui nginx; nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv,noheader' -AllowFail
    Write-Host $r.Out
    Write-Host "URL: https://$($vm.Ip)/"
}

function Invoke-Upload {
    if (-not $File -or -not (Test-Path -LiteralPath $File)) { throw 'Pass an existing file: -Action upload -File C:\path\video.mp4' }
    $vm = Get-Instance
    if (-not $vm -or $vm.Status -ne 'RUNNING') { throw 'The VM is not running. Use -Action up or start.' }
    $leaf = (Split-Path -Leaf $File) -replace '[^A-Za-z0-9._-]', '_'
    Write-Host "Copying $File to the VM as $leaf (large files take a while)..."
    Copy-ToVm $vm.Zone @((Resolve-Path -LiteralPath $File).Path) "/srv/traffic/videos/$leaf"
    Invoke-Ssh $vm.Zone "chmod a+r /srv/traffic/videos/$leaf" | Out-Null
    Write-Host 'Done. Reload the GUI page and pick the video.'
}

function Invoke-Password {
    $vm = Get-Instance
    if (-not $vm -or $vm.Status -ne 'RUNNING') { throw 'The VM is not running.' }
    $password = New-Password
    Send-Password $vm.Zone $password
    Invoke-Ssh $vm.Zone 'sudo /usr/local/bin/ta-set-password' | Out-Null
    Show-Access $vm.Zone $vm.Ip $password
}

function Invoke-Down {
    $vm = Get-Instance
    Confirm-Step "Delete VM '$Name' and its disk (results stored on it are lost) and the firewall rules?"
    if ($vm) {
        Invoke-Gcloud @('compute', 'instances', 'delete', $Name, '--project', $Project, '--zone', $vm.Zone,
            '--delete-disks=all', '--quiet') | Out-Null
        Write-Host 'VM deleted.'
    }
    foreach ($rule in @($HttpsRule, $SshRule)) {
        $r = Invoke-Gcloud @('compute', 'firewall-rules', 'delete', $rule, '--project', $Project, '--quiet') -AllowFail
        if ($r.Code -eq 0) { Write-Host "Firewall rule $rule deleted." }
    }
}

Test-Preflight
switch ($Action) {
    'up' { Invoke-Up }
    'start' { Invoke-Start }
    'repair' { Invoke-Repair }
    'status' { Invoke-Status }
    'firewall' { Update-Firewall }
    'password' { Invoke-Password }
    'upload' { Invoke-Upload }
    'ssh' {
        $vm = Get-Instance
        if (-not $vm) { throw "VM '$Name' does not exist." }
        $all = @('compute', 'ssh', $Name, '--project', $Project, '--zone', $vm.Zone)
        if ($Account) { $all += @('--account', $Account) }
        & gcloud @all
    }
    'down' { Invoke-Down }
}

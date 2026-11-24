#!/usr/bin/env bash
# Runs once on the new GPU VM (started by deploy.ps1 through systemd-run).
# Inputs copied beforehand: /tmp/app.tar.gz, /tmp/ta.env, /tmp/ta_pass.
# Progress: journalctl -u ta-setup   Result: /var/lib/ta-setup.state
set -euo pipefail

STATE=/var/lib/ta-setup.state
APP=/opt/traffic-analysis
DATA=/srv/traffic

log() { echo "[$(date +%T)] $*"; }
trap 'log "FAILED at line $LINENO"; echo failed > "$STATE"' ERR
echo running > "$STATE"

# shellcheck disable=SC1091
source /tmp/ta.env   # TA_USER, TA_MAX_HOURS, TA_IDLE_MIN
EXTERNAL_IP=$(curl -fsS -H 'Metadata-Flavor: Google' \
  http://metadata.google.internal/computeMetadata/v1/instance/network-interfaces/0/access-configs/0/external-ip)

log "waiting for the NVIDIA driver"
for _ in $(seq 1 120); do
  nvidia-smi >/dev/null 2>&1 && break
  sleep 10
done
nvidia-smi >/dev/null 2>&1 || { log "GPU driver never became ready"; exit 1; }
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

log "installing packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y --no-install-recommends ffmpeg nginx openssl curl ca-certificates

log "creating service user and folders"
id traffic >/dev/null 2>&1 || useradd --system --create-home --home-dir /home/traffic \
  --shell /usr/sbin/nologin traffic
mkdir -p "$APP" "$DATA"/{videos,scenes,calibrations,jobs}
tar -xzf /tmp/app.tar.gz -C "$APP"
cp -n "$APP"/configs/*.json "$DATA/scenes/" || true
chown -R traffic:traffic "$APP" "$DATA"
# Any logged-in user may copy videos in (deploy.ps1 -Action upload); files stay readable.
chmod 1777 "$DATA/videos"

log "installing uv and the project"
curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin INSTALLER_NO_MODIFY_PATH=1 sh
sudo -u traffic env HOME=/home/traffic UV_PROJECT_ENVIRONMENT="$APP/.venv" \
  bash -c "cd $APP && uv sync --locked --extra gui --no-dev --python 3.12"
sudo -u traffic "$APP/.venv/bin/python" -c \
  "import torch; assert torch.cuda.is_available(), 'torch cannot see the GPU'; print(torch.cuda.get_device_name(0))"

log "creating the web service"
cat > /etc/systemd/system/traffic-gui.service <<EOF
[Unit]
Description=Roundabout traffic analysis GUI
After=network-online.target

[Service]
User=traffic
WorkingDirectory=$APP
Environment=HOME=/home/traffic
EnvironmentFile=-/etc/traffic-gemini.env
ExecStart=$APP/.venv/bin/traffic-analysis gui --data-dir $DATA --host 127.0.0.1 --port 8501 --max-upload-mb 20000
Restart=on-failure
NoNewPrivileges=true

[Install]
WantedBy=multi-user.target
EOF

log "creating the HTTPS proxy with basic auth"
mkdir -p /etc/nginx/ssl
openssl req -x509 -nodes -newkey rsa:2048 -days 30 \
  -keyout /etc/nginx/ssl/ta.key -out /etc/nginx/ssl/ta.crt \
  -subj "/CN=traffic-gui" -addext "subjectAltName=IP:$EXTERNAL_IP"
chmod 600 /etc/nginx/ssl/ta.key
openssl x509 -in /etc/nginx/ssl/ta.crt -noout -fingerprint -sha256 > /var/lib/ta-cert.fingerprint

cp /tmp/ta.env /etc/ta.env
cat > /usr/local/bin/ta-set-password <<'EOF'
#!/usr/bin/env bash
# Reads the new password from /tmp/ta_pass, installs it, then shreds the file.
set -euo pipefail
source /etc/ta.env
printf '%s:%s\n' "$TA_USER" "$(openssl passwd -apr1 -stdin < /tmp/ta_pass)" > /etc/nginx/.htpasswd
chown root:www-data /etc/nginx/.htpasswd
chmod 640 /etc/nginx/.htpasswd
shred -u /tmp/ta_pass
systemctl reload nginx 2>/dev/null || true
EOF
chmod 755 /usr/local/bin/ta-set-password
/usr/local/bin/ta-set-password

rm -f /etc/nginx/sites-enabled/default
cat > /etc/nginx/conf.d/ta.conf <<'EOF'
map $http_upgrade $connection_upgrade {
    default upgrade;
    ''      close;
}

server {
    listen 443 ssl;
    server_name _;
    ssl_certificate     /etc/nginx/ssl/ta.crt;
    ssl_certificate_key /etc/nginx/ssl/ta.key;
    ssl_protocols       TLSv1.2 TLSv1.3;

    auth_basic           "Roundabout analysis";
    auth_basic_user_file /etc/nginx/.htpasswd;

    client_max_body_size 0;
    proxy_request_buffering off;

    location / {
        proxy_pass http://127.0.0.1:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection $connection_upgrade;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 86400;
        proxy_send_timeout 86400;
    }
}
EOF
nginx -t

log "creating the cost guards (idle and hard-deadline shutdown)"
cat > /usr/local/bin/ta-watchdog <<'EOF'
#!/usr/bin/env bash
# Stops the VM when no job is running and nobody is connected for TA_IDLE_MIN minutes.
source /etc/ta.env
STATE=/run/ta-idle-since
active=0
pgrep -f traffic_analysis.jobs >/dev/null && active=1
ss -Htn state established '( sport = :443 )' | grep -q . && active=1
now=$(date +%s)
if [ "$active" -eq 1 ]; then echo "$now" > "$STATE"; exit 0; fi
[ -f "$STATE" ] || echo "$now" > "$STATE"
since=$(cat "$STATE")
if [ $(( (now - since) / 60 )) -ge "$TA_IDLE_MIN" ]; then
  logger -t ta-watchdog "idle for $TA_IDLE_MIN minutes, shutting down"
  shutdown -h now
fi
EOF
chmod 755 /usr/local/bin/ta-watchdog

cat > /etc/systemd/system/ta-watchdog.service <<'EOF'
[Unit]
Description=Shut down the VM when idle

[Service]
Type=oneshot
ExecStart=/usr/local/bin/ta-watchdog
EOF
cat > /etc/systemd/system/ta-watchdog.timer <<'EOF'
[Unit]
Description=Check for idleness every 5 minutes

[Timer]
OnBootSec=10min
OnUnitActiveSec=5min

[Install]
WantedBy=timers.target
EOF
cat > /etc/systemd/system/ta-deadline.service <<EOF
[Unit]
Description=Hard limit on VM run time

[Service]
Type=oneshot
ExecStart=/sbin/shutdown -h +$((TA_MAX_HOURS * 60))
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable --now traffic-gui.service nginx.service ta-watchdog.timer ta-deadline.service
systemctl reload nginx || systemctl restart nginx

log "waiting for the GUI to answer"
for _ in $(seq 1 30); do
  curl -fsS http://127.0.0.1:8501/_stcore/health >/dev/null 2>&1 && break
  sleep 2
done
curl -fsS http://127.0.0.1:8501/_stcore/health >/dev/null

rm -f /tmp/app.tar.gz
echo ready > "$STATE"
log "ready"

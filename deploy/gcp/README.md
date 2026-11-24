# Run the GUI on a Google Cloud GPU VM

`deploy.ps1` creates one VM (`g2-standard-4`, 1 NVIDIA L4) in project
`dev-project-35318`, installs the app on it, and prints an HTTPS address with a
user name and password. You upload a video in the browser, press Go and read the
counts. The VM stops itself when idle.

## One-time setup (about 15 minutes, plus quota approval)

1. **Sign in with the right Google account.** The account must be allowed to use
   `dev-project-35318`. The script prints the account it uses.

   ```powershell
   gcloud auth login
   gcloud config set account YOU@example.com
   ```

   Or pass `-Account YOU@example.com` to the script.
2. **Enable billing** on the project (Console > Billing).
3. **Roles the account needs on the project** (an Owner has them all):
   - `Compute Admin` (`roles/compute.admin`): create the VM and firewall rules.
   - `Service Usage Admin` (`roles/serviceusage.serviceUsageAdmin`): enable the
     Compute Engine API the first time. Skip it if the API is already on.
   - `Service Account User` (`roles/iam.serviceAccountUser`): only if your
     organisation forces a service account on VMs. The VM here is created with
     **no service account**, so it cannot read or change anything in your project.
4. **Ask for GPU quota.** New projects usually have 0 L4 GPUs.
   Console > IAM & Admin > Quotas, filter `NVIDIA_L4_GPUS`, region `us-central1`,
   request **1**. For Spot VMs request `PREEMPTIBLE_NVIDIA_L4_GPUS` instead.
   The script checks this and tells you if it is missing.
5. **Set a budget alert**: Console > Billing > Budgets & alerts > create a
   budget of 10 USD with alerts at 50% and 100%.
6. Windows needs the OpenSSH client (`ssh -V` works) and `tar` (both ship with
   Windows 10/11).

## Everyday use

```powershell
cd deploy\gcp
.\deploy.ps1 -Action up                 # create everything, takes 10-20 min
.\deploy.ps1 -Action upload -File D:\video\DJI_0001.MP4   # fast way for big videos
.\deploy.ps1 -Action status
.\deploy.ps1 -Action down               # delete VM, disk and firewall rules
```

`up` prints the address, user and password once. Open the address, accept the
certificate warning (the certificate is self-signed; the script prints its
SHA-256 fingerprint so you can compare it), and log in.

In the GUI:
1. **New run**: pick or upload the video, pick the scene (zones) file and a
   detection model under **Options**, then press **Go**. Standard YOLO11m,
   YOLO11l and YOLO11x and experimental aerial YOLO11m-OBB are available.
   The first use of a model may download its weights.
2. **Results**: live progress, then the roundabout counts, wrong-way entry clips,
   speeds and a zip of all result files. The selected model is shown with each
   run. Download what you need before `down`.

The zones are drawn on your PC with `traffic-analysis zones` (a desktop window).
Upload the scene JSON in the GUI, or put it in `configs/` before `up`; the GUI
can preview the zones on a video frame.

### Optional Gemini frame review

This sends sampled JPEGs, timestamps, and scene geometry to the Gemini API,
not the video file or YOLO results. Explicit consent is required per run, and
API usage may incur charges. It is an independent, experimental comparison,
not a validated count. Five-second batches can miss or duplicate movements
across boundaries. Runs are restricted to at most 60 seconds and 10 fps.

Create a Gemini API key separately and keep it out of chat and source files.
On the VM, use `sudo nano /etc/traffic-gemini.env` to create a file containing
`GEMINI_API_KEY=YOUR_KEY` (replace the placeholder with your actual key), then:

```bash
sudo chown root:root /etc/traffic-gemini.env
sudo chmod 600 /etc/traffic-gemini.env
sudo systemctl restart traffic-gui
```

New VMs read this optional file via systemd. The optional
`GEMINI_COUNT_MODEL` variable can select a compatible Gemini model; the default
is `gemini-2.5-flash`. Existing VMs need the same `EnvironmentFile` setting
in a systemd override before restarting. Avoid printing the file or the key.

## Access control

- **Firewall**: only your current public IP can reach ports 443 (the GUI) and 22
  (SSH). If your IP changes (new network, VPN): `.\deploy.ps1 -Action firewall`.
  Use `-AllowedIp x.x.x.x` to set it by hand.
- **Password**: HTTP basic auth in front of the app, with a random 24-character
  password. The app itself listens only on `127.0.0.1`. Rotate it with
  `.\deploy.ps1 -Action password`.
- **Encryption**: HTTPS with a self-signed certificate.
- **SSH**: normal `gcloud compute ssh` keys, no passwords.
- The VM has no service account and no access to other Google Cloud resources.

## Cost controls

| Guard | Default | Change with |
|---|---|---|
| Stops when no job runs and no browser is connected | 60 min | `-IdleMinutes` |
| Hard stop after boot | 6 hours | `-MaxHours` |
| Cheaper, interruptible VM | off | `-Spot` |

A stopped VM costs only its disk (about 15 USD a month for 150 GB), no GPU.
`.\deploy.ps1 -Action start` boots it again (new IP, same disk). Delete
everything with `-Action down` when you finish.

## Troubleshooting

- `Account ... cannot see project`: wrong gcloud account; see step 1.
- `no NVIDIA_L4_GPUS quota`: see step 4.
- `No L4 capacity in any of ...`: try later, or `-Zones us-central1-a,us-central1-c`
  or `-Spot`.
- Setup fails: `.\deploy.ps1 -Action ssh`, then
  `sudo journalctl -u ta-setup -n 100 --no-pager`.
- Browser upload gives a 403 or stalls: use `-Action upload` instead.
- GUI errors: `sudo journalctl -u traffic-gui -n 100 --no-pager`.

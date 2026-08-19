# CARLA session runbook — one evening, ~$1-3 of GPU time

Goal: run `drive_and_detect.py` against a real CARLA 0.9.16 server on a rented
Linux + GPU machine, collect three weather runs, push the results.

All facts below verified Aug 2026 (release URLs, wheel support, flags, prices).

## Option 0 — local Linux laptop with a 4 GB NVIDIA GPU (free, try first)

Below the official minimum (8 GB VRAM) but often workable with tight settings.
Needs: NVIDIA proprietary driver + Vulkan working, ~30 GB free disk.
Same steps as sections 1-4 below (skip the rental), with these changes:

```bash
# server: lowest footprint
./CarlaUE4.sh -RenderOffScreen -nosound -quality-level=Low

# runs: light town, small camera, YOLO on CPU so CARLA keeps the whole GPU
python drive_and_detect.py --frames 600 --weather ClearNoon    --town Town02 --width 640 --height 360 --device cpu
python drive_and_detect.py --frames 600 --weather HardRainNoon --width 640 --height 360 --device cpu
python drive_and_detect.py --frames 600 --weather ClearNight   --width 640 --height 360 --device cpu
```

(`--town` only on the first run — it persists. Sync mode means a slow sim is
fine: the run just takes longer, the data is identical.)
If the server crashes with an out-of-memory / Vulkan device-lost error, don't
fight it — fall back to the rented pod below.

## 0. Rent the machine (~5 min)

**RunPod** (simplest): deploy a Pod — GPU **RTX 3090** (Community, ~$0.22/hr)
or 4090 (~$0.34/hr), template **Ubuntu 22.04 + CUDA** (any recent PyTorch/CUDA
image works), **disk ≥ 50 GB**. Per-second billing.
Alternative: vast.ai kvm VM (filter `vms_enabled=true`), same specs.

> Why not the `carlasim/carla` docker image directly: on these platforms you
> don't control the `docker run` flags, and CARLA needs the `graphics` NVIDIA
> capability for Vulkan. A plain CUDA pod + tarball avoids the whole issue.

## 1. Sanity-check Vulkan FIRST (30 s — do not skip)

```bash
apt update && apt install -y vulkan-tools wget git python3.10-venv libjpeg8 libtiff5 xdg-user-dirs
vulkaninfo --summary | head -20
```

You must see the NVIDIA GPU listed. If you get "Cannot find a compatible
Vulkan Driver (ICD)":

```bash
ls -la /usr/share/vulkan/icd.d/
# known packaging bug: nvidia_icd.json is sometimes an EMPTY DIRECTORY
# if so: rm -rf /usr/share/vulkan/icd.d/nvidia_icd.json and recreate:
cat > /usr/share/vulkan/icd.d/nvidia_icd.json <<'EOF'
{ "file_format_version" : "1.0.0",
  "ICD": { "library_path": "libGLX_nvidia.so.0", "api_version" : "1.3" } }
EOF
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
```

If vulkaninfo still shows nothing, the pod lacks the graphics capability —
kill it and rent another (cheap, per-second billing). Do NOT work around with
`-nullrhi`: it silently disables rendering and cameras return nothing.

## 2. Install CARLA server (~10 min download on datacenter bandwidth)

```bash
wget -O carla.tar.gz https://tiny.carla.org/carla-0-9-16-linux   # 7.8 GiB
mkdir carla-sim && tar -xzf carla.tar.gz -C carla-sim
cd carla-sim
./CarlaUE4.sh -RenderOffScreen -nosound -quality-level=Low &
```

Startup takes 30-90 s before port 2000 accepts connections. `-quality-level`
accepts only `Low` or `Epic`. Do NOT pass `-carla-server` (dead 0.8.x flag).

## 3. Install the client (~2 min)

```bash
cd ~ && git clone https://github.com/include07/ad-perception-bench.git
cd ad-perception-bench
python3 -m venv .venv && source .venv/bin/activate   # python 3.10, 3.11 or 3.12 (carla 0.9.16 wheels)
pip install carla==0.9.16 ultralytics opencv-python-headless
# client MUST match server version — a mismatch prints an RPC warning and breaks
```

## 4. The three runs (~15 min total)

```bash
cd carla
python drive_and_detect.py --frames 600 --weather ClearNoon
python drive_and_detect.py --frames 600 --weather HardRainNoon
python drive_and_detect.py --frames 600 --weather ClearNight
```

Each run: ego + 30 background vehicles on autopilot, 600 synchronous ticks at
20 Hz, YOLOv8 on every frame → `runs/carla_log_<weather>.csv` + annotated
frames in `runs/carla_frames/` (every 20th).

Quick look at what the detector saw per weather:

```bash
for f in ../runs/carla_log_*.csv; do
  echo "$f: $(tail -n +2 $f | awk -F, '{s+=$2} END {print s " detections over " NR " frames"}')"
done
```

## 5. Push the results, kill the pod

```bash
cd ~/ad-perception-bench
gh auth login          # device flow, or: git remote set-url origin https://<PAT>@github.com/include07/ad-perception-bench.git
git add runs/carla_log_*.csv
# frames dir is gitignored — force-add ONE good sample per weather, not all 90:
git add -f runs/carla_frames/ClearNoon_00200.jpg runs/carla_frames/HardRainNoon_00200.jpg runs/carla_frames/ClearNight_00200.jpg
git commit -m "Phase 2: CARLA 0.9.16 closed-loop runs (3 weathers, 600 frames each)"
git push
```

Then STOP THE POD (billing stops). Back on the Mac: `git pull`, update the
README (check the Phase 2 box, add a results line + one annotated frame), push.

## Budget

Download + setup ~25 min, runs ~15 min, margin ~20 min → **about 1 hour of
GPU billing ≈ $0.25-0.75**. Even with fumbling, hard to exceed $3.

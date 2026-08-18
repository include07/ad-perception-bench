# AD Perception Bench

A hands-on lab for **validating perception models the way AD teams validate software**: run a detector against ground-truth driving data, measure it (mAP, per-class AP, FPS), track results across runs — then close the loop in simulation.

Built as a learning project toward **SiL-style validation** of autonomous-driving stacks.

## Status (honest)

- [x] Scaffold: evaluation harness + FPS bench + CARLA client code
- [ ] **Phase 1 — run on my machine**: driving-class evaluation on COCO val2017, results table below
- [ ] **Phase 2 — CARLA closed-loop** (requires Linux + GPU): ego vehicle + camera + live YOLO inference in the simulator

> Phase 2 code lives in [`carla/`](carla/) and is **untested until checked off** — CARLA has no macOS build, so it waits for a Linux/GPU session.

## Why this shape

AD stacks are validated by simulating **billions of kilometers** and measuring everything. The core skills are the same at any scale: a reproducible harness, ground truth, metrics you can defend, and a loop you can re-run after every change. That is what this repo practices:

1. **Evaluate** — YOLOv8 vs COCO ground truth, restricted to the 9 driving-relevant classes (person, bicycle, car, motorcycle, bus, train, truck, traffic light, stop sign).
2. **Benchmark** — FPS on the target hardware (supports Apple `mps`, CUDA, CPU).
3. **Report** — every run appends to `runs/results.csv`; the table below is regenerated from it.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

make smoke   # 2-min sanity check on coco128
make eval    # real numbers: COCO val2017 (downloads ~1GB once), driving classes only
make fps     # inference FPS on this machine
```

## Results

*(to be filled from my runs — `runs/results.csv`)*

| Run | Model | Dataset | mAP50-95 (driving classes) | FPS | Hardware |
|---|---|---|---|---|---|
| — | — | — | — | — | — |

## Roadmap

- [ ] Compare yolov8n / yolov8s / yolov8m: accuracy-vs-FPS tradeoff curve
- [ ] Per-class failure analysis (which driving classes degrade first?)
- [ ] CARLA Phase 2: same detector, live camera stream, scenario variations (weather, night)
- [ ] Log CARLA detections in the same results schema → one harness, two data sources

## Layout

```
src/evaluate.py     # ultralytics val restricted to driving classes → runs/results.csv
src/bench_fps.py    # FPS benchmark (mps/cuda/cpu)
carla/              # Phase 2: CARLA 0.9.15 client (UNTESTED — Linux+GPU required)
```

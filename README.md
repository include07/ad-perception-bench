# AD Perception Bench

A hands-on lab for **validating perception models the way AD teams validate software**: run a detector against ground-truth driving data, measure it (mAP, per-class AP, FPS), track results across runs — then close the loop in simulation.

Built as a learning project toward **SiL-style validation** of autonomous-driving stacks.

## Status

- [x] Scaffold: evaluation harness + FPS bench + CARLA client code
- [x] **Phase 1**: driving-class evaluation on COCO val2017 — results table below (2026-08-18)
- [ ] **Phase 2 — CARLA closed-loop**: ego vehicle + camera + live YOLO inference in the simulator

> Phase 2 code lives in [`carla/`](carla/) and has not been run yet — CARLA has no macOS build; it requires a Linux + GPU machine.

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

*(raw data in `runs/results.csv`)*

| Run | Model | Dataset | mAP50-95 (driving classes) | mAP50 | FPS @640 | Hardware |
|---|---|---|---|---|---|---|
| 2026-08-18 | yolov8n | COCO val2017 (5,000 img) | **0.436** | 0.612 | 76.2 (mps) / ~8 (cpu) | Apple M1 Pro |
| 2026-08-18 | yolov8n | coco128 (smoke — train overlap, not a valid eval) | 0.455 | 0.592 | — | Apple M1 Pro |

First failure-analysis takeaways: small objects hurt most — **traffic light AP 0.21**, truck 0.29, bicycle 0.26 vs train 0.64, bus 0.62. Consistent with the small-object AP gap (AP-small 0.19 vs AP-large 0.54).

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

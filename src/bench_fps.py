"""Benchmark inference FPS on this machine (mps / cuda / cpu).

Appends each run to runs/fps.csv so plot_tradeoff.py can join latency
with the mAP results.
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np
from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--device", default="mps", help="mps | cuda | cpu")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    model = YOLO(args.model)
    frame = np.random.randint(0, 255, (args.imgsz, args.imgsz, 3), dtype=np.uint8)

    for _ in range(10):  # warmup
        model(frame, device=args.device, verbose=False)

    start = time.perf_counter()
    for _ in range(args.n):
        model(frame, device=args.device, verbose=False)
    elapsed = time.perf_counter() - start

    fps = args.n / elapsed
    ms = elapsed / args.n * 1000
    print(f"{args.model} @ {args.imgsz}px on {args.device}: "
          f"{fps:.1f} FPS ({ms:.1f} ms/frame)")

    out = Path("runs/fps.csv")
    out.parent.mkdir(exist_ok=True)
    is_new = not out.exists()
    with out.open("a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["model", "device", "imgsz", "ms_per_frame", "fps"])
        writer.writerow([args.model, args.device, args.imgsz,
                         round(ms, 2), round(fps, 2)])


if __name__ == "__main__":
    main()

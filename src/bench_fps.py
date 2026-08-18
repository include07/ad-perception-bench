"""Benchmark inference FPS on this machine (mps / cuda / cpu)."""
import argparse
import time

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

    print(f"{args.model} @ {args.imgsz}px on {args.device}: "
          f"{args.n / elapsed:.1f} FPS ({elapsed / args.n * 1000:.1f} ms/frame)")


if __name__ == "__main__":
    main()

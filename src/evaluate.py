"""Evaluate a YOLO model on driving-relevant classes against COCO ground truth.

Appends one row per run to runs/results.csv so results accumulate across
models/datasets and the README table can be regenerated from one file.
"""
import argparse
import csv
import datetime
import platform
from pathlib import Path

from ultralytics import YOLO

# COCO class ids relevant to driving scenes
DRIVING_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    6: "train",
    7: "truck",
    9: "traffic light",
    11: "stop sign",
}

RESULTS_CSV = Path("runs/results.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--data", default="coco128.yaml",
                        help="coco128.yaml (smoke) or coco.yaml (real val2017)")
    parser.add_argument("--device", default=None,
                        help="mps | cuda | cpu (auto if omitted)")
    args = parser.parse_args()

    model = YOLO(args.model)
    metrics = model.val(
        data=args.data,
        classes=sorted(DRIVING_CLASSES),
        device=args.device,
        plots=True,
    )

    RESULTS_CSV.parent.mkdir(exist_ok=True)
    is_new = not RESULTS_CSV.exists()
    with RESULTS_CSV.open("a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["timestamp", "model", "dataset",
                             "map50_95", "map50", "hardware"])
        writer.writerow([
            datetime.datetime.now().isoformat(timespec="seconds"),
            args.model,
            args.data,
            round(metrics.box.map, 4),
            round(metrics.box.map50, 4),
            platform.machine(),
        ])

    print(f"\nmAP50-95 (driving classes): {metrics.box.map:.4f}")
    print(f"mAP50              : {metrics.box.map50:.4f}")
    print(f"Appended to {RESULTS_CSV}")
    print("Per-class AP: see the printed table above and runs/detect/val*/")


if __name__ == "__main__":
    main()

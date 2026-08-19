"""Plot the accuracy-vs-latency tradeoff from accumulated runs.

Joins runs/results.csv (mAP per model, val2017 rows only) with runs/fps.csv
(latency per model/device) and writes runs/tradeoff.png.
"""
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_csv(path: Path) -> list:
    with path.open() as f:
        return list(csv.DictReader(f))


def main() -> None:
    results = read_csv(Path("runs/results.csv"))
    fps_rows = read_csv(Path("runs/fps.csv"))

    # keep only real evaluations (val2017), last run per model
    map_by_model = {}
    for row in results:
        if "coco.yaml" in row["dataset"]:
            map_by_model[row["model"]] = float(row["map50_95"])

    # last FPS run per (model, device)
    fps_by_model = defaultdict(dict)
    for row in fps_rows:
        fps_by_model[row["model"]][row["device"]] = float(row["ms_per_frame"])

    fig, ax = plt.subplots(figsize=(7, 5))
    for model, map_val in sorted(map_by_model.items()):
        for device, ms in fps_by_model.get(model, {}).items():
            ax.scatter(ms, map_val, s=80)
            ax.annotate(f"{model.replace('.pt', '')} ({device})",
                        (ms, map_val), textcoords="offset points",
                        xytext=(8, 4), fontsize=9)

    ax.set_xlabel("Latency (ms/frame)")
    ax.set_ylabel("mAP50-95 — driving classes, COCO val2017")
    ax.set_title("Accuracy vs latency (Apple M1 Pro)")
    ax.grid(True, alpha=0.3)
    out = Path("runs/tradeoff.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out}")
    if not map_by_model:
        print("NB: no val2017 rows found yet — run evaluate with --data coco.yaml first")


if __name__ == "__main__":
    main()

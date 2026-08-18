"""Phase 2 (UNTESTED — requires a running CARLA 0.9.15 server on Linux + GPU).

Spawns an ego vehicle on autopilot, attaches an RGB camera, runs YOLOv8 on
every frame in synchronous mode, and logs detections to runs/carla_log.csv
with annotated frames in runs/carla_frames/.

Usage (once the CARLA server is running):
    python drive_and_detect.py --host 127.0.0.1 --frames 600 --weather ClearNoon
"""
import argparse
import csv
import queue
from pathlib import Path

import carla  # pip install carla==0.9.15
import cv2
import numpy as np
from ultralytics import YOLO

OUT_DIR = Path("runs/carla_frames")
LOG_CSV = Path("runs/carla_log.csv")


def to_bgr(image: "carla.Image") -> np.ndarray:
    arr = np.frombuffer(image.raw_data, dtype=np.uint8)
    arr = arr.reshape((image.height, image.width, 4))
    return arr[:, :, :3].copy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--frames", type=int, default=600)
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--weather", default="ClearNoon",
                        help="ClearNoon | WetNoon | HardRainNoon | ClearNight ...")
    args = parser.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.get_world()
    world.set_weather(getattr(carla.WeatherParameters, args.weather))

    # Synchronous mode so simulation and inference stay in lockstep (SiL-style)
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05  # 20 Hz
    world.apply_settings(settings)

    blueprints = world.get_blueprint_library()
    ego_bp = blueprints.filter("vehicle.tesla.model3")[0]
    spawn = world.get_map().get_spawn_points()[0]
    ego = world.spawn_actor(ego_bp, spawn)
    ego.set_autopilot(True)

    cam_bp = blueprints.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", "1280")
    cam_bp.set_attribute("image_size_y", "720")
    cam_tf = carla.Transform(carla.Location(x=1.5, z=1.7))
    camera = world.spawn_actor(cam_bp, cam_tf, attach_to=ego)

    frames: "queue.Queue[carla.Image]" = queue.Queue()
    camera.listen(frames.put)

    model = YOLO(args.model)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_CSV.parent.mkdir(exist_ok=True)

    try:
        with LOG_CSV.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["frame", "n_detections", "classes", "weather"])
            for i in range(args.frames):
                world.tick()
                image = frames.get(timeout=5.0)
                bgr = to_bgr(image)
                result = model(bgr, verbose=False)[0]
                names = [result.names[int(c)] for c in result.boxes.cls]
                writer.writerow([i, len(names), "|".join(names), args.weather])
                if i % 20 == 0:  # save every 20th annotated frame
                    cv2.imwrite(str(OUT_DIR / f"{i:05d}.jpg"), result.plot())
        print(f"Done: {args.frames} frames -> {LOG_CSV}, samples in {OUT_DIR}")
    finally:
        settings.synchronous_mode = False
        world.apply_settings(settings)
        camera.destroy()
        ego.destroy()


if __name__ == "__main__":
    main()

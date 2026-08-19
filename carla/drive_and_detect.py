"""Phase 2: closed-loop CARLA run (requires a CARLA server on Linux + GPU).

Spawns an ego vehicle on autopilot plus background traffic, attaches an RGB
camera, runs YOLOv8 on every frame in synchronous mode, and logs detections
to runs/carla_log_<weather>.csv with annotated frames in runs/carla_frames/.

Usage (once the CARLA server is running):
    python drive_and_detect.py --frames 600 --weather ClearNoon
    python drive_and_detect.py --frames 600 --weather HardRainNoon
    python drive_and_detect.py --frames 600 --weather ClearNight
"""
import argparse
import csv
import queue
import random
from pathlib import Path

import carla  # pip install carla (must match the server version)
import cv2
import numpy as np
from ultralytics import YOLO

OUT_DIR = Path("runs/carla_frames")


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
    parser.add_argument("--traffic", type=int, default=30,
                        help="background vehicles to spawn")
    parser.add_argument("--weather", default="ClearNoon",
                        help="ClearNoon | WetNoon | HardRainNoon | ClearNight ...")
    args = parser.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(20.0)
    world = client.get_world()
    world.set_weather(getattr(carla.WeatherParameters, args.weather))

    # Synchronous mode: simulation, traffic manager, and inference in lockstep.
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05  # 20 Hz
    world.apply_settings(settings)
    tm = client.get_trafficmanager()
    tm.set_synchronous_mode(True)

    blueprints = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)

    # Ego vehicle: take the first spawn point that works.
    ego_bp = blueprints.filter("vehicle.tesla.model3")[0]
    ego = None
    for sp in spawn_points:
        ego = world.try_spawn_actor(ego_bp, sp)
        if ego is not None:
            break
    if ego is None:
        raise RuntimeError("no free spawn point for the ego vehicle")
    ego.set_autopilot(True, tm.get_port())

    # Background traffic so the detector has something to detect.
    vehicle_bps = blueprints.filter("vehicle.*")
    traffic = []
    for sp in spawn_points:
        if len(traffic) >= args.traffic:
            break
        actor = world.try_spawn_actor(random.choice(vehicle_bps), sp)
        if actor is not None:
            actor.set_autopilot(True, tm.get_port())
            traffic.append(actor)

    cam_bp = blueprints.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", "1280")
    cam_bp.set_attribute("image_size_y", "720")
    cam_tf = carla.Transform(carla.Location(x=1.5, z=1.7))
    camera = world.spawn_actor(cam_bp, cam_tf, attach_to=ego)

    frames: "queue.Queue[carla.Image]" = queue.Queue()
    camera.listen(frames.put)

    model = YOLO(args.model)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log_csv = Path(f"runs/carla_log_{args.weather}.csv")

    try:
        with log_csv.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["frame", "n_detections", "classes", "weather"])
            for i in range(args.frames):
                world.tick()
                image = frames.get(timeout=10.0)
                bgr = to_bgr(image)
                result = model(bgr, verbose=False)[0]
                names = [result.names[int(c)] for c in result.boxes.cls]
                writer.writerow([i, len(names), "|".join(names), args.weather])
                if i % 20 == 0:  # save every 20th annotated frame
                    cv2.imwrite(str(OUT_DIR / f"{args.weather}_{i:05d}.jpg"),
                                result.plot())
        print(f"Done: {args.frames} frames -> {log_csv}, samples in {OUT_DIR}")
    finally:
        camera.stop()
        camera.destroy()
        for actor in traffic:
            actor.destroy()
        ego.destroy()
        tm.set_synchronous_mode(False)
        settings.synchronous_mode = False
        world.apply_settings(settings)


if __name__ == "__main__":
    main()

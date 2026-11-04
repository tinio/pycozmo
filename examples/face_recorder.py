#!/usr/bin/env python
"""Continuously record video while Cozmo seeks out face activity with people.

The robot keeps its camera streaming and writes every frame to disk as JPEG,
with a JSONL sidecar describing what faces were in sight. When faces are
visible it turns toward the nearest one to keep people centered; when no face
has been seen for a while it slowly scans in place to find people.

Needs OpenCV and the face models for tracking (pip install
opencv-python-headless, then pycozmo_faces.py download). Without them it
still records video, it just does not turn toward faces.

Usage:
    python examples/face_recorder.py [--output-dir DIR] [--duration S]
                                     [--fps FPS] [--no-move] [--color]
"""

import argparse
import json
import math
import queue
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

import pycozmo
import pycozmo.event
import pycozmo.util


SCAN_AFTER_S = 5.0
TURN_STEP_RAD = math.radians(30.0)
AIM_TOLERANCE_RAD = math.radians(8.0)
SETTLE_S = 0.3


def wrap_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi]."""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def bearing_to_face(robot_x: float, robot_y: float, heading: float, face_x: float, face_y: float) -> float:
    """Which way a face is from where the robot looks, left positive, in radians."""
    return wrap_angle(math.atan2(face_y - robot_y, face_x - robot_x) - heading)


def should_scan(now: float, last_face_time: Optional[float], scan_after_s: float = SCAN_AFTER_S) -> bool:
    """Whether no face has been seen recently enough to start scanning."""
    if last_face_time is None:
        return True
    return (now - last_face_time) >= scan_after_s


def build_output_dir(base: str, when: Optional[datetime] = None) -> Path:
    """A timestamped recording directory under base."""
    stamp = (when or datetime.now()).strftime("%Y%m%d_%H%M%S")
    return Path(base) / ("face_recorder_" + stamp)


def save_frame(image: Image.Image, path: Path) -> None:
    """Write one camera frame as JPEG, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(str(path), "JPEG", quality=90)


def append_record(log_path: Path, record: Dict[str, Any]) -> None:
    """Append one JSON record to a JSONL log."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def list_frames(frames_dir: Path) -> List[Path]:
    """Recorded frames in order."""
    return sorted(frames_dir.glob("frame_*.jpg"))


def assemble_video(frames_dir: Path, output_path: Path, fps: float = 14.0) -> Optional[Path]:
    """Assemble recorded JPEGs into a video file if a writer is available.

    Tries imageio, then OpenCV, then falls back to an animated GIF via Pillow.
    Returns the written path, or None when no frames were found.
    """
    frames = list_frames(frames_dir)
    if not frames:
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Prefer MP4 via imageio.
    try:
        import imageio.v2 as imageio  # type: ignore

        with imageio.get_writer(str(output_path), fps=fps) as writer:
            for frame in frames:
                writer.append_data(imageio.imread(str(frame)))
        return output_path
    except Exception:
        pass
    # Then OpenCV.
    try:
        import cv2  # type: ignore

        first = cv2.imread(str(frames[0]))
        if first is None:
            raise ValueError("unreadable first frame")
        height, width = first.shape[:2]
        out = output_path.with_suffix(".avi") if output_path.suffix.lower() == ".mp4" else output_path
        writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height))
        for frame in frames:
            img = cv2.imread(str(frame))
            if img is not None:
                writer.write(img)
        writer.release()
        return out
    except Exception:
        pass
    # Fallback: animated GIF preview of every 2nd frame, downscaled.
    try:
        images = []
        for frame in frames[::2][:300]:
            with Image.open(frame) as im:
                images.append(im.convert("RGB").resize((160, 120)))
        if not images:
            return None
        gif_path = output_path.with_suffix(".gif")
        images[0].save(str(gif_path), save_all=True, append_images=images[1:], duration=70, loop=0)
        return gif_path
    except Exception:
        return None


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=None,
                        help="recording directory (default: recordings/face_recorder_<time>)")
    parser.add_argument("--duration", type=float, default=0.0, help="stop after seconds (0 = run until Ctrl-C)")
    parser.add_argument("--fps", type=float, default=14.0, help="max frames to process per second")
    parser.add_argument("--scan-after", type=float, default=SCAN_AFTER_S, help="seconds without a face before scanning")
    parser.add_argument("--turn-step-deg", type=float, default=30.0, help="scan turn step in degrees")
    parser.add_argument("--aim-tolerance-deg", type=float, default=8.0, help="face bearing ignored below degrees")
    parser.add_argument("--head-deg", type=float, default=20.0, help="head tilt in degrees")
    parser.add_argument("--color", action="store_true", help="record in color (default: grayscale)")
    parser.add_argument("--no-move", action="store_true", help="record only, do not turn toward faces")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir) if args.output_dir else build_output_dir("recordings")
    frames_dir = out_dir / "frames"
    log_path = out_dir / "frames.jsonl"
    out_dir.mkdir(parents=True, exist_ok=True)
    print("Recording to {}".format(out_dir))

    turn_step = math.radians(args.turn_step_deg)
    aim_tolerance = math.radians(args.aim_tolerance_deg)
    min_interval = 1.0 / max(args.fps, 0.1)

    with pycozmo.connect() as cli:
        calibration = cli.read_camera_calibration()
        cli.set_lift_height(pycozmo.robot.MIN_LIFT_HEIGHT.mm)
        cli.set_head_angle(pycozmo.util.Angle(degrees=args.head_deg).radians)
        time.sleep(1.0)
        faces_ok = cli.faces.available()
        if not faces_ok:
            print("Face tracking unavailable (need opencv + models): recording only.")
        cli.enable_camera(True, color=args.color)

        images: queue.Queue = queue.Queue(maxsize=2)
        stats = {"frames": 0, "appeared": 0, "identified": 0, "disappeared": 0}

        def on_image(cli: Any, image: Image.Image) -> None:
            try:
                images.put_nowait((image.copy(), time.time(), float(cli.head_angle.radians)))
            except queue.Full:
                try:
                    images.get_nowait()
                except queue.Empty:
                    pass
                try:
                    images.put_nowait((image.copy(), time.time(), float(cli.head_angle.radians)))
                except queue.Full:
                    pass

        def on_appeared(cli: Any, face: Any) -> None:
            stats["appeared"] += 1
            print("A face appeared: {}".format(face))

        def on_identified(cli: Any, face: Any) -> None:
            stats["identified"] += 1
            print("It is {}.".format(face.name))

        def on_disappeared(cli: Any, face: Any) -> None:
            stats["disappeared"] += 1
            print("Gone: {}".format(face))

        cli.add_handler(pycozmo.event.EvtNewRawCameraImage, on_image)
        if faces_ok:
            cli.add_handler(pycozmo.event.EvtFaceAppeared, on_appeared)
            cli.add_handler(pycozmo.event.EvtFaceIdentified, on_identified)
            cli.add_handler(pycozmo.event.EvtFaceDisappeared, on_disappeared)

        start = time.perf_counter()
        last_face_time: Optional[float] = None
        last_scan_time = 0.0
        last_saved = 0.0
        print("Watching. Ctrl-C to stop.")
        try:
            while True:
                now = time.perf_counter()
                if args.duration > 0.0 and (now - start) >= args.duration:
                    break
                try:
                    image, stamp, head_angle = images.get(timeout=0.2)
                except queue.Empty:
                    continue
                if now - last_saved < min_interval:
                    continue
                last_saved = now
                idx = stats["frames"]
                frame_path = frames_dir / "frame_{:06d}.jpg".format(idx)
                save_frame(image, frame_path)

                visible: List[Any] = []
                if faces_ok and not cli.robot_moving:
                    try:
                        cli.faces.process(image, calibration, head_angle, cli.pose_pitch.radians, now)
                    except Exception as e:
                        print("Face processing failed: {}".format(e))
                    visible = cli.faces.visible(now)
                    if visible:
                        last_face_time = now

                record = {
                    "frame": idx,
                    "file": frame_path.name,
                    "t": stamp,
                    "faces": [
                        {"id": f.face_id, "name": f.name, "distance": f.distance,
                         "pose": [f.pose.x, f.pose.y, f.pose.z] if f.pose else None}
                        for f in visible
                    ],
                    "pose": [cli.pose.position.x, cli.pose.position.y,
                             cli.pose.rotation.angle_z.radians],
                }
                append_record(log_path, record)
                stats["frames"] += 1

                if args.no_move or not faces_ok or cli.robot_moving:
                    continue
                if visible:
                    face = visible[0]
                    if face.pose is None:
                        continue
                    bearing = bearing_to_face(cli.pose.position.x, cli.pose.position.y,
                                              cli.pose.rotation.angle_z.radians,
                                              face.pose.x, face.pose.y)
                    if abs(bearing) > aim_tolerance:
                        cli.turn_in_place(pycozmo.util.Angle(radians=bearing), wait=False)
                        time.sleep(SETTLE_S)
                elif should_scan(now, last_face_time, args.scan_after):
                    if now - last_scan_time >= args.scan_after:
                        last_scan_time = now
                        cli.turn_in_place(pycozmo.util.Angle(radians=turn_step), wait=False)
                        time.sleep(SETTLE_S)
        except KeyboardInterrupt:
            pass
        finally:
            try:
                cli.enable_camera(False)
            except Exception:
                pass

    print("Saved {} frames to {}".format(stats["frames"], frames_dir))
    print("Faces: {} appeared, {} identified, {} disappeared.".format(
        stats["appeared"], stats["identified"], stats["disappeared"]))
    video = assemble_video(frames_dir, out_dir / "video.mp4", fps=args.fps)
    if video is not None:
        print("Video: {}".format(video))
    else:
        print("No video assembled (no frames or no writer); JPEGs + frames.jsonl kept.")


if __name__ == "__main__":
    main()

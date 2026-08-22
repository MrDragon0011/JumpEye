#!/usr/bin/env python3
"""Count jumps using your webcam and a YOLO pose model.

Tracks the vertical position of the hips (midpoint of COCO keypoints
11 "left_hip" and 12 "right_hip") smoothed over time, and detects a jump
as a rise-then-fall cycle that clears a minimum amplitude threshold.

Controls:
  q - quit
  r - reset counter
  s - toggle skeleton / bounding-box display
"""

import argparse
import time
from collections import deque

import cv2
from ultralytics import YOLO

# COCO pose keypoint indices
LEFT_HIP, RIGHT_HIP = 11, 12
NOSE = 0

STATE_GROUND = "ground"
STATE_AIRBORNE = "airborne"


def parse_args():
    p = argparse.ArgumentParser(description="Jump counter using YOLO pose")
    p.add_argument("--camera", type=int, default=0, help="camera index")
    p.add_argument(
        "--model",
        default="yolo11n-pose.pt",
        help="YOLO pose model (auto-downloads on first run)",
    )
    p.add_argument(
        "--display",
        choices=["skeleton", "box"],
        default="skeleton",
        help="what to draw: full pose skeleton or just a bounding box",
    )
    p.add_argument(
        "--sensitivity",
        type=float,
        default=0.04,
        help="min hip-rise as a fraction of frame height to count as a jump "
        "(lower = more sensitive)",
    )
    p.add_argument("--conf", type=float, default=0.5, help="detection confidence threshold")
    return p.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.model)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera index {args.camera}")

    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

    jump_count = 0
    state = STATE_GROUND
    baseline_y = None  # smoothed "standing" hip height (pixels, lower = higher on screen -> wait, y grows downward)
    min_y_this_rep = None  # smallest y seen while airborne (highest point reached)
    history = deque(maxlen=5)  # smoothing window of hip y

    show_mode = args.display
    threshold_px = args.sensitivity * frame_h

    # hysteresis band around baseline to avoid jitter false-triggers
    noise_band = 0.01 * frame_h

    prev_time = time.time()
    fps = 0.0

    print("Press 'q' to quit, 'r' to reset, 's' to toggle skeleton/box view.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model.track(frame, persist=True, verbose=False, conf=args.conf)
        result = results[0]

        annotated = frame.copy()
        hip_y = None

        if result.keypoints is not None and len(result.keypoints.xy) > 0:
            # use the first / most confident detected person
            kpts = result.keypoints.xy[0].cpu().numpy()
            kconf = (
                result.keypoints.conf[0].cpu().numpy()
                if result.keypoints.conf is not None
                else None
            )

            def visible(idx):
                if kconf is None:
                    return kpts[idx][0] > 0 or kpts[idx][1] > 0
                return kconf[idx] > 0.3

            if visible(LEFT_HIP) and visible(RIGHT_HIP):
                hip_y = (kpts[LEFT_HIP][1] + kpts[RIGHT_HIP][1]) / 2.0
            elif visible(NOSE):
                # fallback: track nose if hips aren't visible
                hip_y = kpts[NOSE][1]

            if show_mode == "skeleton":
                annotated = result.plot(img=annotated, boxes=False)
            else:
                if result.boxes is not None and len(result.boxes) > 0:
                    x1, y1, x2, y2 = result.boxes.xyxy[0].cpu().numpy().astype(int)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if hip_y is not None:
            history.append(hip_y)
            smoothed_y = sum(history) / len(history)

            if baseline_y is None:
                baseline_y = smoothed_y

            if state == STATE_GROUND:
                # slowly drift baseline toward standing position when grounded
                baseline_y = 0.9 * baseline_y + 0.1 * smoothed_y
                if smoothed_y < baseline_y - threshold_px:
                    state = STATE_AIRBORNE
                    min_y_this_rep = smoothed_y
            else:  # airborne
                min_y_this_rep = min(min_y_this_rep, smoothed_y)
                # landed: back near baseline within noise band
                if smoothed_y > baseline_y - noise_band:
                    peak_height = baseline_y - min_y_this_rep
                    if peak_height >= threshold_px:
                        jump_count += 1
                    state = STATE_GROUND
        else:
            history.clear()

        now = time.time()
        dt = now - prev_time
        prev_time = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)

        cv2.putText(
            annotated,
            f"Jumps: {jump_count}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.4,
            (0, 255, 255),
            3,
        )
        cv2.putText(
            annotated,
            f"state: {state}  fps: {fps:.0f}  view: {show_mode}",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        cv2.imshow("Jump Counter", annotated)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            jump_count = 0
            state = STATE_GROUND
            baseline_y = None
            history.clear()
        elif key == ord("s"):
            show_mode = "box" if show_mode == "skeleton" else "skeleton"

    cap.release()
    cv2.destroyAllWindows()
    print(f"Final jump count: {jump_count}")


if __name__ == "__main__":
    main()

import cv2
import gradio as gr
import numpy as np
from ultralytics import YOLO

LEFT_HIP, RIGHT_HIP, NOSE = 11, 12, 0
HISTORY_LEN = 5

# Runs on CPU deliberately: per-frame @spaces.GPU calls queue through
# ZeroGPU's shared node scheduler (~1s+ dispatch overhead each), which makes
# GPU-backed frame-by-frame video slower and choppier than plain CPU
# inference for a small pose model like this one.
model = YOLO("yolo11n-pose.pt")


def fresh_state():
    return {
        "count": 0,
        "phase": "ground",
        "baseline_y": None,
        "min_y": None,
        "history": [],
        "goal_hit": False,
    }


INFER_SIZE = 256


def run_pose(frame):
    results = model(frame, verbose=False, conf=0.5, imgsz=INFER_SIZE)
    return results[0]


def process_frame(frame, goal, display_mode, state):
    if frame is None:
        return None, state, "0", gr.update(visible=False)

    if state is None:
        state = fresh_state()

    goal = int(goal) if goal else 0
    result = run_pose(frame)
    annotated = frame

    hip_y = None
    if result.keypoints is not None and len(result.keypoints.xy) > 0:
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
            hip_y = kpts[NOSE][1]

        if display_mode == "Skeleton":
            annotated = result.plot(
                img=frame.copy(), boxes=False, labels=False, conf=False, kpt_radius=4
            )
        else:
            annotated = frame.copy()
            if result.boxes is not None and len(result.boxes) > 0:
                x1, y1, x2, y2 = result.boxes.xyxy[0].numpy().astype(int)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)

    frame_h = frame.shape[0]
    threshold_px = 0.045 * frame_h
    noise_band = 0.01 * frame_h

    if hip_y is not None:
        history = state["history"]
        history.append(hip_y)
        if len(history) > HISTORY_LEN:
            history.pop(0)
        smoothed_y = sum(history) / len(history)

        if state["baseline_y"] is None:
            state["baseline_y"] = smoothed_y

        if state["phase"] == "ground":
            state["baseline_y"] = 0.9 * state["baseline_y"] + 0.1 * smoothed_y
            if smoothed_y < state["baseline_y"] - threshold_px:
                state["phase"] = "airborne"
                state["min_y"] = smoothed_y
        else:
            state["min_y"] = min(state["min_y"], smoothed_y)
            if smoothed_y > state["baseline_y"] - noise_band:
                peak_height = state["baseline_y"] - state["min_y"]
                if peak_height >= threshold_px:
                    state["count"] += 1
                state["phase"] = "ground"
    else:
        state["history"] = []

    goal_hit = goal > 0 and state["count"] >= goal
    state["goal_hit"] = goal_hit

    return annotated, state, str(state["count"]), gr.update(visible=goal_hit)


def reset_state():
    return fresh_state(), "0", gr.update(visible=False)


CHECKMARK_SVG = """
<div style="display:flex;align-items:center;justify-content:center;height:100%;">
  <svg width="160" height="160" viewBox="0 0 100 100">
    <circle cx="50" cy="50" r="42" stroke="#4ade80" stroke-width="4" fill="none" />
    <polyline points="30,52 44,66 72,34" stroke="#4ade80" stroke-width="6"
      fill="none" stroke-linecap="round" stroke-linejoin="round" />
  </svg>
</div>
"""

CSS = """
#count_display { font-size: 3rem; font-weight: 700; text-align: center; }
#checkmark_box { min-height: 160px; }
"""

with gr.Blocks(title="JumpEye", css=CSS) as demo:
    gr.Markdown("# JumpEye\nJump. Get counted. Hit your goal.")

    jump_state = gr.State(value=None)

    with gr.Row():
        with gr.Column(scale=2):
            cam = gr.Image(sources=["webcam"], streaming=True, label="Camera")
        with gr.Column(scale=1):
            count_display = gr.Textbox(value="0", label="Jumps", elem_id="count_display", interactive=False)
            goal_input = gr.Number(value=20, label="Goal", precision=0, minimum=1)
            display_mode = gr.Radio(["Skeleton", "Box"], value="Skeleton", label="Display")
            reset_btn = gr.Button("Reset")
            checkmark = gr.HTML(CHECKMARK_SVG, visible=False, elem_id="checkmark_box")

    cam.stream(
        fn=process_frame,
        inputs=[cam, goal_input, display_mode, jump_state],
        outputs=[cam, jump_state, count_display, checkmark],
        show_progress="hidden",
        stream_every=0.2,
        concurrency_limit=1,
    )

    reset_btn.click(
        fn=reset_state,
        outputs=[jump_state, count_display, checkmark],
    )

if __name__ == "__main__":
    demo.launch()

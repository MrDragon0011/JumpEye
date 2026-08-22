---
title: JumpEye
emoji: 🤸
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: "5.9.1"
app_file: app.py
pinned: false
python_version: "3.12"
---

# JumpEye

Jump-counting webcam app powered by YOLO pose detection, running on CPU.
Set a goal, jump, and get a checkmark when you hit it.

ZeroGPU was tried first but its per-call scheduling overhead (~1s+ per
`@spaces.GPU` call, shared across all Spaces on the node) made frame-by-frame
video slower and choppier than plain CPU inference for a model this small.
CPU has no queueing hop, so it processes each frame immediately.

## Local dev

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

## Deploy

1. Create a new Space on huggingface.co: SDK = Gradio, Hardware = CPU basic
   (no GPU needed).
2. Push this folder's contents to the Space's git repo (or use the `hf` CLI /
   web upload).
3. The `yolo11n-pose.pt` weights auto-download on first run.

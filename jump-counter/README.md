# Jump Counter

Counts jumps using your webcam and a YOLO pose model, drawing either the
full pose skeleton or a bounding box over you.

## Setup

```bash
cd jump-counter
python3.10 -m venv .venv        # ultralytics needs <=3.12; 3.10 used here
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python jump_counter.py                    # skeleton view (default)
python jump_counter.py --display box      # bounding box view
```

Press `s` while running to toggle skeleton/box, `r` to reset the count,
`q` to quit.

The first run downloads the small `yolo11n-pose.pt` model automatically.

If it's over- or under-counting, tune `--sensitivity` (fraction of frame
height the hips must rise to register as a jump; default `0.04` — lower
is more sensitive).

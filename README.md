# SuiteMoCa Viewer

A lightweight desktop viewer for synchronized AVI/TIF playback with ROI trace inspection and motion artifact detection.

## Features
- Paired session loading from JSON config
- Side-by-side behavior video (AVI) and 2-photon imaging (TIF) frames
- ROI border overlay on TIF
- ROI fluorescence trace with cursor sync
- Motion energy computation with threshold, smoothing, and min-duration filtering
- Artifact regions overlaid on both motion plot and fluorescence trace
- Time pins and pin-time copy

## Requirements
- Python 3.10+
- numpy
- opencv-python
- tifffile
- pyqtgraph
- PyQt6 (or PyQt5 / PySide6)

## Install
```bash
pip install numpy opencv-python tifffile pyqtgraph PyQt6
```

## Run (GUI)
```bash
python viewer.py --root "<DATA_ROOT>" --config "pairs.example.json"
```

You can also load or reload a JSON config from the menu:
- `File > Open JSON Config...`
- `File > Reload JSON Config`

## CLI / Scripting (no display required)

`viewer_core.py` contains the motion energy functions with **no Qt/pyqtgraph dependency**.
Use it directly in analysis scripts without a display:

```python
from viewer_core import compute_motion_energy, moving_average, enforce_min_duration
import numpy as np

# Compute motion energy from an AVI file
energies, times = compute_motion_energy("path/to/video.avi")

# Smooth (e.g. 1-second window at 30 fps → win=30)
y_smooth = moving_average(energies, win=30)

# Threshold + remove short bursts (< 1s)
above = y_smooth >= 1.4
above_filtered = enforce_min_duration(above, dt=1/30, min_s=1.0)

# Quiet segments: ~above_filtered
```

## JSON config format
`pairs.example.json` shows the expected schema.

Required per pair:
- `id`
- `cell`
- `state`
- `variant`
- `avi`
- `tif`
- `suite2p_dir`

Optional per pair:
- `session_index` (omit to auto-infer from TIF filename)
- `tif_fps` (omit to use CLI default)

Notes:
- Paths may be absolute or relative to `base_root`.
- A pair is skipped if any required file is missing.

## Motion artifact controls (GUI)

| Control | Description |
|---------|-------------|
| `Motion threshold` | Energy level above which a frame is flagged as artifact |
| `Smooth (s)` | Moving-average window in seconds applied before thresholding |
| `Min duration (s)` | Minimum duration of a flagged region; shorter bursts are ignored |

Pipeline:
1. Frame-to-frame pixel difference → raw motion energy
2. Moving-average smoothing
3. Threshold
4. Remove runs shorter than min duration
5. Overlay artifact regions on motion plot and fluorescence trace

## File structure

| File | Description |
|------|-------------|
| `viewer.py` | GUI entry point (requires Qt + pyqtgraph) |
| `viewer_core.py` | Core motion energy functions — no Qt dependency, CLI-safe |
| `pairs.example.json` | Example JSON config |
| `requirements.txt` | Python dependencies |

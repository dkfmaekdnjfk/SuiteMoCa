# SuiteMoCa Viewer

A lightweight desktop viewer for synchronized AVI/TIF playback with ROI trace inspection.

## Features
- Paired session loading from JSON config
- Side-by-side video and imaging frames
- ROI border overlay on TIF
- ROI fluorescence trace with cursor sync
- Motion energy view with threshold, smoothing, and minimum-duration filtering
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

## Run
```bash
python viewer.py --root "<DATA_ROOT>" --config "pairs.example.json"
```

## JSON format
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
- `session_index`
- `tif_fps`

Notes:
- Paths may be absolute or relative to `base_root`.
- A pair is loaded only when required files are present.

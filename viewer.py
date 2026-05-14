import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pyqtgraph as pg
import tifffile

from viewer_core import compute_motion_energy, moving_average, enforce_min_duration

pg.setConfigOptions(imageAxisOrder="row-major")

try:
    from PyQt5 import QtCore, QtGui, QtWidgets
except ImportError:
    try:
        from PyQt6 import QtCore, QtGui, QtWidgets
    except ImportError:
        from PySide6 import QtCore, QtGui, QtWidgets


DEFAULT_ROOT = Path.cwd()
DEFAULT_TIF_FPS = 2.55784

try:
    HORIZONTAL = QtCore.Qt.Orientation.Horizontal
except AttributeError:
    HORIZONTAL = QtCore.Qt.Horizontal


def parse_image_number(path: Path) -> int | None:
    m = re.search(r"_Image(\d+)", path.stem, re.IGNORECASE)
    return int(m.group(1)) if m else None


def parse_tif_date(path: Path) -> str | None:
    m = re.search(r"_(\d{6})_(?:Image\d+|spon\d*)", path.stem)
    return m.group(1) if m else None


def parse_avi_date(path: Path) -> str | None:
    m = re.search(r"\((\d{8})_\d+\)", path.stem)
    if not m:
        return None
    return m.group(1)[2:8]


def parse_state(path: Path) -> str:
    s = path.stem.lower()
    if "cfa" in s:
        return "cfa"
    if "saline" in s:
        return "saline"
    return "normal"


def parse_image_label_to_num(v: str):
    s = str(v).strip().lower()
    if not s:
        return None
    if s.startswith("image"):
        m = re.search(r"image\s*(\d+)", s, re.IGNORECASE)
        return int(m.group(1)) if m else None
    if s in {"spon", "spon2"}:
        return s
    return None


def infer_avi_variant(path: Path):
    s = path.stem.lower()
    if "spon2" in s:
        return "spon2"
    if "spon" in s:
        return "spon"
    m = re.search(r"(?:^|_)cfa_(\d+)(?:_|$)", s)
    if m:
        return f"Image{int(m.group(1))}"
    m = re.search(r"(?:^|_)(\d+)(?:_|$)", s)
    if m:
        return f"Image{int(m.group(1))}"
    return None


def infer_tif_variant(path: Path):
    s = path.stem.lower()
    if "spon2" in s:
        return "spon2"
    if "spon" in s:
        return "spon"
    m = re.search(r"_image(\d+)", s)
    if m:
        return f"Image{int(m.group(1))}"
    return None


def scan_dataset(root: Path) -> dict[str, dict]:
    dataset: dict[str, dict] = {}
    motion_root = root / "motion video"
    if not motion_root.exists():
        return dataset

    for cell_dir in sorted([d for d in root.iterdir() if d.is_dir() and d.name not in {"motion video", "suite2p parameter"}]):
        suite2p = cell_dir / "suite2p" / "plane0"
        required = [suite2p / "F.npy", suite2p / "stat.npy", suite2p / "ops.npy"]
        if not all(p.exists() for p in required):
            continue

        tifs = sorted(cell_dir.glob("*.tif"))
        avis = sorted((motion_root / cell_dir.name).glob("*.avi")) if (motion_root / cell_dir.name).exists() else []
        if not tifs or not avis:
            continue

        avi_entries = []
        for a in avis:
            avi_entries.append({"path": a, "state": parse_state(a), "date": parse_avi_date(a)})

        dataset[cell_dir.name] = {
            "cell": cell_dir.name,
            "suite2p": suite2p,
            "tifs": tifs,
            "avis": avi_entries,
        }

    return dataset


class SyncedViewer(QtWidgets.QMainWindow):
    def __init__(self, root: Path, tif_fps: float, config_path: Path | None = None) -> None:
        super().__init__()
        self.setWindowTitle("AVI-TIF Synced Viewer")
        self.root = root
        self.tif_fps = float(tif_fps)
        self.config_path = config_path

        self.strict_pairs = []

        self.cap = None
        self.F = None
        self.stat = None
        self.ops = None
        self.tif_stack = None

        self.roi_mask_cache: dict[int, np.ndarray] = {}
        self.roi_border_cache: dict[int, np.ndarray] = {}

        self.avi_offset_s = 0.0
        self.pins = []
        self.pin_lines = []
        self.trace_y_range = None
        self.duration_s = 1.0
        self.time_to_avi = 30.0
        self.time_to_tif = self.tif_fps

        # Motion energy
        self._current_avi_path: Path | None = None
        self.motion_energy: np.ndarray | None = None
        self.motion_energy_times: np.ndarray | None = None
        self._trace_artifact_bg: pg.ImageItem | None = None

        self._build_ui()
        self._build_menu()
        self._refresh_pairs_from_source()
        self._init_selectors()
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus if hasattr(QtCore.Qt, "FocusPolicy") else QtCore.Qt.StrongFocus)

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)

        top_bar = QtWidgets.QHBoxLayout()
        root.addLayout(top_bar)

        top_bar.addWidget(QtWidgets.QLabel("Cell"))
        self.cell_combo = QtWidgets.QComboBox()
        self.cell_combo.currentIndexChanged.connect(self._on_cell_changed)
        top_bar.addWidget(self.cell_combo)

        top_bar.addWidget(QtWidgets.QLabel("State"))
        self.state_combo = QtWidgets.QComboBox()
        self.state_combo.currentIndexChanged.connect(self._on_state_changed)
        top_bar.addWidget(self.state_combo)

        top_bar.addWidget(QtWidgets.QLabel("AVI"))
        self.avi_combo = QtWidgets.QComboBox()
        self.avi_combo.currentIndexChanged.connect(self._on_avi_changed)
        top_bar.addWidget(self.avi_combo)

        top_bar.addWidget(QtWidgets.QLabel("TIF"))
        self.tif_combo = QtWidgets.QComboBox()
        self.tif_combo.currentIndexChanged.connect(self._on_tif_changed)
        top_bar.addWidget(self.tif_combo)

        self.load_btn = QtWidgets.QPushButton("Load Session")
        self.load_btn.clicked.connect(self._load_current_selection)
        top_bar.addWidget(self.load_btn)

        self.motion_btn = QtWidgets.QPushButton("Compute Motion")
        self.motion_btn.clicked.connect(self._on_compute_motion_clicked)
        self.motion_btn.setEnabled(False)
        top_bar.addWidget(self.motion_btn)

        self.play_btn = QtWidgets.QPushButton("Play")
        self.play_btn.clicked.connect(self._toggle_play)
        top_bar.addWidget(self.play_btn)

        self.time_label = QtWidgets.QLabel("t=0.000s")
        top_bar.addWidget(self.time_label)

        top_bar.addWidget(QtWidgets.QLabel("ROI"))
        self.roi_combo = QtWidgets.QComboBox()
        self.roi_combo.currentIndexChanged.connect(self._on_roi_changed)
        top_bar.addWidget(self.roi_combo)

        self.roi_input = QtWidgets.QSpinBox()
        self.roi_input.setMinimum(0)
        self.roi_input.valueChanged.connect(self._on_roi_input_changed)
        top_bar.addWidget(self.roi_input)

        self.enhance_check = QtWidgets.QCheckBox("Suite2p-like enhance")
        self.enhance_check.setChecked(True)
        self.enhance_check.stateChanged.connect(self._on_enhance_changed)
        top_bar.addWidget(self.enhance_check)
        top_bar.addStretch(1)

        img_layout = QtWidgets.QHBoxLayout()
        root.addLayout(img_layout, stretch=3)

        avi_col = QtWidgets.QVBoxLayout()
        avi_col.addWidget(QtWidgets.QLabel("Left: Behavior AVI"))
        self.avi_view = pg.ImageView(view=pg.PlotItem())
        self.avi_view.ui.roiBtn.hide()
        self.avi_view.ui.menuBtn.hide()
        self.avi_view.ui.histogram.hide()
        self.avi_view.getView().setAspectLocked(True)
        avi_col.addWidget(self.avi_view, stretch=1)
        img_layout.addLayout(avi_col, stretch=1)

        tif_col = QtWidgets.QVBoxLayout()
        tif_col.addWidget(QtWidgets.QLabel("Right: 2P TIF (+ROI border)"))
        self.tif_view = pg.ImageView(view=pg.PlotItem())
        self.tif_view.ui.roiBtn.hide()
        self.tif_view.ui.menuBtn.hide()
        self.tif_view.ui.histogram.hide()
        self.tif_view.getView().setAspectLocked(True)
        tif_col.addWidget(self.tif_view, stretch=1)
        img_layout.addLayout(tif_col, stretch=1)

        # Fluorescence trace
        self.trace_plot = pg.PlotWidget(title="ROI Fluorescence")
        self.trace_plot.showGrid(x=True, y=True, alpha=0.2)
        self.trace_plot.setLabel("bottom", "Time", units="s")
        self.trace_plot.setLabel("left", "F")
        root.addWidget(self.trace_plot, stretch=2)

        # Motion energy threshold row
        thresh_row = QtWidgets.QHBoxLayout()
        root.addLayout(thresh_row)
        thresh_row.addWidget(QtWidgets.QLabel("Motion threshold:"))
        self.thresh_spin = QtWidgets.QDoubleSpinBox()
        self.thresh_spin.setRange(0.1, 500.0)
        self.thresh_spin.setSingleStep(1.0)
        self.thresh_spin.setValue(5.0)
        self.thresh_spin.setDecimals(1)
        self.thresh_spin.valueChanged.connect(self._on_threshold_changed)
        thresh_row.addWidget(self.thresh_spin)
        thresh_row.addWidget(QtWidgets.QLabel("Smooth (s):"))
        self.smooth_spin = QtWidgets.QDoubleSpinBox()
        self.smooth_spin.setRange(0.0, 5.0)
        self.smooth_spin.setSingleStep(0.1)
        self.smooth_spin.setValue(0.5)
        self.smooth_spin.setDecimals(2)
        self.smooth_spin.valueChanged.connect(self._on_threshold_changed)
        thresh_row.addWidget(self.smooth_spin)
        thresh_row.addWidget(QtWidgets.QLabel("Min duration (s):"))
        self.min_dur_spin = QtWidgets.QDoubleSpinBox()
        self.min_dur_spin.setRange(0.0, 5.0)
        self.min_dur_spin.setSingleStep(0.1)
        self.min_dur_spin.setValue(0.3)
        self.min_dur_spin.setDecimals(2)
        self.min_dur_spin.valueChanged.connect(self._on_threshold_changed)
        thresh_row.addWidget(self.min_dur_spin)
        self.motion_status_label = QtWidgets.QLabel("(load session to compute)")
        thresh_row.addWidget(self.motion_status_label)
        thresh_row.addStretch(1)

        # Motion energy plot (x-axis linked to trace_plot)
        self.motion_plot = pg.PlotWidget(title="Motion Energy  [green=OK / red=above threshold]")
        self.motion_plot.showGrid(x=True, y=True, alpha=0.2)
        self.motion_plot.setLabel("bottom", "Time (TIF)", units="s")
        self.motion_plot.setLabel("left", "Mean |?I|")
        self.motion_plot.setXLink(self.trace_plot)
        root.addWidget(self.motion_plot, stretch=1)

        # Time slider
        slider_row = QtWidgets.QHBoxLayout()
        root.addLayout(slider_row)
        slider_row.addWidget(QtWidgets.QLabel("Time (s)"))
        self.slider = QtWidgets.QSlider(HORIZONTAL)
        self.slider.setMinimum(0)
        self.slider.setMaximum(1)
        self.slider.valueChanged.connect(self._on_slider_changed)
        slider_row.addWidget(self.slider)

        # AVI offset slider
        offset_row = QtWidgets.QHBoxLayout()
        root.addLayout(offset_row)
        self.offset_label = QtWidgets.QLabel("AVI offset: +0.000 s")
        offset_row.addWidget(self.offset_label)
        self.offset_slider = QtWidgets.QSlider(HORIZONTAL)
        self.offset_slider.setMinimum(-5000)
        self.offset_slider.setMaximum(5000)
        self.offset_slider.setValue(0)
        self.offset_slider.valueChanged.connect(self._on_offset_changed)
        offset_row.addWidget(self.offset_slider)

        self.play_timer = QtCore.QTimer(self)
        self.play_timer.timeout.connect(self._on_play_tick)

        # Pin controls
        pin_row = QtWidgets.QHBoxLayout()
        root.addLayout(pin_row)
        self.pin_add_btn = QtWidgets.QPushButton("Add Pin (P)")
        self.pin_add_btn.clicked.connect(self._add_pin)
        pin_row.addWidget(self.pin_add_btn)
        self.pin_del_btn = QtWidgets.QPushButton("Delete Pin")
        self.pin_del_btn.clicked.connect(self._delete_selected_pin)
        pin_row.addWidget(self.pin_del_btn)
        self.pin_clear_btn = QtWidgets.QPushButton("Clear Pins")
        self.pin_clear_btn.clicked.connect(self._clear_pins)
        pin_row.addWidget(self.pin_clear_btn)
        self.pin_copy_btn = QtWidgets.QPushButton("Copy Pin Times")
        self.pin_copy_btn.clicked.connect(self._copy_pin_times)
        pin_row.addWidget(self.pin_copy_btn)

        self.pin_list = QtWidgets.QListWidget()
        self.pin_list.itemDoubleClicked.connect(self._jump_to_pin)
        root.addWidget(self.pin_list, stretch=1)

    def _build_menu(self) -> None:
        menu = self.menuBar()
        file_menu = menu.addMenu("File")
        open_cfg_action = QtGui.QAction("Open JSON Config...", self)
        open_cfg_action.triggered.connect(self._on_open_json_config)
        file_menu.addAction(open_cfg_action)
        reload_cfg_action = QtGui.QAction("Reload JSON Config", self)
        reload_cfg_action.triggered.connect(self._on_reload_json_config)
        file_menu.addAction(reload_cfg_action)

        view_menu = menu.addMenu("View")

        info_action = QtGui.QAction("Show Session Info", self)
        info_action.triggered.connect(self._show_session_info)
        view_menu.addAction(info_action)

    def _refresh_pairs_from_source(self) -> None:
        if self.config_path is None:
            raise RuntimeError("JSON config is required. Use --config or File > Open JSON Config...")
        self.strict_pairs = self._load_pairs_from_json(self.config_path)
        if not self.strict_pairs:
            raise RuntimeError("No valid AVI-TIF pairs found.")

    def _on_open_json_config(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open JSON Config",
            str(self.root),
            "JSON files (*.json);;All files (*.*)",
        )
        if not path:
            return
        self.config_path = Path(path)
        self._refresh_pairs_from_source()
        self._init_selectors()

    def _on_reload_json_config(self) -> None:
        if self.config_path is None:
            QtWidgets.QMessageBox.information(self, "Reload JSON Config", "No JSON config loaded.")
            return
        self._refresh_pairs_from_source()
        self._init_selectors()

    def _resolve_cfg_path(self, p: str, base_root: Path) -> Path:
        pp = Path(p)
        if pp.is_absolute():
            return pp
        return (base_root / pp).resolve()

    def _load_pairs_from_json(self, path: Path) -> list[dict]:
        with open(path, "r", encoding="utf-8-sig") as f:
            cfg = json.load(f)
        base_root = Path(cfg.get("base_root", str(self.root)))
        pairs = []
        skipped = []
        for item in cfg.get("pairs", []):
            need = ["id", "cell", "state", "variant", "avi", "tif", "suite2p_dir"]
            miss = [k for k in need if k not in item]
            if miss:
                skipped.append(f"{item.get('id', '<no-id>')}: missing fields {miss}")
                continue
            pair = {
                "id": item["id"],
                "cell": str(item["cell"]),
                "state": str(item["state"]).lower(),
                "variant": str(item["variant"]),
                "avi": self._resolve_cfg_path(item["avi"], base_root),
                "tif": self._resolve_cfg_path(item["tif"], base_root),
                "suite2p": self._resolve_cfg_path(item["suite2p_dir"], base_root),
            }
            req_suite = [pair["suite2p"] / "F.npy", pair["suite2p"] / "stat.npy", pair["suite2p"] / "ops.npy"]
            if not pair["avi"].exists() or not pair["tif"].exists() or not all(p.exists() for p in req_suite):
                skipped.append(f"{pair['id']}: missing file(s)")
                continue
            if "session_index" in item:
                pair["session_index"] = int(item["session_index"])
            if "tif_fps" in item:
                pair["tif_fps"] = float(item["tif_fps"])
            pairs.append(pair)
        if skipped:
            print("[JSON] skipped invalid pairs:")
            for s in skipped:
                print("  -", s)
        return pairs

    def _init_selectors(self) -> None:
        cells = sorted(set(p["cell"] for p in self.strict_pairs))
        self.cell_combo.clear()
        for c in cells:
            self.cell_combo.addItem(c)
        if cells:
            self._on_cell_changed()

    def _get_current_cell_data(self):
        return self.cell_combo.currentText()

    def _on_cell_changed(self) -> None:
        d = self._get_current_cell_data()
        if not d:
            return
        states = sorted(set(p["state"] for p in self.strict_pairs if p["cell"] == d))
        self.state_combo.blockSignals(True)
        self.state_combo.clear()
        for s in states:
            self.state_combo.addItem(s)
        self.state_combo.blockSignals(False)
        self._on_state_changed()

    def _on_state_changed(self) -> None:
        d = self._get_current_cell_data()
        if not d:
            return
        state = self.state_combo.currentText()
        entries = [p for p in self.strict_pairs if p["cell"] == d and p["state"] == state]

        self.avi_combo.blockSignals(True)
        self.avi_combo.clear()
        for a in entries:
            self.avi_combo.addItem(f"{a['variant']} | {a['avi'].name}", a)
        self.avi_combo.blockSignals(False)

        self._rebuild_tif_combo()

    def _on_avi_changed(self) -> None:
        self._rebuild_tif_combo()

    def _rebuild_tif_combo(self) -> None:
        selected = self.avi_combo.currentData()
        if not selected:
            return
        self.tif_combo.blockSignals(True)
        self.tif_combo.clear()
        self.tif_combo.addItem(selected["tif"].name, selected["tif"])
        self.tif_combo.blockSignals(False)

    def _on_tif_changed(self) -> None:
        pass

    def _session_index_from_tif(self, tif_path: Path, n_sessions: int) -> int:
        all_tifs = sorted(tif_path.parent.glob("*.tif"))
        for i, t in enumerate(all_tifs):
            if t.name == tif_path.name and i < n_sessions:
                return i
        return 0

    def _load_current_selection(self) -> None:
        avi_entry = self.avi_combo.currentData()
        tif_path = self.tif_combo.currentData()
        if not avi_entry or not tif_path:
            return

        self._load_session(
            avi_path=avi_entry["avi"],
            tif_path=tif_path,
            suite2p_dir=avi_entry["suite2p"],
            session_index_override=avi_entry.get("session_index"),
            tif_fps_override=avi_entry.get("tif_fps"),
        )

    def _load_session(
        self,
        avi_path: Path,
        tif_path: Path,
        suite2p_dir: Path,
        session_index_override: int | None = None,
        tif_fps_override: float | None = None,
    ) -> None:
        if self.cap is not None:
            self.cap.release()

        f_path = suite2p_dir / "F.npy"
        stat_path = suite2p_dir / "stat.npy"
        ops_path = suite2p_dir / "ops.npy"

        self.tif_stack = tifffile.imread(str(tif_path))
        if self.tif_stack.ndim != 3:
            raise ValueError(f"Unexpected TIFF shape: {self.tif_stack.shape}")
        self.tif_n, self.tif_h, self.tif_w = self.tif_stack.shape

        self.F = np.load(f_path)
        self.stat = np.load(stat_path, allow_pickle=True)
        self.ops = np.load(ops_path, allow_pickle=True).item()

        frames_per_file = np.array(self.ops.get("frames_per_file", []), dtype=int)
        if len(frames_per_file) == 0:
            raise ValueError("ops.npy does not include frames_per_file")

        if tif_fps_override is not None:
            self.tif_fps = float(tif_fps_override)

        if session_index_override is None:
            session_index = self._session_index_from_tif(tif_path, len(frames_per_file))
        else:
            session_index = int(session_index_override)
            if session_index < 0 or session_index >= len(frames_per_file):
                raise ValueError(f"session_index out of range: {session_index}")
        self.concat_start = int(frames_per_file[:session_index].sum())
        self.concat_stop = self.concat_start + int(frames_per_file[session_index])

        self.cap = cv2.VideoCapture(str(avi_path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open AVI: {avi_path}")
        self._current_avi_path = avi_path
        self.avi_fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        self.avi_n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.duration_s = min((self.tif_n - 1) / self.tif_fps, (self.avi_n - 1) / max(1e-6, self.avi_fps))
        self.time_to_avi = self.avi_fps
        self.time_to_tif = self.tif_fps

        self.tif_min = np.percentile(self.tif_stack, 1)
        self.tif_max = np.percentile(self.tif_stack, 99.8)
        self.mean_img = self.ops.get("meanImg", None)
        if self.mean_img is not None:
            self.mean_img = np.asarray(self.mean_img, dtype=np.float32)
            if self.mean_img.shape != (self.tif_h, self.tif_w):
                self.mean_img = None

        self.roi_mask_cache.clear()
        self.roi_border_cache.clear()

        self.slider.blockSignals(True)
        self.slider.setMaximum(max(1, int(self.duration_s * 1000)))
        self.slider.setValue(0)
        self.slider.blockSignals(False)

        self._reset_roi_controls()
        # motion energy 珥덇린??(?댁쟾 ?몄뀡 ?곗씠???대━??
        self.motion_energy = None
        self.motion_energy_times = None
        self.motion_plot.clear()
        self.motion_status_label.setText("(click Compute Motion)")
        self.motion_btn.setEnabled(True)

        self._plot_trace()
        self._render_at_time(0.0)

        print("[INFO] loaded:", avi_path.name, "|", tif_path.name)
        print("[INFO] tif shape:", self.tif_stack.shape)
        print("[INFO] F shape:", self.F.shape)
        print("[INFO] session_index:", session_index, "concat:", f"{self.concat_start}:{self.concat_stop}")

    # ?? Motion Energy ?????????????????????????????????????????????????????????

    def _compute_motion_energy(self) -> None:
        """AVI ?꾨젅??媛??덈? 李⑥씠???됯퇏??motion energy濡?怨꾩궛."""
        self.motion_energy = None
        self.motion_energy_times = None

        if self._current_avi_path is None or self.avi_n <= 1:
            self.motion_status_label.setText("(no AVI)")
            return

        print(f"[MOTION] Computing motion energy ({self.avi_n} frames)...", flush=True)
        self.motion_status_label.setText("Computing...")
        QtWidgets.QApplication.processEvents()

        def _progress(i, n):
            if i % 300 == 0:
                QtWidgets.QApplication.processEvents()

        try:
            energies, times = compute_motion_energy(self._current_avi_path, progress_cb=_progress)
        except Exception as exc:
            self.motion_status_label.setText(f"(error: {exc})")
            return

        self.motion_energy = energies
        self.motion_energy_times = times

        e_min, e_max = energies.min(), energies.max()
        print(f"[MOTION] Done. range: {e_min:.2f} – {e_max:.2f}")
        self.motion_status_label.setText(f"range [{e_min:.1f}, {e_max:.1f}]")

    def _make_artifact_bg_item(
        self, x: np.ndarray, above: np.ndarray, y_lo: float, y_hi: float
    ) -> pg.ImageItem:
        """threshold 珥덇낵 ?щ? 諛곗뿴(above)濡??⑥씪 ImageItem??留뚮뱺??
        y異?踰붿쐞???몄텧?먭? 紐낆떆?곸쑝濡??꾨떖??auto-range ?ㅼ뿼??諛⑹?."""
        N = len(x)
        img = np.zeros((1, N, 4), dtype=np.uint8)
        img[0, above] = [220, 60, 60, 70]
        item = pg.ImageItem(img)   # axisOrder???꾩뿭 ?ㅼ젙 ?ъ슜
        dx = float(x[-1] - x[0]) / max(N - 1, 1)
        item.setRect(float(x[0]) - dx / 2, y_lo,
                     float(x[-1] - x[0]) + dx, y_hi - y_lo)
        item.setZValue(-10)
        return item

    def _artifact_console_summary(
        self, x: np.ndarray, y: np.ndarray, above: np.ndarray, threshold: float
    ) -> None:
        """contiguous artifact 援ш컙??肄섏넄??異쒕젰?쒕떎."""
        print(f"[MOTION] Artifact regions (threshold={threshold:.1f}):")
        in_region = False
        start_idx = 0
        found = False
        for i, a in enumerate(above):
            if a and not in_region:
                in_region, start_idx = True, i
            elif not a and in_region:
                in_region = False
                print(f"  t={x[start_idx]:.2f}s ??{x[i-1]:.2f}s  "
                      f"(peak={y[start_idx:i].max():.1f})")
                found = True
        if in_region:
            print(f"  t={x[start_idx]:.2f}s ??{x[-1]:.2f}s  "
                  f"(peak={y[start_idx:].max():.1f})")
            found = True
        if not found:
            print("  none")

    def _moving_average(self, y: np.ndarray, win: int) -> np.ndarray:
        return moving_average(y, win)

    def _enforce_min_duration(self, above: np.ndarray, dt: float, min_s: float = 0.3) -> np.ndarray:
        return enforce_min_duration(above, dt, min_s)

    def _plot_motion_energy(self) -> None:
        """Motion energy PlotWidget瑜?媛깆떊?쒕떎."""
        self.motion_plot.clear()

        if self.motion_energy is None:
            return

        threshold = self.thresh_spin.value()
        x = self.motion_energy_times - self.avi_offset_s  # TIF ?쒓컙
        y_raw = self.motion_energy
        dt = 1.0 / max(1e-6, self.avi_fps)
        win = max(1, int(round(self.smooth_spin.value() / dt)))
        y = self._moving_average(y_raw, win)
        above = y >= threshold
        above = self._enforce_min_duration(above, dt, min_s=float(self.min_dur_spin.value()))

        # threshold 珥덇낵 援ш컙 ???⑥씪 ImageItem (鍮좊쫫)
        # motion_plot y-range: 0 ~ max(y) * 1.1
        y_hi_m = float(y.max()) * 1.1 if y.max() > 0 else 1.0
        self.motion_plot.addItem(self._make_artifact_bg_item(x, above, 0.0, y_hi_m))

        # motion energy 怨≪꽑
        self.motion_plot.plot(x, y, pen=pg.mkPen((255, 165, 0), width=1.2))

        # threshold horizontal line
        dash = (
            QtCore.Qt.PenStyle.DashLine
            if hasattr(QtCore.Qt, "PenStyle")
            else QtCore.Qt.DashLine
        )
        self.motion_plot.addItem(pg.InfiniteLine(
            pos=threshold, angle=0, movable=False,
            pen=pg.mkPen((255, 80, 80, 200), width=1.5, style=dash),
        ))

        # cursor
        self.motion_cursor_line = pg.InfiniteLine(
            pos=0.0, angle=90, movable=False, pen=pg.mkPen("y", width=2)
        )
        self.motion_plot.addItem(self.motion_cursor_line)

        # 肄섏넄 ?붿빟
        self._artifact_console_summary(x, y, above, threshold)

        # trace_plot 諛곌꼍??媛깆떊
        self._add_artifact_regions_to_trace(x, above)

    def _add_artifact_regions_to_trace(
        self,
        x: np.ndarray | None = None,
        above: np.ndarray | None = None,
    ) -> None:
        """trace_plot??artifact 諛곌꼍 ImageItem??異붽?/援먯껜?쒕떎."""
        if self._trace_artifact_bg is not None:
            try:
                self.trace_plot.removeItem(self._trace_artifact_bg)
            except Exception:
                pass
            self._trace_artifact_bg = None

        if self.motion_energy is None:
            return
        if x is None:
            x = self.motion_energy_times - self.avi_offset_s
        if above is None:
            above = self.motion_energy >= self.thresh_spin.value()

        # Keep y-range fixed to trace data; offset updates must not rescale y-axis.
        if self.trace_y_range is None:
            roi = self._current_roi()
            y_data = self.F[roi, self.concat_start:self.concat_stop]
            y_lo, y_hi = float(y_data.min()), float(y_data.max())
        else:
            y_lo, y_hi = self.trace_y_range
        self._trace_artifact_bg = self._make_artifact_bg_item(x, above, y_lo, y_hi)
        self.trace_plot.addItem(self._trace_artifact_bg)
        self.trace_plot.setYRange(y_lo, y_hi, padding=0.05)

    def _on_threshold_changed(self, _) -> None:
        """threshold 蹂寃??????뚮’ 紐⑤몢 媛깆떊."""
        if self.motion_energy is not None:
            self._plot_motion_energy()

    def _on_compute_motion_clicked(self) -> None:
        """Compute Motion 踰꾪듉: motion energy 怨꾩궛 ???뚮’."""
        if self.cap is None:
            return
        self.motion_btn.setEnabled(False)
        self._compute_motion_energy()
        self._plot_motion_energy()
        self.motion_btn.setEnabled(True)

    # ?? Session Info ??????????????????????????????????????????????????????????

    def _show_session_info(self) -> None:
        selected = self.avi_combo.currentData()
        if self.cap is None or self.tif_stack is None or not selected:
            QtWidgets.QMessageBox.information(self, "Session Info", "No session loaded.")
            return

        tif_sec = (self.tif_n - 1) / max(1e-6, self.tif_fps)
        avi_sec = (self.avi_n - 1) / max(1e-6, self.avi_fps)
        diff_sec = avi_sec - tif_sec

        msg = (
            f"Cell: {selected['cell']}\n"
            f"State: {selected['state']}\n"
            f"Variant: {selected['variant']}\n\n"
            f"AVI: {selected['avi'].name}\n"
            f"  frames={self.avi_n}, fps={self.avi_fps:.5f}, duration={avi_sec:.3f}s\n\n"
            f"TIF: {selected['tif'].name}\n"
            f"  frames={self.tif_n}, fps={self.tif_fps:.5f}, duration={tif_sec:.3f}s\n\n"
            f"Duration diff (AVI - TIF): {diff_sec:+.3f}s\n"
            f"Current offset: {self.avi_offset_s:+.3f}s"
        )
        QtWidgets.QMessageBox.information(self, "Session Info", msg)

    # ?? Pins ??????????????????????????????????????????????????????????????????

    def _refresh_pin_list(self) -> None:
        self.pin_list.clear()
        for i, pin in enumerate(self.pins):
            self.pin_list.addItem(
                f"{i+1:02d} | t={pin['t_sec']:.3f}s | tif={pin['tif_idx']} | avi={pin['avi_idx']}"
            )
        self._refresh_pin_lines()

    def _refresh_pin_lines(self) -> None:
        if not hasattr(self, "trace_plot"):
            return
        for ln in self.pin_lines:
            try:
                self.trace_plot.removeItem(ln)
            except Exception:
                pass
        self.pin_lines = []
        for p in self.pins:
            ln = pg.InfiniteLine(
                pos=float(p["t_sec"]),
                angle=90,
                movable=False,
                pen=pg.mkPen((180, 220, 255, 90), width=1),
            )
            self.trace_plot.addItem(ln)
            self.pin_lines.append(ln)

    def _add_pin(self) -> None:
        if self.cap is None or self.tif_stack is None:
            return
        t_sec = self.slider.value() / 1000.0
        tif_idx = int(np.clip(round(t_sec * self.time_to_tif), 0, self.tif_n - 1))
        avi_idx = int(np.clip(round((t_sec + self.avi_offset_s) * self.time_to_avi), 0, self.avi_n - 1))
        self.pins.append({"t_sec": t_sec, "tif_idx": tif_idx, "avi_idx": avi_idx})
        self._refresh_pin_list()

    def _delete_selected_pin(self) -> None:
        row = self.pin_list.currentRow()
        if row < 0 or row >= len(self.pins):
            return
        self.pins.pop(row)
        self._refresh_pin_list()

    def _clear_pins(self) -> None:
        self.pins = []
        self._refresh_pin_list()

    def _copy_pin_times(self) -> None:
        if not self.pins:
            QtWidgets.QMessageBox.information(self, "Copy Pin Times", "No pins to copy.")
            return
        text = "\n".join([f"{p['t_sec']:.3f}" for p in self.pins])
        QtWidgets.QApplication.clipboard().setText(text)
        QtWidgets.QMessageBox.information(self, "Copy Pin Times", f"Copied {len(self.pins)} pin times.")

    def _jump_to_pin(self, item) -> None:
        row = self.pin_list.row(item)
        if row < 0 or row >= len(self.pins):
            return
        t_sec = float(self.pins[row]["t_sec"])
        self.slider.setValue(int(round(t_sec * 1000)))

    # ?? ROI Controls ??????????????????????????????????????????????????????????

    def _reset_roi_controls(self) -> None:
        n_rois = int(self.F.shape[0])
        self.roi_combo.blockSignals(True)
        self.roi_combo.clear()
        for i in range(n_rois):
            self.roi_combo.addItem(str(i), i)
        self.roi_combo.setCurrentIndex(0)
        self.roi_combo.blockSignals(False)

        self.roi_input.blockSignals(True)
        self.roi_input.setMaximum(max(0, n_rois - 1))
        self.roi_input.setValue(0)
        self.roi_input.blockSignals(False)

    def _get_roi_mask(self, roi_idx: int) -> np.ndarray:
        if roi_idx in self.roi_mask_cache:
            return self.roi_mask_cache[roi_idx]
        mask = np.zeros((self.tif_h, self.tif_w), dtype=np.uint8)
        roi_stat = self.stat[roi_idx]
        ypix = np.asarray(roi_stat["ypix"], dtype=int)
        xpix = np.asarray(roi_stat["xpix"], dtype=int)
        ypix = np.clip(ypix, 0, self.tif_h - 1)
        xpix = np.clip(xpix, 0, self.tif_w - 1)
        mask[ypix, xpix] = 1
        self.roi_mask_cache[roi_idx] = mask
        return mask

    def _get_roi_border(self, roi_idx: int) -> np.ndarray:
        if roi_idx in self.roi_border_cache:
            return self.roi_border_cache[roi_idx]
        mask = self._get_roi_mask(roi_idx)
        eroded = cv2.erode(mask, np.ones((3, 3), dtype=np.uint8), iterations=1)
        border = np.clip(mask - eroded, 0, 1).astype(np.uint8)
        self.roi_border_cache[roi_idx] = border
        return border

    def _current_roi(self) -> int:
        return int(self.roi_combo.currentData()) if self.roi_combo.currentData() is not None else 0

    def _on_roi_changed(self) -> None:
        roi = self._current_roi()
        self.roi_input.blockSignals(True)
        self.roi_input.setValue(roi)
        self.roi_input.blockSignals(False)
        self._plot_trace()
        self._render_at_time(self.slider.value() / 1000.0)

    def _on_roi_input_changed(self, roi: int) -> None:
        idx = self.roi_combo.findData(int(roi))
        if idx >= 0:
            self.roi_combo.setCurrentIndex(idx)

    # ?? Slider / Offset ???????????????????????????????????????????????????????

    def _on_slider_changed(self, v: int) -> None:
        self._render_at_time(v / 1000.0)

    def _on_offset_changed(self, v: int) -> None:
        self.avi_offset_s = v / 1000.0
        self.offset_label.setText(f"AVI offset: {self.avi_offset_s:+0.3f} s")
        self._render_at_time(self.slider.value() / 1000.0)
        # offset 蹂寃???motion plot 諛?trace spon region 媛깆떊
        if self.motion_energy is not None:
            self._plot_motion_energy()

    def _on_enhance_changed(self) -> None:
        self._render_at_time(self.slider.value() / 1000.0)

    # ?? Playback ??????????????????????????????????????????????????????????????

    def _toggle_play(self) -> None:
        if self.play_timer.isActive():
            self.play_timer.stop()
            self.play_btn.setText("Play")
            return
        self.play_timer.start(33)
        self.play_btn.setText("Pause")

    def _on_play_tick(self) -> None:
        step = 1.0 / max(1e-6, self.avi_fps)
        t = self.slider.value() / 1000.0 + step
        if t >= self.duration_s:
            t = self.duration_s
            self.play_timer.stop()
            self.play_btn.setText("Play")
        self.slider.setValue(int(round(t * 1000)))

    # ?? Plotting ??????????????????????????????????????????????????????????????

    def _plot_trace(self) -> None:
        if self.F is None:
            return
        self.trace_plot.clear()
        self._trace_artifact_bg = None  # clear() 濡??대? ?쒓굅?? 李몄“留?由ъ뀑

        roi = self._current_roi()
        y = self.F[roi, self.concat_start:self.concat_stop]
        n = min(len(y), self.tif_n)
        x = np.arange(n) / self.tif_fps
        y_plot = y[:n]
        self.trace_plot.plot(x, y_plot, pen=pg.mkPen("w", width=1.5))
        y_lo = float(np.min(y_plot))
        y_hi = float(np.max(y_plot))
        if y_hi <= y_lo:
            y_hi = y_lo + 1e-6
        self.trace_y_range = (y_lo, y_hi)
        self.trace_plot.setYRange(y_lo, y_hi, padding=0.05)
        self.trace_plot.enableAutoRange(axis="y", enable=False)
        self.cursor_line = pg.InfiniteLine(pos=0.0, angle=90, movable=False, pen=pg.mkPen("y", width=2))
        self.trace_plot.addItem(self.cursor_line)
        self._refresh_pin_lines()

        # artifact 諛곌꼍 異붽? (motion energy 濡쒕뱶??寃쎌슦)
        self._add_artifact_regions_to_trace()

    def _read_avi_frame(self, idx: int) -> np.ndarray:
        idx = int(np.clip(idx, 0, self.avi_n - 1))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self.cap.read()
        if not ok:
            return np.zeros((494, 660, 3), dtype=np.uint8)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def _render_at_time(self, t_sec: float) -> None:
        if self.cap is None or self.tif_stack is None:
            return
        t_sec = float(np.clip(t_sec, 0.0, self.duration_s))
        tif_idx = int(np.clip(round(t_sec * self.time_to_tif), 0, self.tif_n - 1))
        avi_idx = int(np.clip(round((t_sec + self.avi_offset_s) * self.time_to_avi), 0, self.avi_n - 1))

        avi = self._read_avi_frame(avi_idx)
        self.avi_view.setImage(avi, autoLevels=False, autoRange=False, levels=(0, 255))

        tif = self.tif_stack[tif_idx].astype(np.float32)
        if self.enhance_check.isChecked() and self.mean_img is not None:
            tif = np.maximum(0.0, tif - 0.7 * self.mean_img)
            lo = np.percentile(tif, 1.0)
            hi = np.percentile(tif, 99.7)
        else:
            lo = self.tif_min
            hi = self.tif_max
        tif_norm = np.clip((tif - lo) / max(1e-6, hi - lo), 0, 1)

        border = self._get_roi_border(self._current_roi())
        rgb = np.stack([tif_norm, tif_norm, tif_norm], axis=-1)
        rgb[..., 0] = np.maximum(rgb[..., 0], border * 1.0)
        rgb[..., 1] = np.maximum(rgb[..., 1], border * 0.1)
        rgb[..., 2] = np.maximum(rgb[..., 2], border * 0.1)
        self.tif_view.setImage((rgb * 255).astype(np.uint8), autoLevels=False, autoRange=False, levels=(0, 255))

        if hasattr(self, "cursor_line"):
            self.cursor_line.setValue(t_sec)
        if hasattr(self, "motion_cursor_line") and self.motion_cursor_line is not None:
            self.motion_cursor_line.setValue(t_sec)
        self.time_label.setText(f"t={t_sec:0.3f}s | tif={tif_idx} | avi={avi_idx} | offset={self.avi_offset_s:+0.3f}s")

    # ?? Keyboard ??????????????????????????????????????????????????????????????

    def keyPressEvent(self, event) -> None:
        key = event.key()
        key_left  = QtCore.Qt.Key.Key_Left  if hasattr(QtCore.Qt, "Key") else QtCore.Qt.Key_Left
        key_right = QtCore.Qt.Key.Key_Right if hasattr(QtCore.Qt, "Key") else QtCore.Qt.Key_Right
        key_space = QtCore.Qt.Key.Key_Space if hasattr(QtCore.Qt, "Key") else QtCore.Qt.Key_Space
        key_p     = QtCore.Qt.Key.Key_P     if hasattr(QtCore.Qt, "Key") else QtCore.Qt.Key_P
        key_c     = QtCore.Qt.Key.Key_C     if hasattr(QtCore.Qt, "Key") else QtCore.Qt.Key_C
        mod_ctrl  = (QtCore.Qt.KeyboardModifier.ControlModifier
                     if hasattr(QtCore.Qt, "KeyboardModifier")
                     else QtCore.Qt.ControlModifier)

        if key == key_space:
            self._toggle_play(); event.accept(); return
        if key == key_p:
            self._add_pin(); event.accept(); return
        if key == key_c and (event.modifiers() & mod_ctrl):
            self._copy_pin_times(); event.accept(); return
        if key == key_left:
            self.slider.setValue(max(0, self.slider.value() - 33)); event.accept(); return
        if key == key_right:
            self.slider.setValue(min(self.slider.maximum(), self.slider.value() + 33)); event.accept(); return
        super().keyPressEvent(event)

    def closeEvent(self, event) -> None:
        try:
            if self.cap is not None:
                self.cap.release()
        finally:
            super().closeEvent(event)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AVI-TIF synchronized viewer with dynamic cell/state selection.")
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--config", type=Path, required=True, help="Path to JSON pair config")
    p.add_argument("--tif-fps", type=float, default=DEFAULT_TIF_FPS)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.root.exists():
        raise FileNotFoundError(f"Root not found: {args.root}")

    app = QtWidgets.QApplication(sys.argv)
    win = SyncedViewer(root=args.root, tif_fps=args.tif_fps, config_path=args.config)
    win.resize(1700, 1050)
    win.show()
    if hasattr(app, "exec"):
        sys.exit(app.exec())
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()



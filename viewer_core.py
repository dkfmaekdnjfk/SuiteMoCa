"""viewer_core.py
================
Qt/pyqtgraph에 의존하지 않는 핵심 함수 모음.

GUI 없이 CLI 스크립트에서 직접 import 가능:
    from viewer_core import compute_motion_energy, moving_average, enforce_min_duration
"""

from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Motion Energy
# ---------------------------------------------------------------------------

def compute_motion_energy(
    avi_path: Path,
    progress_cb=None,
) -> tuple[np.ndarray, np.ndarray]:
    """AVI 파일에서 frame-by-frame motion energy를 계산한다.

    Parameters
    ----------
    avi_path : Path
        AVI 파일 경로.
    progress_cb : callable(i, n) | None
        진행 상황 콜백. i=현재 프레임, n=총 프레임 수.
        GUI에서는 QApplication.processEvents() 호출에 사용.

    Returns
    -------
    energies : np.ndarray  shape (n_frames,)
        프레임별 motion energy. energies[t] = mean(|gray[t] - gray[t-1]|).
        energies[0] = 0 (첫 프레임은 이전 프레임 없음).
    times : np.ndarray  shape (n_frames,)
        각 프레임의 시간 (초). times[t] = t / avi_fps.
    """
    cap = cv2.VideoCapture(str(avi_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open AVI: {avi_path}")

    avi_n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    avi_fps = float(cap.get(cv2.CAP_PROP_FPS))

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ok, prev = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError(f"Cannot read first frame: {avi_path}")

    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY).astype(np.float32)
    energies = np.zeros(avi_n, dtype=np.float32)

    for i in range(1, avi_n):
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        energies[i] = np.mean(np.abs(gray - prev_gray))
        prev_gray = gray
        if progress_cb is not None:
            progress_cb(i, avi_n)

    cap.release()
    times = np.arange(avi_n, dtype=np.float32) / max(avi_fps, 1e-6)
    return energies, times


def moving_average(y: np.ndarray, win: int) -> np.ndarray:
    """Rectangular (box) moving average.

    Parameters
    ----------
    y : np.ndarray
    win : int
        윈도우 크기 (프레임 수). win <= 1이면 원본 반환.
    """
    if win <= 1:
        return y
    k = np.ones(win, dtype=np.float32) / float(win)
    return np.convolve(y, k, mode="same")


def enforce_min_duration(
    above: np.ndarray,
    dt: float,
    min_s: float = 0.3,
) -> np.ndarray:
    """above=True인 연속 구간 중 min_s 미만인 것을 False로 되돌린다.

    Parameters
    ----------
    above : np.ndarray  bool
        artifact 여부 배열.
    dt : float
        프레임 간격 (초). = 1 / fps.
    min_s : float
        최소 지속 시간 (초). 이보다 짧은 artifact 구간은 노이즈로 처리.
    """
    out = above.copy()
    n = len(out)
    min_len = max(1, int(round(min_s / max(dt, 1e-6))))
    i = 0
    while i < n:
        if not out[i]:
            i += 1
            continue
        j = i + 1
        while j < n and out[j]:
            j += 1
        if (j - i) < min_len:
            out[i:j] = False
        i = j
    return out

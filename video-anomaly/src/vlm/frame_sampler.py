from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from src.common.utils import ensure_dir
from src.common.video_io import sample_frames_at_timestamps


def uniform_sample_timestamps(
    duration_sec: float, window_sec: int, n_frames: int
) -> List[float]:
    if n_frames <= 0:
        return []

    if duration_sec <= 0:
        return [0.0]

    effective_window = float(min(max(window_sec, 1), max(duration_sec, 1.0)))
    if n_frames == 1:
        return [0.0]

    # Endpoint False avoids sampling the exact end-frame repeatedly on short clips.
    times = np.linspace(0.0, effective_window, num=n_frames, endpoint=False)
    return [round(float(t), 3) for t in times.tolist()]


def sample_and_save_frames(
    video_path: Path | str,
    duration_sec: float,
    window_sec: int,
    n_frames: int,
    output_dir: Path | str,
) -> List[Dict[str, str | float]]:
    timestamps = uniform_sample_timestamps(duration_sec, window_sec, n_frames)
    target_dir = ensure_dir(output_dir)

    for old_file in target_dir.glob("*.jpg"):
        old_file.unlink(missing_ok=True)

    samples = sample_frames_at_timestamps(video_path, timestamps)

    saved: List[Dict[str, str | float]] = []
    for idx, (timestamp_sec, frame) in enumerate(samples):
        frame_name = f"frame_{idx:03d}_{timestamp_sec:08.3f}.jpg"
        frame_path = target_dir / frame_name
        ok = cv2.imwrite(str(frame_path), frame)
        if not ok:
            continue

        saved.append(
            {
                "timestamp_sec": round(float(timestamp_sec), 3),
                "frame_path": str(frame_path.resolve()),
            }
        )

    return saved

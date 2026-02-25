from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


def get_video_metadata(video_path: Path | str) -> Dict[str, float]:
    path = str(Path(video_path).resolve())
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"Cannot open video: {path}")

    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration_sec = (frame_count / fps) if fps > 0 else 0.0

        return {
            "fps": fps,
            "frame_count": frame_count,
            "duration_sec": max(duration_sec, 0.0),
        }
    finally:
        cap.release()


def sample_frames_at_fps(video_path: Path | str, sample_fps: float) -> List[Tuple[float, np.ndarray]]:
    if sample_fps <= 0:
        raise ValueError("sample_fps must be > 0")

    metadata = get_video_metadata(video_path)
    duration_sec = metadata["duration_sec"]
    if duration_sec <= 0:
        return sample_frames_at_timestamps(video_path, [0.0])

    step_sec = 1.0 / sample_fps
    timestamps: List[float] = []
    t = 0.0
    while t < duration_sec:
        timestamps.append(round(t, 3))
        t += step_sec

    if not timestamps:
        timestamps = [0.0]

    return sample_frames_at_timestamps(video_path, timestamps)


def sample_frames_at_timestamps(
    video_path: Path | str, timestamps_sec: List[float]
) -> List[Tuple[float, np.ndarray]]:
    path = str(Path(video_path).resolve())
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"Cannot open video: {path}")

    frames: List[Tuple[float, np.ndarray]] = []
    try:
        for ts in timestamps_sec:
            safe_ts = max(float(ts), 0.0)
            cap.set(cv2.CAP_PROP_POS_MSEC, safe_ts * 1000.0)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            frames.append((safe_ts, frame))
    finally:
        cap.release()

    return frames

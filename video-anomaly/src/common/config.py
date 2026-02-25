from __future__ import annotations

import os
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
RAW_VIDEO_DIR = (DATA_DIR / "raw_videos").resolve()
FRAME_CACHE_DIR = (DATA_DIR / "frames").resolve()
RUNS_DIR = (REPO_ROOT / "runs").resolve()
YOLO_RUN_DIR = (RUNS_DIR / "yolo_events").resolve()
VLM_RUN_DIR = (RUNS_DIR / "vlm_events").resolve()
MERGED_RESULTS_PATH = (RUNS_DIR / "merged_results.csv").resolve()

DEFAULT_VIDEO_SOURCES = [
    "/Users/naeem/Downloads/archive-2",
    "/Users/naeem/Downloads/Dataset",
]

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}


def _parse_video_sources(value: str) -> List[str]:
    if not value.strip():
        return []

    parsed: List[str] = []
    for raw in value.split(":"):
        cleaned = raw.strip()
        if not cleaned:
            continue
        parsed.append(str(Path(cleaned).expanduser().resolve()))
    return parsed


VIDEO_SOURCES = _parse_video_sources(os.getenv("VIDEO_SOURCES", ""))
if not VIDEO_SOURCES:
    VIDEO_SOURCES = [str(Path(path).expanduser().resolve()) for path in DEFAULT_VIDEO_SOURCES]


def get_video_sources() -> List[str]:
    sources = list(VIDEO_SOURCES)
    local_raw = str(RAW_VIDEO_DIR)
    if local_raw not in sources:
        sources.append(local_raw)

    unique_sources: List[str] = []
    seen = set()
    for source in sources:
        normalized = str(Path(source).expanduser().resolve())
        if normalized in seen:
            continue
        seen.add(normalized)
        unique_sources.append(normalized)
    return unique_sources


GLOBAL_SEED = int(os.getenv("GLOBAL_SEED", "42"))

CONF_TH = float(os.getenv("CONF_TH", "0.35"))
EVENT_TH = float(os.getenv("EVENT_TH", "0.15"))
SAMPLE_FPS = float(os.getenv("SAMPLE_FPS", "1.0"))
YOLO_MAX_VIDEOS = int(os.getenv("YOLO_MAX_VIDEOS", "0"))
YOLO_MAX_FRAMES_PER_VIDEO = int(os.getenv("YOLO_MAX_FRAMES_PER_VIDEO", "0"))
YOLO_MAX_SECONDS = int(os.getenv("YOLO_MAX_SECONDS", "0"))
YOLO_SUSPICIOUS_CLASSES = [
    item.strip().lower()
    for item in os.getenv("YOLO_SUSPICIOUS_CLASSES", "").split(",")
    if item.strip()
]
ROBOFLOW_MAX_RETRIES = int(os.getenv("ROBOFLOW_MAX_RETRIES", "2"))
ROBOFLOW_RETRY_BACKOFF_SEC = float(os.getenv("ROBOFLOW_RETRY_BACKOFF_SEC", "0.75"))

WINDOW_SEC = int(os.getenv("WINDOW_SEC", "60"))
N_FRAMES = int(os.getenv("N_FRAMES", "24"))

REQUEST_TIMEOUT_SEC = int(os.getenv("REQUEST_TIMEOUT_SEC", "45"))

ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "").strip()
ROBOFLOW_WORKSPACE = os.getenv("ROBOFLOW_WORKSPACE", "").strip()
ROBOFLOW_PROJECT = os.getenv("ROBOFLOW_PROJECT", "").strip()
ROBOFLOW_VERSION = os.getenv("ROBOFLOW_VERSION", "").strip()
ROBOFLOW_API_HOST = os.getenv("ROBOFLOW_API_HOST", "https://detect.roboflow.com").rstrip("/")

VLM_BASE_URL = os.getenv("VLM_BASE_URL", "http://127.0.0.1:8080").rstrip("/")
VLM_MODEL = os.getenv("VLM_MODEL", "").strip()
VLM_MODE = os.getenv("VLM_MODE", "openai").strip().lower()
VLM_CUSTOM_ENDPOINT = os.getenv("VLM_CUSTOM_ENDPOINT", "/completion").strip()
VLM_HTTP_TIMEOUT_SEC = int(os.getenv("VLM_HTTP_TIMEOUT_SEC", "90"))

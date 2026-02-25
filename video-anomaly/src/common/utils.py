from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from src.common import config


def setup_logger(name: str = "video_anomaly") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logger.setLevel(level)

    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def set_deterministic(seed: int = config.GLOBAL_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: Path | str) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def discover_videos(video_sources: Iterable[str]) -> List[Path]:
    videos: List[Path] = []
    seen_paths = set()

    for source in video_sources:
        source_path = Path(source).expanduser().resolve()
        if not source_path.exists() or not source_path.is_dir():
            continue

        for file_path in source_path.rglob("*"):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in config.VIDEO_EXTENSIONS:
                continue

            abs_path = str(file_path.resolve())
            if abs_path in seen_paths:
                continue
            seen_paths.add(abs_path)
            videos.append(Path(abs_path))

    videos.sort(key=lambda p: str(p))
    return videos


def make_video_id(video_path: Path | str) -> str:
    path = Path(video_path).resolve()
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", path.stem).strip("_") or "video"
    suffix = hashlib.md5(str(path).encode("utf-8")).hexdigest()[:10]
    return f"{stem}_{suffix}"


def safe_json_parse(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None

    cleaned = text.strip()

    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()

    direct = _try_json_loads(cleaned)
    if isinstance(direct, dict):
        return direct

    candidate = _extract_first_json_object(cleaned)
    if not candidate:
        return None

    parsed = _try_json_loads(candidate)
    return parsed if isinstance(parsed, dict) else None


def _try_json_loads(value: str) -> Any:
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return None


def _extract_first_json_object(text: str) -> Optional[str]:
    start = text.find("{")
    while start != -1:
        depth = 0
        for idx in range(start, len(text)):
            char = text[idx]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start : idx + 1]
        start = text.find("{", start + 1)
    return None

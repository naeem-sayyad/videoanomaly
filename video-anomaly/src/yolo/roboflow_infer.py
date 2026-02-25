from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import cv2
import pandas as pd
import requests

from src.common import config
from src.common.utils import (
    discover_videos,
    ensure_dir,
    make_video_id,
    set_deterministic,
    setup_logger,
)
from src.common.video_io import get_video_metadata, sample_frames_at_fps


@dataclass
class RoboflowHostedClient:
    api_key: str
    workspace: str
    project: str
    version: str
    host: str
    timeout_sec: int
    max_retries: int
    retry_backoff_sec: float

    def _candidate_urls(self) -> List[str]:
        urls: List[str] = []
        if self.workspace:
            urls.append(f"{self.host}/{self.workspace}/{self.project}/{self.version}")
        urls.append(f"{self.host}/{self.project}/{self.version}")

        seen = set()
        unique = []
        for url in urls:
            if url in seen:
                continue
            seen.add(url)
            unique.append(url)
        return unique

    def infer_frame(self, frame_bgr, conf_th: float) -> Dict[str, Any]:
        ok, encoded = cv2.imencode(".jpg", frame_bgr)
        if not ok:
            raise ValueError("Failed to encode frame as JPEG")

        payload = encoded.tobytes()
        params = {
            "api_key": self.api_key,
            "confidence": conf_th,
        }
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
        }

        last_error = ""
        for url in self._candidate_urls():
            for attempt in range(self.max_retries + 1):
                try:
                    response = requests.post(
                        url,
                        params=params,
                        data=payload,
                        headers=headers,
                        timeout=self.timeout_sec,
                    )
                except requests.RequestException as exc:
                    last_error = str(exc)
                    if attempt < self.max_retries:
                        delay = self.retry_backoff_sec * (2**attempt)
                        time.sleep(delay)
                        continue
                    break

                if response.status_code == 404:
                    last_error = f"404 at {url}"
                    break

                if response.status_code == 429 or 500 <= response.status_code < 600:
                    last_error = (
                        f"HTTP {response.status_code} at {url}: {response.text[:200]}"
                    )
                    if attempt < self.max_retries:
                        delay = self.retry_backoff_sec * (2**attempt)
                        time.sleep(delay)
                        continue
                    break

                response.raise_for_status()
                return response.json()

        raise RuntimeError(f"Roboflow request failed: {last_error or 'unknown error'}")


def _frame_has_hit(
    predictions: List[Dict[str, Any]], conf_th: float, suspicious_classes: List[str]
) -> bool:
    allowed = {item.lower() for item in suspicious_classes}
    for pred in predictions:
        conf = pred.get("confidence", pred.get("score", 0.0))
        pred_class = str(pred.get("class", pred.get("label", ""))).strip().lower()
        try:
            conf_ok = float(conf) >= conf_th
        except (TypeError, ValueError):
            continue
        if not conf_ok:
            continue
        if allowed and pred_class not in allowed:
            continue
        return True
    return False


def _missing_env_vars() -> List[str]:
    required = {
        "ROBOFLOW_API_KEY": config.ROBOFLOW_API_KEY,
        "ROBOFLOW_PROJECT": config.ROBOFLOW_PROJECT,
        "ROBOFLOW_VERSION": config.ROBOFLOW_VERSION,
    }
    return [name for name, value in required.items() if not value]


def main() -> None:
    logger = setup_logger("yolo.roboflow")
    set_deterministic(config.GLOBAL_SEED)

    ensure_dir(config.YOLO_RUN_DIR)

    columns = [
        "video_name",
        "video_path",
        "duration_sec",
        "sampled_frames",
        "frames_with_hits",
        "event_score",
        "label",
    ]

    missing_env = _missing_env_vars()
    if missing_env:
        logger.error(
            "Missing required environment variables for Roboflow: %s",
            ", ".join(missing_env),
        )
        out_csv = config.YOLO_RUN_DIR / "yolo_summary.csv"
        pd.DataFrame(columns=columns).to_csv(out_csv, index=False)
        logger.info("Wrote empty YOLO summary CSV: %s", out_csv)
        return

    video_sources = config.get_video_sources()
    logger.info("Video sources: %s", video_sources)

    videos = discover_videos(video_sources)
    if config.YOLO_MAX_VIDEOS > 0:
        videos = videos[: config.YOLO_MAX_VIDEOS]
    logger.info("Discovered %d videos", len(videos))
    if config.YOLO_MAX_VIDEOS > 0:
        logger.info("YOLO_MAX_VIDEOS active: %d", config.YOLO_MAX_VIDEOS)
    if config.YOLO_MAX_SECONDS > 0:
        logger.info("YOLO_MAX_SECONDS active: %d", config.YOLO_MAX_SECONDS)
    if config.YOLO_MAX_FRAMES_PER_VIDEO > 0:
        logger.info(
            "YOLO_MAX_FRAMES_PER_VIDEO active: %d",
            config.YOLO_MAX_FRAMES_PER_VIDEO,
        )
    if config.YOLO_SUSPICIOUS_CLASSES:
        logger.info("YOLO_SUSPICIOUS_CLASSES active: %s", config.YOLO_SUSPICIOUS_CLASSES)

    client = RoboflowHostedClient(
        api_key=config.ROBOFLOW_API_KEY,
        workspace=config.ROBOFLOW_WORKSPACE,
        project=config.ROBOFLOW_PROJECT,
        version=config.ROBOFLOW_VERSION,
        host=config.ROBOFLOW_API_HOST,
        timeout_sec=config.REQUEST_TIMEOUT_SEC,
        max_retries=max(config.ROBOFLOW_MAX_RETRIES, 0),
        retry_backoff_sec=max(config.ROBOFLOW_RETRY_BACKOFF_SEC, 0.1),
    )

    rows: List[Dict[str, Any]] = []
    for video_path in videos:
        try:
            metadata = get_video_metadata(video_path)
            duration_sec = float(metadata["duration_sec"])
        except Exception as exc:
            logger.warning("Skipping unreadable video %s (%s)", video_path, exc)
            continue

        try:
            samples = sample_frames_at_fps(video_path, config.SAMPLE_FPS)
        except Exception as exc:
            logger.warning("Skipping video with sampling error %s (%s)", video_path, exc)
            continue

        if config.YOLO_MAX_SECONDS > 0:
            samples = [(ts, frame) for ts, frame in samples if ts <= config.YOLO_MAX_SECONDS]
        if config.YOLO_MAX_FRAMES_PER_VIDEO > 0:
            samples = samples[: config.YOLO_MAX_FRAMES_PER_VIDEO]

        sampled_frames = len(samples)
        frames_with_hits = 0
        per_frame: List[Dict[str, Any]] = []

        for idx, (timestamp_sec, frame) in enumerate(samples):
            try:
                response = client.infer_frame(frame, config.CONF_TH)
                predictions = response.get("predictions", [])
                if not isinstance(predictions, list):
                    predictions = []
            except Exception as exc:
                logger.warning(
                    "Inference failed for %s frame %d at %.2fs (%s)",
                    video_path,
                    idx,
                    timestamp_sec,
                    exc,
                )
                predictions = []

            has_hit = _frame_has_hit(
                predictions=predictions,
                conf_th=config.CONF_TH,
                suspicious_classes=config.YOLO_SUSPICIOUS_CLASSES,
            )
            if has_hit:
                frames_with_hits += 1

            max_conf = 0.0
            for pred in predictions:
                conf = pred.get("confidence", pred.get("score", 0.0))
                try:
                    max_conf = max(max_conf, float(conf))
                except (TypeError, ValueError):
                    continue

            per_frame.append(
                {
                    "frame_index": idx,
                    "timestamp_sec": round(float(timestamp_sec), 3),
                    "prediction_count": len(predictions),
                    "max_confidence": round(max_conf, 4),
                    "hit": has_hit,
                }
            )

        event_score = (frames_with_hits / sampled_frames) if sampled_frames else 0.0
        label = "suspicious" if event_score >= config.EVENT_TH else "normal"

        row = {
            "video_name": video_path.name,
            "video_path": str(video_path.resolve()),
            "duration_sec": round(duration_sec, 3),
            "sampled_frames": sampled_frames,
            "frames_with_hits": frames_with_hits,
            "event_score": round(event_score, 6),
            "label": label,
        }
        rows.append(row)

        video_id = make_video_id(video_path)
        out_json = config.YOLO_RUN_DIR / f"{video_id}.json"
        payload = {
            "video": row,
            "thresholds": {
                "conf_th": config.CONF_TH,
                "event_th": config.EVENT_TH,
                "sample_fps": config.SAMPLE_FPS,
                "yolo_max_videos": config.YOLO_MAX_VIDEOS,
                "yolo_max_frames_per_video": config.YOLO_MAX_FRAMES_PER_VIDEO,
                "yolo_max_seconds": config.YOLO_MAX_SECONDS,
                "yolo_suspicious_classes": config.YOLO_SUSPICIOUS_CLASSES,
            },
            "frames": per_frame,
        }
        with out_json.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        logger.info(
            "Processed %s | sampled=%d hits=%d score=%.4f label=%s",
            video_path.name,
            sampled_frames,
            frames_with_hits,
            event_score,
            label,
        )

    df = pd.DataFrame(rows, columns=columns)
    if not df.empty:
        df = df.sort_values(["video_path", "video_name"], kind="stable").reset_index(drop=True)

    out_csv = config.YOLO_RUN_DIR / "yolo_summary.csv"
    df.to_csv(out_csv, index=False)
    logger.info("Wrote YOLO summary CSV: %s", out_csv)


if __name__ == "__main__":
    main()

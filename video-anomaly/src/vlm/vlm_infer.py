from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from src.common import config
from src.common.utils import (
    discover_videos,
    ensure_dir,
    make_video_id,
    safe_json_parse,
    set_deterministic,
    setup_logger,
)
from src.common.video_io import get_video_metadata
from src.vlm.frame_sampler import sample_and_save_frames
from src.vlm.vlm_client import VLMClient

ALLOWED_LABELS = {"normal", "suspicious", "unknown"}
ALLOWED_TYPES = {
    "concealment",
    "intrusion",
    "tailgating",
    "loitering",
    "restricted-zone",
    "aggressive-grab",
    "unknown",
}


def _build_prompt() -> str:
    return (
        "You are a strict video safety classifier for retail/shop settings. "
        "Given sampled frames from one short video, return ONLY valid JSON and no extra text. "
        "Use this exact schema:\n"
        "{\n"
        '  "label": "normal"|"suspicious"|"unknown",\n'
        '  "suspicion_type": "concealment"|"intrusion"|"tailgating"|"loitering"|"restricted-zone"|"aggressive-grab"|"unknown",\n'
        '  "confidence": number between 0 and 1,\n'
        '  "evidence": [string, string, string] (max 3 short bullet-like strings)\n'
        "}\n"
        "Rules: output must be strict JSON object; no markdown fences; no additional keys. "
        "If uncertain, return label='unknown' and confidence <= 0.3."
    )


def _normalize_vlm_output(parsed: Dict[str, Any] | None) -> Dict[str, Any]:
    if not parsed:
        return {
            "label": "unknown",
            "suspicion_type": "unknown",
            "confidence": 0.0,
            "evidence": [],
        }

    label = str(parsed.get("label", "unknown")).strip().lower()
    if label not in ALLOWED_LABELS:
        label = "unknown"

    suspicion_type = str(parsed.get("suspicion_type", "unknown")).strip().lower()
    if suspicion_type not in ALLOWED_TYPES:
        suspicion_type = "unknown"

    confidence_raw = parsed.get("confidence", 0.0)
    try:
        confidence = float(confidence_raw)
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    evidence_raw = parsed.get("evidence", [])
    if isinstance(evidence_raw, list):
        evidence = [str(item).strip() for item in evidence_raw if str(item).strip()][:3]
    else:
        evidence = []

    return {
        "label": label,
        "suspicion_type": suspicion_type,
        "confidence": round(confidence, 6),
        "evidence": evidence,
    }


def main() -> None:
    logger = setup_logger("vlm.infer")
    set_deterministic(config.GLOBAL_SEED)

    ensure_dir(config.VLM_RUN_DIR)
    ensure_dir(config.FRAME_CACHE_DIR)

    video_sources = config.get_video_sources()
    logger.info("Video sources: %s", video_sources)

    videos = discover_videos(video_sources)
    logger.info("Discovered %d videos", len(videos))

    client = VLMClient(
        base_url=config.VLM_BASE_URL,
        model=config.VLM_MODEL,
        mode=config.VLM_MODE,
        custom_endpoint=config.VLM_CUSTOM_ENDPOINT,
        timeout_sec=config.VLM_HTTP_TIMEOUT_SEC,
        logger=logger,
    )

    prompt = _build_prompt()

    rows: List[Dict[str, Any]] = []

    for video_path in videos:
        try:
            metadata = get_video_metadata(video_path)
            duration_sec = float(metadata["duration_sec"])
        except Exception as exc:
            logger.warning("Skipping unreadable video %s (%s)", video_path, exc)
            continue

        video_id = make_video_id(video_path)
        frame_dir = config.FRAME_CACHE_DIR / video_id

        try:
            saved_frames = sample_and_save_frames(
                video_path=video_path,
                duration_sec=duration_sec,
                window_sec=config.WINDOW_SEC,
                n_frames=config.N_FRAMES,
                output_dir=frame_dir,
            )
        except Exception as exc:
            logger.warning("Frame sampling failed for %s (%s)", video_path, exc)
            saved_frames = []

        frame_paths = [str(item["frame_path"]) for item in saved_frames]

        result = client.infer(prompt=prompt, image_paths=frame_paths)
        if result.error:
            logger.warning("VLM request issue for %s: %s", video_path, result.error)

        if result.force_unknown:
            normalized = {
                "label": "unknown",
                "suspicion_type": "unknown",
                "confidence": 0.0,
                "evidence": [],
            }
            parsed_json = None
        else:
            parsed_json = safe_json_parse(result.text)
            normalized = _normalize_vlm_output(parsed_json)

        effective_window = float(min(config.WINDOW_SEC, duration_sec)) if duration_sec > 0 else 0.0

        row = {
            "video_name": video_path.name,
            "video_path": str(video_path.resolve()),
            "duration_sec": round(duration_sec, 3),
            "window_sec": round(effective_window, 3),
            "n_frames": len(frame_paths),
            "label": normalized["label"],
            "suspicion_type": normalized["suspicion_type"],
            "confidence": normalized["confidence"],
        }
        rows.append(row)

        out_json = config.VLM_RUN_DIR / f"{video_id}.json"
        payload = {
            "video": row,
            "config": {
                "vlm_mode": config.VLM_MODE,
                "vlm_model": config.VLM_MODEL,
                "window_sec": config.WINDOW_SEC,
                "n_frames": config.N_FRAMES,
            },
            "sampled_frames": saved_frames,
            "parsed_response": normalized,
            "raw_response": result.text,
            "raw_json": parsed_json,
        }
        with out_json.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        logger.info(
            "Processed %s | frames=%d label=%s type=%s conf=%.3f",
            video_path.name,
            len(frame_paths),
            normalized["label"],
            normalized["suspicion_type"],
            normalized["confidence"],
        )

    columns = [
        "video_name",
        "video_path",
        "duration_sec",
        "window_sec",
        "n_frames",
        "label",
        "suspicion_type",
        "confidence",
    ]

    df = pd.DataFrame(rows, columns=columns)
    if not df.empty:
        df = df.sort_values(["video_path", "video_name"], kind="stable").reset_index(drop=True)

    out_csv = config.VLM_RUN_DIR / "vlm_summary.csv"
    df.to_csv(out_csv, index=False)
    logger.info("Wrote VLM summary CSV: %s", out_csv)


if __name__ == "__main__":
    main()

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from src.common import config
from src.common.utils import ensure_dir, setup_logger

EXPECTED_YOLO_FIELDS = [
    "duration_sec",
    "sampled_frames",
    "frames_with_hits",
    "event_score",
    "label",
]

EXPECTED_VLM_FIELDS = [
    "duration_sec",
    "window_sec",
    "n_frames",
    "label",
    "suspicion_type",
    "confidence",
]


def _load_csv(path, logger) -> pd.DataFrame:
    if not path.exists():
        logger.warning("CSV not found: %s", path)
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as exc:
        logger.warning("Failed to load %s (%s)", path, exc)
        return pd.DataFrame()


def _norm(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() == "nan":
        return ""
    return text


def _combined_columns(yolo_fields: List[str], vlm_fields: List[str]) -> List[str]:
    columns = ["video_name", "video_path", "match_strategy"]
    columns.extend([f"yolo_{field}" for field in yolo_fields])
    columns.extend([f"vlm_{field}" for field in vlm_fields])
    return columns


def main() -> None:
    logger = setup_logger("eval.merge")

    ensure_dir(config.RUNS_DIR)

    yolo_csv = config.YOLO_RUN_DIR / "yolo_summary.csv"
    vlm_csv = config.VLM_RUN_DIR / "vlm_summary.csv"

    yolo_df = _load_csv(yolo_csv, logger)
    vlm_df = _load_csv(vlm_csv, logger)

    yolo_records = yolo_df.to_dict(orient="records") if not yolo_df.empty else []
    vlm_records = vlm_df.to_dict(orient="records") if not vlm_df.empty else []

    yolo_fields = [c for c in yolo_df.columns if c not in {"video_name", "video_path"}]
    vlm_fields = [c for c in vlm_df.columns if c not in {"video_name", "video_path"}]

    if not yolo_fields:
        yolo_fields = EXPECTED_YOLO_FIELDS
    if not vlm_fields:
        vlm_fields = EXPECTED_VLM_FIELDS

    vlm_by_path: Dict[str, int] = {}
    vlm_by_name: Dict[str, List[int]] = {}

    for idx, row in enumerate(vlm_records):
        path_key = _norm(row.get("video_path"))
        name_key = _norm(row.get("video_name"))

        if path_key and path_key not in vlm_by_path:
            vlm_by_path[path_key] = idx

        if name_key:
            vlm_by_name.setdefault(name_key, []).append(idx)

    used_vlm = set()
    merged_rows: List[Dict[str, Any]] = []

    for yolo_row in yolo_records:
        yolo_path = _norm(yolo_row.get("video_path"))
        yolo_name = _norm(yolo_row.get("video_name"))

        chosen_idx = None
        match_strategy = "none"

        if yolo_path in vlm_by_path:
            candidate = vlm_by_path[yolo_path]
            if candidate not in used_vlm:
                chosen_idx = candidate
                match_strategy = "video_path"

        if chosen_idx is None and yolo_name in vlm_by_name:
            for candidate in vlm_by_name[yolo_name]:
                if candidate not in used_vlm:
                    chosen_idx = candidate
                    match_strategy = "video_name"
                    break

        vlm_row = vlm_records[chosen_idx] if chosen_idx is not None else {}
        if chosen_idx is not None:
            used_vlm.add(chosen_idx)

        merged = {
            "video_name": yolo_name or _norm(vlm_row.get("video_name")),
            "video_path": yolo_path or _norm(vlm_row.get("video_path")),
            "match_strategy": match_strategy,
        }

        for field in yolo_fields:
            merged[f"yolo_{field}"] = yolo_row.get(field, "")
        for field in vlm_fields:
            merged[f"vlm_{field}"] = vlm_row.get(field, "")

        merged_rows.append(merged)

    for idx, vlm_row in enumerate(vlm_records):
        if idx in used_vlm:
            continue

        merged = {
            "video_name": _norm(vlm_row.get("video_name")),
            "video_path": _norm(vlm_row.get("video_path")),
            "match_strategy": "vlm_only",
        }

        for field in yolo_fields:
            merged[f"yolo_{field}"] = ""
        for field in vlm_fields:
            merged[f"vlm_{field}"] = vlm_row.get(field, "")

        merged_rows.append(merged)

    columns = _combined_columns(yolo_fields, vlm_fields)
    merged_df = pd.DataFrame(merged_rows, columns=columns)
    if not merged_df.empty:
        merged_df = merged_df.sort_values(
            ["video_path", "video_name"], kind="stable"
        ).reset_index(drop=True)

    out_csv = config.MERGED_RESULTS_PATH
    merged_df.to_csv(out_csv, index=False)
    logger.info("Wrote merged CSV: %s", out_csv)


if __name__ == "__main__":
    main()

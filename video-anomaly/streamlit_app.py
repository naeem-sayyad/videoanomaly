from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st

from src.common import config
from src.common.utils import discover_videos, ensure_dir

REPO_ROOT = Path(__file__).resolve().parent
RAW_VIDEO_DIR = (REPO_ROOT / "data" / "raw_videos").resolve()
YOLO_CSV = REPO_ROOT / "runs" / "yolo_events" / "yolo_summary.csv"
VLM_CSV = REPO_ROOT / "runs" / "vlm_events" / "vlm_summary.csv"
MERGED_CSV = REPO_ROOT / "runs" / "merged_results.csv"


def _secret_value(key: str) -> str:
    try:
        if key in st.secrets:
            value = st.secrets[key]
            if value is not None:
                return str(value)
    except Exception:
        return ""
    return ""


def _env_or_secret(key: str, fallback: str = "") -> str:
    env_val = os.getenv(key, "").strip()
    if env_val:
        return env_val

    secret_val = _secret_value(key).strip()
    if secret_val:
        return secret_val

    return fallback


def _float_value(key: str, fallback: float) -> float:
    raw = _env_or_secret(key, str(fallback))
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(fallback)


def _int_value(key: str, fallback: int) -> int:
    raw = _env_or_secret(key, str(fallback))
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return int(fallback)


def _parse_video_sources(raw: str) -> List[str]:
    parsed: List[str] = []
    for part in raw.split(":"):
        candidate = part.strip()
        if not candidate:
            continue
        parsed.append(str(Path(candidate).expanduser().resolve()))

    if not parsed:
        parsed = [str(Path(p).expanduser().resolve()) for p in config.DEFAULT_VIDEO_SOURCES]

    local_source = str(RAW_VIDEO_DIR)
    if local_source not in parsed:
        parsed.append(local_source)

    deduped: List[str] = []
    seen = set()
    for path in parsed:
        if path in seen:
            continue
        seen.add(path)
        deduped.append(path)
    return deduped


def _build_env(overrides: Dict[str, str]) -> Dict[str, str]:
    env = os.environ.copy()
    for key, value in overrides.items():
        if value is None:
            continue
        env[key] = str(value)
    return env


def _run_module(module_name: str, env: Dict[str, str]) -> Tuple[int, str]:
    cmd = [sys.executable, "-m", module_name]
    result = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    pieces: List[str] = []
    if result.stdout:
        pieces.append(result.stdout.strip())
    if result.stderr:
        pieces.append(result.stderr.strip())

    output = "\n\n".join(p for p in pieces if p)
    return result.returncode, output


def _render_csv(title: str, path: Path, key_prefix: str) -> None:
    st.subheader(title)
    if not path.exists():
        st.info(f"Not generated yet: {path}")
        return

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        st.error(f"Failed to read {path}: {exc}")
        return

    st.caption(f"{path} | rows: {len(df)}")
    st.dataframe(df, use_container_width=True)
    st.download_button(
        label=f"Download {path.name}",
        data=path.read_bytes(),
        file_name=path.name,
        mime="text/csv",
        key=f"{key_prefix}_download",
    )


def _save_uploaded_videos(uploaded_files) -> Tuple[int, List[str]]:
    if not uploaded_files:
        return 0, []

    ensure_dir(RAW_VIDEO_DIR)
    saved: List[str] = []
    skipped = 0

    for uploaded in uploaded_files:
        suffix = Path(uploaded.name).suffix.lower()
        if suffix not in config.VIDEO_EXTENSIONS:
            skipped += 1
            continue

        output_path = RAW_VIDEO_DIR / uploaded.name
        output_path.write_bytes(uploaded.getbuffer())
        saved.append(str(output_path.resolve()))

    return skipped, saved


def main() -> None:
    st.set_page_config(page_title="Video Anomaly Baselines", layout="wide")

    st.title("Video Anomaly Baselines")
    st.write("Run YOLO (Roboflow), VLM, and merge pipelines from one Streamlit app.")

    default_sources_raw = _env_or_secret(
        "VIDEO_SOURCES", ":".join(config.DEFAULT_VIDEO_SOURCES)
    )

    with st.sidebar:
        st.header("Runtime Config")

        video_sources_raw = st.text_input(
            "VIDEO_SOURCES",
            value=default_sources_raw,
            help="Colon-separated paths. data/raw_videos is appended automatically.",
        )

        st.markdown("### Roboflow")
        rf_key = st.text_input(
            "ROBOFLOW_API_KEY",
            value=_env_or_secret("ROBOFLOW_API_KEY", ""),
            type="password",
        )
        rf_workspace = st.text_input(
            "ROBOFLOW_WORKSPACE",
            value=_env_or_secret("ROBOFLOW_WORKSPACE", ""),
        )
        rf_project = st.text_input(
            "ROBOFLOW_PROJECT",
            value=_env_or_secret("ROBOFLOW_PROJECT", ""),
        )
        rf_version = st.text_input(
            "ROBOFLOW_VERSION",
            value=_env_or_secret("ROBOFLOW_VERSION", ""),
        )

        conf_th = st.number_input("CONF_TH", min_value=0.0, max_value=1.0, value=_float_value("CONF_TH", 0.35), step=0.01)
        event_th = st.number_input("EVENT_TH", min_value=0.0, max_value=1.0, value=_float_value("EVENT_TH", 0.15), step=0.01)
        sample_fps = st.number_input("SAMPLE_FPS", min_value=0.1, max_value=30.0, value=max(_float_value("SAMPLE_FPS", 1.0), 0.1), step=0.1)

        st.markdown("### VLM")
        vlm_base_url = st.text_input(
            "VLM_BASE_URL",
            value=_env_or_secret("VLM_BASE_URL", "http://127.0.0.1:8080"),
        )
        vlm_model = st.text_input(
            "VLM_MODEL",
            value=_env_or_secret("VLM_MODEL", ""),
        )
        vlm_mode = st.selectbox(
            "VLM_MODE",
            options=["openai", "custom"],
            index=0 if _env_or_secret("VLM_MODE", "openai").lower() != "custom" else 1,
        )
        vlm_custom_endpoint = st.text_input(
            "VLM_CUSTOM_ENDPOINT",
            value=_env_or_secret("VLM_CUSTOM_ENDPOINT", "/completion"),
        )
        window_sec = st.number_input(
            "WINDOW_SEC",
            min_value=1,
            max_value=180,
            value=max(_int_value("WINDOW_SEC", 60), 1),
            step=1,
        )
        n_frames = st.number_input(
            "N_FRAMES",
            min_value=1,
            max_value=96,
            value=max(_int_value("N_FRAMES", 24), 1),
            step=1,
        )

        st.markdown("### General")
        global_seed = st.number_input(
            "GLOBAL_SEED",
            min_value=0,
            max_value=2_147_483_647,
            value=max(_int_value("GLOBAL_SEED", 42), 0),
            step=1,
        )

    resolved_sources = _parse_video_sources(video_sources_raw)
    st.caption("Resolved video sources:")
    for source in resolved_sources:
        st.code(source, language="text")

    st.subheader("Upload Videos")
    uploaded_files = st.file_uploader(
        "Upload video files for processing",
        type=["mp4", "mov", "avi", "mkv"],
        accept_multiple_files=True,
    )
    if st.button("Save Uploaded Videos"):
        skipped, saved = _save_uploaded_videos(uploaded_files)
        if saved:
            st.success(f"Saved {len(saved)} video(s) to {RAW_VIDEO_DIR}")
        if skipped:
            st.warning(f"Skipped {skipped} file(s) with unsupported extensions")
        if not saved and not skipped:
            st.info("No files uploaded")

    existing_raw_videos = discover_videos([str(RAW_VIDEO_DIR)])
    st.caption(f"Videos currently in data/raw_videos: {len(existing_raw_videos)}")

    runtime_env = _build_env(
        {
            "VIDEO_SOURCES": video_sources_raw,
            "ROBOFLOW_API_KEY": rf_key,
            "ROBOFLOW_WORKSPACE": rf_workspace,
            "ROBOFLOW_PROJECT": rf_project,
            "ROBOFLOW_VERSION": rf_version,
            "CONF_TH": str(conf_th),
            "EVENT_TH": str(event_th),
            "SAMPLE_FPS": str(sample_fps),
            "VLM_BASE_URL": vlm_base_url,
            "VLM_MODEL": vlm_model,
            "VLM_MODE": vlm_mode,
            "VLM_CUSTOM_ENDPOINT": vlm_custom_endpoint,
            "WINDOW_SEC": str(int(window_sec)),
            "N_FRAMES": str(int(n_frames)),
            "GLOBAL_SEED": str(int(global_seed)),
        }
    )

    if st.button("Scan Video Sources"):
        videos = discover_videos(resolved_sources)
        st.success(f"Discovered {len(videos)} video(s)")
        preview = [str(v) for v in videos[:20]]
        if preview:
            st.write("First discovered videos:")
            for item in preview:
                st.code(item, language="text")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        run_yolo = st.button("Run YOLO")
    with col2:
        run_vlm = st.button("Run VLM")
    with col3:
        run_merge = st.button("Run Merge")
    with col4:
        run_all = st.button("Run All")

    steps: List[Tuple[str, str]] = []
    if run_all:
        steps = [
            ("src.yolo.roboflow_infer", "YOLO"),
            ("src.vlm.vlm_infer", "VLM"),
            ("src.eval.merge_results", "MERGE"),
        ]
    elif run_yolo:
        steps = [("src.yolo.roboflow_infer", "YOLO")]
    elif run_vlm:
        steps = [("src.vlm.vlm_infer", "VLM")]
    elif run_merge:
        steps = [("src.eval.merge_results", "MERGE")]

    if steps:
        logs: List[str] = []
        failures: List[str] = []

        for module_name, label in steps:
            with st.spinner(f"Running {label}..."):
                code, output = _run_module(module_name, runtime_env)

            logs.append(f"[{label}] exit_code={code}")
            if output:
                logs.append(output)

            if code != 0:
                failures.append(label)

        combined_logs = "\n\n".join(logs)
        st.text_area("Execution Logs", value=combined_logs, height=320)

        if failures:
            st.error(f"Failed steps: {', '.join(failures)}")
        else:
            st.success("Selected step(s) completed")

    st.divider()
    _render_csv("YOLO Summary", YOLO_CSV, "yolo")
    _render_csv("VLM Summary", VLM_CSV, "vlm")
    _render_csv("Merged Summary", MERGED_CSV, "merged")


if __name__ == "__main__":
    main()

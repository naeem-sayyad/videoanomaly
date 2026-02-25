# video-anomaly

Minimal production-clean Python repo for two short-video anomaly baselines:

1. Roboflow hosted YOLO-style detection baseline for shoplifting/suspicious activity.
2. VLM baseline via local HTTP server (OpenAI-compatible or custom llama.cpp-style endpoint).

Both baselines produce per-video JSON + CSV, and a merged CSV report.

## Requirements

- Python 3.10+
- Mac/Linux
- CPU is sufficient
- Dependencies:
  - `opencv-python`
  - `numpy`
  - `requests`
  - `pandas`
  - `streamlit`

## Install

```bash
cd video-anomaly
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Streamlit UI (Local)

Start the app:

```bash
streamlit run streamlit_app.py
```

Optional local secrets file:

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

The sidebar lets you set environment/runtime values for:

- `VIDEO_SOURCES`
- Roboflow credentials and thresholds
- VLM server settings and frame-sampling settings
- `GLOBAL_SEED`

From the UI you can:

- Scan video folders
- Upload videos into `data/raw_videos/`
- Run YOLO baseline
- Run VLM baseline
- Run merge
- Download generated CSVs

### If YOLO feels slow or misses suspicious clips

Use these Streamlit sidebar settings for faster, higher-recall iteration:

- Enable `YOLO only uploaded videos`
- Set `YOLO_MAX_VIDEOS=1` while testing one clip
- Set `YOLO_MAX_FRAMES_PER_VIDEO=120` (or lower for speed)
- Set `CONF_TH=0.2` and `EVENT_TH=0.05` for higher recall

## Deploy to Streamlit Community Cloud

1. Push this folder to a GitHub repo.
2. In Streamlit Community Cloud, create a new app with entrypoint `streamlit_app.py`.
3. Keep `packages.txt` in the repo root (Streamlit Cloud will install these system packages for OpenCV/video IO).
4. In app Secrets, add keys from `.streamlit/secrets.toml.example`.
5. Redeploy the app.

Important: Streamlit Cloud cannot read local machine paths like `/Users/naeem/...`.
Set `VIDEO_SOURCES` in Secrets to paths available inside your deployment environment, or use `data/raw_videos/` in the repo.
Also set `VLM_BASE_URL` to a reachable endpoint from Streamlit Cloud (not your laptop `127.0.0.1`).

## Video Inputs

By default, the pipeline scans these absolute folders recursively:

- `/Users/naeem/Downloads/archive-2`
- `/Users/naeem/Downloads/Dataset`

It also scans optional local folder:

- `data/raw_videos/`

Supported extensions: `.mp4 .mov .avi .mkv`

Videos are deduplicated by absolute file path.

### Override video sources

Use a colon-separated env var (`Mac/Linux`):

```bash
export VIDEO_SOURCES="/path/a:/path/b"
```

`data/raw_videos/` remains included automatically as an optional extra source.

## Baseline 1: Roboflow YOLO Hosted API

### Required environment variables

```bash
export ROBOFLOW_API_KEY="<your_api_key>"
export ROBOFLOW_WORKSPACE="<your_workspace>"   # optional for some endpoint formats
export ROBOFLOW_PROJECT="<your_project_slug>"
export ROBOFLOW_VERSION="<model_version_number>"
```

### Optional tuning (defaults in `src/common/config.py`)

```bash
export CONF_TH=0.35
export EVENT_TH=0.15
export SAMPLE_FPS=1.0
```

### Run

```bash
python -m src.yolo.roboflow_infer
```

### Output

- Per-video JSON: `runs/yolo_events/<video_id>.json`
- Summary CSV: `runs/yolo_events/yolo_summary.csv`

CSV columns:

- `video_name`
- `video_path`
- `duration_sec`
- `sampled_frames`
- `frames_with_hits`
- `event_score`
- `label` (`suspicious` or `normal`)

## Baseline 2: VLM over HTTP

### Environment variables

```bash
export VLM_BASE_URL="http://127.0.0.1:8080"
export VLM_MODEL="<model_name_if_required>"
export VLM_MODE="openai"   # or "custom"
```

Optional custom endpoint for `VLM_MODE=custom`:

```bash
export VLM_CUSTOM_ENDPOINT="/completion"
```

Optional sampling config:

```bash
export WINDOW_SEC=60
export N_FRAMES=24
```

### Run

```bash
python -m src.vlm.vlm_infer
```

### VLM image-support limitation handling

Not all local servers support image input. This pipeline is resilient:

- If image payload fails with 4xx, it logs a warning and falls back safely.
- It still emits output for that video with:
  - `label="unknown"`
  - `suspicion_type="unknown"`
  - `confidence=0`

### Output

- Saved frames: `data/frames/<video_id>/`
- Per-video JSON: `runs/vlm_events/<video_id>.json`
- Summary CSV: `runs/vlm_events/vlm_summary.csv`

CSV columns:

- `video_name`
- `video_path`
- `duration_sec`
- `window_sec`
- `n_frames`
- `label`
- `suspicion_type`
- `confidence`

## Merge Reports

Run after both baselines:

```bash
python -m src.eval.merge_results
```

Output:

- `runs/merged_results.csv`

Merge behavior:

- Primary key: `video_path`
- Fallback key: `video_name`

## Determinism and Robustness

- Seeded where applicable (`GLOBAL_SEED`, default `42`).
- Graceful skipping of unreadable/bad videos with warning logs.
- Stable sorting of video lists and CSV outputs.

# Italy Pipeline - NAS Downloader & Cow Identification System

End-to-end pipeline for downloading, processing, and sorting thermal/depth video recordings of dairy cows at Italian farms (Piacenza, Padova). Combines UNet segmentation with Siamese network-based cow identification to automatically organize images by individual animal.

**Location:** `C:\Users\lobat\Dropbox\US_2026\Italy_pipeline`

---

## Table of Contents

- [Quick Start](#quick-start)
- [System Architecture](#system-architecture)
- [Directory Structure](#directory-structure)
- [Web UI](#web-ui)
- [Pipeline Phases](#pipeline-phases)
- [Siamese Network](#siamese-network)
- [UNet Segmentation](#unet-segmentation)
- [NAS Integration](#nas-integration)
- [CSV Format](#csv-format)
- [Configuration](#configuration)
- [Environment Setup](#environment-setup)

---

## Quick Start

```bash
cd Italy_pipeline/downloader
python app.py
```

Or double-click `downloader/run.bat`. The browser opens automatically at **http://localhost:5050**.

---

## System Architecture

```
                          +---------------------+
                          |   Web UI (Flask)    |
                          |   localhost:5050    |
                          +----------+----------+
                                     |
                     +---------------+---------------+
                     |                               |
              +------v------+               +--------v--------+
              |  NAS API    |               |  CSV Parser     |
              |  (Synology  |               |  (Sorting Gate) |
              |  FileStation)|              +---------+--------+
              +------+------+                        |
                     |                               |
            +--------v---------+          +----------v----------+
            | Phase 1: Download|          | Timestamp matching  |
            | raw mp4/npz files|          | Italy CET/CEST->UTC |
            +--------+---------+          +----------+----------+
                     |                               |
            +--------v---------+                     |
            | Phase 2: Extract |                     |
            | IR + Depth frames|                     |
            +--------+---------+                     |
                     |                               |
            +--------v---------+                     |
            | Phase 3: UNet    |                     |
            | Segmentation     |                     |
            +--------+---------+                     |
                     |                               |
            +--------v---------+                     |
            | Phase 4: Filter  |                     |
            | Bad Masks        |                     |
            +--------+---------+                     |
                     |                               |
            +--------v---------+                     |
            | Phase 5: Overlap |                     |
            | Mask + Crop IR   |                     |
            +--------+---------+                     |
                     |                               |
            +--------v---------+    +----------------+
            | Phase 6: Siamese |<---+
            | Cow ID Sorting   |
            | (GPU subprocess) |
            +--------+---------+
                     |
            +--------v---------+
            | Phase 7: Sync    |
            | folders by cow   |
            +--------+---------+
                     |
            +--------v---------+
            | Cleanup raw/     |
            | segmentation/    |
            | trash/           |
            +------------------+
```

---

## Directory Structure

```
Italy_pipeline/
├── downloader/                               # Web application
│   ├── app.py                                # Flask backend + all pipeline logic
│   ├── templates/index.html                  # Frontend UI (single-page, dark theme)
│   └── run.bat                               # Windows launcher
│
├── scripts/
│   └── siamese_sort_by_day.py                # Siamese cow-ID sorting (GPU subprocess)
│
├── Siamese/                                  # Siamese network assets
│   ├── best.pth                              # ResNet50 checkpoint (~98 MB)
│   ├── centroids_cache.pt                    # Precomputed class centroids (~83 KB)
│   └── train/<cow_id>/                       # Training images per Animal ID
│
├── segmentation/                             # UNet segmentation
│   └── group_pen_or_lanes/
│       ├── combined_depth_processing.py      # Depth processing utilities
│       └── unet/
│           ├── config/config.yml             # UNet configuration
│           ├── models/marshfield_chute_100.pth  # UNet weights (~98 MB)
│           └── src/                          # UNet source code
│
├── csv/                                      # Uploaded sorting gate CSV files
│
├── pendent_data_<farm>/                      # Pipeline output (per farm)
│   └── labeled/
│       ├── infrared/YYYYMMDD/<cow_id>/       # IR grayscale frames (.png)
│       ├── depth/YYYYMMDD/<cow_id>/          # 16-bit depth arrays (.npy)
│       ├── masks/YYYYMMDD/<cow_id>/          # Binary segmentation masks (.png)
│       └── infrared_overlap/YYYYMMDD/<cow_id>/ # Masked + tight-cropped IR
│
└── PIPELINE.md                               # Technical documentation
```

---

## Web UI

The application provides a single-page web interface with 4 cards:

### Card 1 - Select Farm
Choose between **Piacenza** or **Padova**. Queries the NAS and displays the available date range and total file count.

### Card 2 - Select Mode + Scan
Two modes:
- **Manual Dates** - Pick start/end date, click *Scan Files*
- **CSV Upload** - Upload sorting gate CSV. The app parses it, shows a summary (animals, records, date range), with optional date filtering

After scanning:
- **Download Only** - Copy raw files from NAS (no processing)
- **Download + Organize by Cow** - Full 7-phase pipeline

### Card 3 - Update Centroids Cache
Rebuild `Siamese/centroids_cache.pt` from training set images.
- Browse button for folder selection (native Windows dialog)
- GPU (CUDA) / CPU radio toggle
- Runs in background via Siamese conda environment

### Card 4 - Progress
Live progress bar with phase label, current file indicator, error list, and completion message.

---

## Pipeline Phases

### Phase 1 - Download from NAS
- Authenticates with Synology NAS via FileStation API
- Uses **binary search** on the sorted file listing to efficiently locate date ranges
- Downloads `_i.mp4` (infrared video), `_d.mp4` (depth video), `.npz` (frame metadata)
- Saves to `pendent_data_<farm>/raw/`

### Phase 2 - Extract Frames
**Infrared** (`_i.mp4` + `_i.npz`):
- Reads video frames with `skvideo.io`
- Each frame saved as grayscale PNG using `matplotlib.imsave(cmap="gray")`
- Output: `labeled/infrared/YYYYMMDD/<filename>_i.png`

**Depth** (`_d.mp4` + `_d.npz`):
- Reconstructs 16-bit depth from two 8-bit video channels: `depth = R * 256 + G`
- Output: `labeled/depth/YYYYMMDD/<filename>.npy`

Day folder (YYYYMMDD) extracted from filename timestamp, e.g., `jetnano1_20250616042137493_i.png`

### Phase 3 - UNet Segmentation
- Normalizes depth arrays to grayscale (1st-99th percentile clipping)
- Runs UNet inference (batch size 16) on all grayscale images
- Binary threshold at 0.5 on sigmoid output
- Model: `marshfield_chute_100.pth` (trained on lane/chute segmentation)
- Output masks: `labeled/masks/YYYYMMDD/`
- Also generates splash visualizations (mask overlay on grayscale)

### Phase 4 - Filter Bad Masks
Two rejection criteria:
1. **Area < 15%** of total pixels (cow too small / partial)
2. **Mask touches image border** (cow not fully in frame)

Rejected frames (depth + IR + mask) are moved to `trash/`.

### Phase 5 - Create Infrared Overlap
For each IR frame with a valid mask:
1. Apply binary mask to IR image (background set to black)
2. Compute tight bounding box around masked region
3. Crop to bounding box
4. Save to `labeled/infrared_overlap/YYYYMMDD/`

IR and depth frames without a matching mask are moved to `trash/`.

### Phase 6 - Siamese Cow ID Sorting
Runs `scripts/siamese_sort_by_day.py` as a **subprocess** using the Siamese conda env (GPU/CUDA).

For each image in `infrared_overlap/YYYYMMDD/`:
1. Parse UTC timestamp from filename
2. Convert to Italy local time (CET UTC+1 or CEST UTC+2, auto-detected)
3. Query CSV for candidate Animal IDs within **+/-5 minutes** of the image timestamp
4. Filter candidates to those present in the training set (`centroids_cache.pt`)
5. Compute cosine similarity against candidate centroids only
6. Apply softmax with **temperature 0.1** for sharp probability distribution
7. If confidence >= **0.20** -> move to `YYYYMMDD/<cow_id>/`
8. If no candidates or confidence < 0.20 -> move to `YYYYMMDD/out/`

**Why candidate filtering?**
Multiple cows share the same timestamp window (e.g., 8 cows at 06:22). Comparing only against the cows present at that time gives confidence 0.84-1.0 instead of ~0.20 when comparing against all classes.

### Phase 7 - Sync Folders
Mirrors the cow subfolder structure from `infrared_overlap/` into `infrared/`, `depth/`, and `masks/`:
- `*_i.png` -> `infrared/YYYYMMDD/<cow_id>/`
- `*.npy` -> `depth/YYYYMMDD/<cow_id>/`
- `*.png` -> `masks/YYYYMMDD/<cow_id>/`

### Cleanup
After completion, temporary folders are automatically deleted:
- `raw/` (downloaded videos)
- `segmentation/` (UNet intermediates)
- `trash/` (rejected frames)

Only `labeled/` is retained.

---

## Siamese Network

| Property       | Value                                    |
|----------------|------------------------------------------|
| Architecture   | ResNet50 backbone                        |
| Embedding dim  | 128                                      |
| Similarity     | Cosine similarity (L2-normalized)        |
| Gallery        | Mean centroid per class, precomputed     |
| Inference      | Vectorized matrix multiply `(1xD) @ (NxD)^T` |
| Checkpoint     | `Siamese/best.pth`                      |
| Source code    | `C:\Users\lobat\Dropbox\US_2026\Siamese` |

### Centroids Cache
`Siamese/centroids_cache.pt` stores:
- `matrix`: `(N_classes x 128)` tensor of L2-normalized centroid embeddings
- `classes`: sorted list of class names (Animal ID strings)

Rebuilt via the UI ("Update Centroids Cache" card) or by running the update script manually.

### Key Parameters

| Parameter       | Value | Description                                       |
|-----------------|-------|---------------------------------------------------|
| `WINDOW_MIN`    | 5     | +/- minutes around image timestamp to search CSV  |
| `MIN_CONF`      | 0.20  | Minimum softmax confidence to accept prediction   |
| `SOFTMAX_TEMP`  | 0.1   | Temperature for softmax sharpening                |

---

## UNet Segmentation

- **Task:** Binary segmentation of cow body in depth frames (lane/chute view)
- **Model:** `marshfield_chute_100.pth` - trained for 100 epochs on Marshfield chute data
- **Input:** Grayscale depth images (percentile-normalized)
- **Output:** Binary masks (sigmoid > 0.5)
- **Batch size:** 16
- **Config:** `segmentation/group_pen_or_lanes/unet/config/config.yml`

---

## NAS Integration

Connects to a Synology NAS via the **FileStation API**:

| Endpoint    | API                        | Usage                    |
|-------------|----------------------------|--------------------------|
| Login       | `SYNO.API.Auth`            | Authenticate, get SID    |
| List        | `SYNO.FileStation.List`    | Browse folders, paginate |
| Download    | `SYNO.FileStation.Download`| Download individual files|

### Binary Search for Date Ranges
Files on the NAS are sorted by name (which contains the date). The pipeline uses binary search on the file listing offsets to efficiently locate the start and end of a date range, avoiding full-list scans of potentially hundreds of thousands of files.

### NAS Folder Structure
```
NAS03/Backups/
├── piacenza/     # Piacenza farm recordings
│   ├── jetnano1_20250610041500000_i.mp4
│   ├── jetnano1_20250610041500000_d.mp4
│   ├── jetnano1_20250610041500000.npz
│   └── ...
└── padova/       # Padova farm recordings
    └── ...
```

File naming convention: `<device>_<YYYYMMDD HHMMSS mmm>_<type>.<ext>`
- `_i.mp4` / `_i.npz` - Infrared video + metadata
- `_d.mp4` / `_d.npz` - Depth video + metadata

---

## CSV Format

The sorting gate CSV contains one row per cow passage:

| Column      | Type    | Description                               |
|-------------|---------|-------------------------------------------|
| `Animal ID` | Integer | Cow identifier                            |
| `End Time`  | String  | Local Italy time when cow exited the gate |
| `Weight`    | Float   | Body weight in kg (rows with 0 discarded) |

### Timezone Conversion
Italy uses CET (UTC+1) in winter and CEST (UTC+2) in summer. The pipeline auto-detects:
- **CET** (Oct 27 - Mar 30): subtract 1 hour
- **CEST** (Mar 30 - Oct 27): subtract 2 hours

---

## Configuration

All paths and credentials are defined at the top of `downloader/app.py`:

| Variable             | Description                              |
|----------------------|------------------------------------------|
| `NAS_BASE`           | Synology FileStation API base URL        |
| `NAS_USER`           | NAS username                             |
| `NAS_PASS`           | NAS password                             |
| `NAS_FOLDERS`        | Dict mapping farm name -> NAS path       |
| `SIAMESE_PYTHON`     | Path to Siamese conda env Python exe     |
| `SIAMESE_CHECKPOINT` | Path to `best.pth`                       |
| `CENTROIDS_CACHE`    | Path to `centroids_cache.pt`             |
| `SORT_SCRIPT`        | Path to `siamese_sort_by_day.py`         |
| `SEG_DIR`            | Root of segmentation code                |
| `SEG_CONFIG`         | UNet config YAML                         |
| `SEG_MODEL`          | UNet model weights                       |
| `WINDOW_MIN`         | CSV match window in minutes (default: 5) |
| `MIN_CONF`           | Minimum Siamese confidence (default: 0.20)|

---

## Environment Setup

### Main Flask App (Python 3.14)
```
pip install flask requests pandas numpy Pillow skvideo opencv-python torch
```

### Siamese Conda Environment (GPU inference)
The Siamese sorting and centroid rebuild run as **subprocesses** using a separate conda environment with CUDA-enabled PyTorch. This isolates GPU dependencies from the main app.

```bash
conda create -n Siamese python=3.11
conda activate Siamese
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install Pillow pandas numpy
```

Path: `C:\Users\lobat\miniconda3\envs\Siamese\python.exe`

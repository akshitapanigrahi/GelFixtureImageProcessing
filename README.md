# Gel Tracking & Volume Estimation Pipeline

This repository contains a Python pipeline that loads images from a directory, detects gel meshes, tracks their position/area/volume over time **and** computes those metrics relative to a moving test fixture. It also ships helper utilities for one-time pre-processing (bounding boxes, mm↔px calibration, FPS estimation) and rich visualizations (overlays, per-gel plots, and movies).

---

## Highlights

* **Gel detection** using \[Segment Anything (SAM)] with post-filtering inside a user-defined ROI.
* **Per-gel tracking** (left→right association per frame) with centroid, area, and an estimated volume.
* **Fixture-relative metrics** via automatic detection of a vertical white stripe inside a fixture ROI.
* **Physical units** (mm, mm², mm³) using your mm/pixel calibration.
* **Batch outputs**: master CSV, per-gel CSVs, overlay PNGs, time-series plots, and MP4s.
* **Pre-processing tools** for bounding boxes, mm/pixel, and FPS metadata (great on macOS / “Phantom-like” tasks without vendor software).

---

## Quick Start

1. **Install dependencies**

```bash
# Python >=3.10 recommended
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # or cpu wheels
pip install opencv-python matplotlib pyyaml tqdm pandas
```

2. **Download a SAM checkpoint** (e.g., `sam_vit_h_4b8939.pth`) and set its path in `config.yaml` (see below).

3. **Set up `config.yaml`** with your paths and parameters.

4. **Run the pre-processing helpers** (see detailed steps next).

5. **Process your image sequence**

```bash
python OBJECT_DETECTION.py -c config.yaml
```

Outputs will be written to `overlay_dir` and `metrics_dir` from your config.

---

## Pre-processing Workflow (Do This Once per Dataset)

These steps populate `config.yaml` with scene-specific constants so the main script can compute correct, stable metrics.

### 1) Pick bounding boxes (run **twice**)

You will create **two** ROIs by running the same tool twice and pasting the printed arrays into `config.yaml`.

```bash
python find_bounding_box.py -c config.yaml
```

* First run: **Meshes ROI**
  Click **top** and **bottom** of the gel banding region (x spans the full image automatically).
  Copy the printed `input_box = [x_min, y_min, x_max, y_max]` and paste into:

  ```yaml
  bounding_box_meshes: [x_min, y_min, x_max, y_max]
  ```

* Second run: **Fixture ROI**
  Click **top** and **bottom** around the fixture’s vertical white stripe.
  Paste into:

  ```yaml
  fixture_bounding_box: [x_min, y_min, x_max, y_max]
  ```

> Tip: You can re-run anytime if you want to tighten/expand ROIs.

### 2) Compute mm/pixel (optional but recommended)

If you have a known width (default assumes **5 mm**), measure it to get a scene-specific scale:

```bash
python get_mm_px.py -c config.yaml
```

* Click two points spanning the known distance (horizontally for the provided default).
* The script prints `X mm/pixel`. Paste into:

  ```yaml
  mm_per_pixel: X
  ```

> The known width is set by `vert_known_width` in `config.yaml` (default `5` mm).
> If you already know `mm_per_pixel`, you can skip this step.

### 3) Estimate FPS from a video (optional)

If you don’t know the source FPS (e.g., exported TIFFs came from a camera stream), point to a related video:

```bash
python get_video_metadata.py -c config.yaml
```

* It prints `(total_frames, fps)`. Paste `fps` into:

  ```yaml
  fps: <value>
  ```

---

## Configuration (`config.yaml`)

An example (adapt to your paths):

```yaml
image_dir: /path/to/your/images                   # folder of .tif frames
video_mp4: /path/to/optional/video.mp4            # (optional) for FPS readback
base_image_dir: /path/to/one/reference_frame.tif  # used by the helper tools
save_mesh_overlays: True
overlay_dir: /path/to/output/overlays
metrics_dir: /path/to/output/metrics

# Segment Anything (SAM)
segment_anything_dir: /path/to/segment_anything
sam_checkpoint: /path/to/sam_vit_h_4b8939.pth
model_type: vit_h

# ROIs & scene constants
bounding_box_meshes: [2, 98, 638, 144]      # set via find_bounding_box.py (meshes)
fixture_bounding_box:  [0, 65, 639, 167]    # set via find_bounding_box.py (fixture)
fps: 30
mm_per_pixel: 0.2                           # set via get_mm_px.py or known a priori
min_area: 500                               # px² — reject tiny fragments
max_area: 1080                              # px² — reject giant masks
vert_known_width: 5                         # mm (used by get_mm_px.py)
top_area_known: 196.24                      # mm² (if you have a known footprint)
```

**Important knobs**

* `min_area` / `max_area`: quick way to ignore spurious SAM masks.
* `top_area_known`: if you know the nominal top footprint area, volume estimation uses it by default.

---

## How It Works

### 1) Mesh extraction (SAM + ROI filtering)

* A SAM model is loaded from `sam_checkpoint`.
* For each image:

  1. Generate candidate masks.
  2. **Filter** to those fully inside `bounding_box_meshes` and within area thresholds.
  3. **Sort** masks **left→right** to get a stable **Gel #** ordering per frame (0,1,2,…).

Result: a per-image ordered list of gel masks.

### 2) Fixture tracking (reference stripe)

* Inside `fixture_bounding_box`, the algorithm:

  * Converts to grayscale, equalizes contrast (CLAHE), and takes the **x-gradient** (Scharr).
  * Averages rows → a 1D column signal.
  * Smooths and finds the **first strong positive peak** (black→white edge).
* This yields `x_first` (pixel) for that frame; the script multiplies by `mm_per_pixel` to store **`Position (mm)`**.

This gives a fixture-relative “cursor” so you can analyze gel metrics as a function of fixture position, not just time.

<img width="1189" height="422" alt="Img000156_overlay" src="https://github.com/user-attachments/assets/ef252be8-8cc3-40d7-89eb-276b635e60e7" />

### 3) Point tracking & metrics

For each gel mask:

* **Centroid (px)** is computed from mask pixels; exported also in **mm**.
* **Area (px²)** from SAM → converted to **mm²** via `mm_per_pixel²`.
* **Left→Right indexing** keeps identities consistent across frames without temporal assignment (works when gels remain ordered).

### 4) Volume estimation (thickness profile + footprint)

**Goal:** Approximate **mm³** from a single 2D view.

* A simple **thickness profile** is computed from the binary mask:
  for each image column, thickness (px) = count of mask pixels.
  Convert to **mm** by multiplying by `mm_per_pixel`.
  Take the **mean thickness** across the mask’s width.

* **Footprint area (`top_area`)**:

  * If `Area (mm²)` is computed from the current mask, we can use that directly **or**
  * Use a **known nominal footprint** (`top_area_known`) if more appropriate/less noisy.

* **Volume (mm³)** ≈ `top_area * mean_thickness`
  (See `thickness_profile()` and `volume_from_profile()`.)

> This is a pragmatic side-view approximation. If you have orthogonal views or 3D recon, consider replacing `volume_from_profile()` with a better model (e.g., shape-from-silhouette or parametric geometry).

---

## Running the Main Script

```bash
python OBJECT_DETECTION.py -c config.yaml
```

### What you’ll get

* **Metrics (CSV)**

  * `metrics_dir/gel_master_table.csv` — one row per (gel, image):

    * `Img #`, `Time (ms)`, `Position (mm)` (fixture stripe), `# Gels Found`,
      `Gel #`, `Centroid X (mm)`, `Centroid Y (mm)`, `Area (mm^2)`, `Volume (mm^3)`, `Image Name`
  * `metrics_dir/gel_detection_summary.csv` — one row per image.

* **Per-gel CSVs**

  * `metrics_dir/gel_<k>_trajectory.csv` with time series for each gel.

* **Overlays**

  * `overlay_dir/<image>_overlay.png` — masks, centroids, and the red fixture stripe drawn on top.

* **Per-gel composite images**

  * `overlay_dir/per_gel_all_time_meshes/gel_<k>_all_time_meshes.png` — all timepoint silhouettes colored & blended.
 
<img width="2177" height="737" alt="image" src="https://github.com/user-attachments/assets/d093e360-c74d-4a6d-a929-0af4878cd176" />

* **Per-gel videos**

  * `overlay_dir/per_gel_videos/gel_<k>_motion_current.mp4` — evolution over time (or `trail` mode if selected).

https://github.com/user-attachments/assets/96ed037c-5353-4295-927d-4670376eaff1

* **Plots**

  * In `metrics_dir/`, subplots per gel for:

    * **vs. time**: centroid X, area (mm²), volume (mm³)
    * **vs. fixture position (mm)**: same metrics against `Position (mm)`
   
<img width="3300" height="1400" alt="image" src="https://github.com/user-attachments/assets/2ec72542-7a40-4aaf-a650-fcb7018d135e" />

<img width="3300" height="1400" alt="image" src="https://github.com/user-attachments/assets/2b16c55b-c2f0-416d-a449-c05e501aee15" />

<img width="3300" height="1400" alt="image" src="https://github.com/user-attachments/assets/de81c513-7a9e-4646-8197-7a7b3062fb1c" />

---

## Interpreting Coordinates & Units

* **Pixels → mm**: controlled by `mm_per_pixel`.
* **Centroid X/Y** are exported directly in **mm** for convenience.
* **Fixture position** is reported as **`Position (mm)`** from the detected white stripe’s x-edge.
* **Gel indexing** (`Gel #`) is **per-frame left→right**; this is robust when gels don’t cross paths horizontally.

---

## Tuning & Tips

* If you see extra blobs: **increase `min_area`** or tighten `bounding_box_meshes`.
* If a true gel gets dropped: **lower `min_area`** or **raise `max_area`**.
* Stripe not detected? Make sure `fixture_bounding_box` tightly brackets the vertical stripe and the stripe has sufficient contrast; you can reduce `min_prominence` inside `find_first_white_stripe_x()` if needed.
* If mm-scales look off: re-run **`get_mm_px.py`** and verify click points span the known **5 mm**.

---

## CLI Reference

* Detect & measure:

  ```bash
  python OBJECT_DETECTION.py -c config.yaml
  ```

* Pick ROIs:

  ```bash
  python find_bounding_box.py -c config.yaml
  # Run once for meshes, again for fixture; paste both into config.yaml
  ```

* Calibrate scale:

  ```bash
  python get_mm_px.py -c config.yaml
  # Click two points spanning the known 5 mm (or change vert_known_width)
  ```

* Estimate FPS:

  ```bash
  python get_video_metadata.py -c config.yaml
  # Paste the printed fps into config.yaml
  ```

---

## File Structure (key parts)

```
OBJECT_DETECTION.py        # main pipeline (SAM, metrics, plots, videos)
find_bounding_box.py       # click-to-define mesh/fixture ROIs (run twice)
get_mm_px.py               # click-to-measure → mm/pixel
get_video_metadata.py      # probe video → fps
load_config.py             # small YAML loader
config.yaml                # your scene configuration
```

---

## Assumptions & Limitations

* Gels remain ordered **left→right** (no overtaking); identities rely on spatial order per frame.
* Volume is an **approximation** from a side silhouette; accuracy depends on calibration, mask quality, and how representative the mean thickness is for your sample.
* The fixture stripe must be reasonably visible within its ROI for stable detection.

---

## Reproducibility

* Freeze `config.yaml` alongside outputs.
* Log the SAM checkpoint path & version.
* Save a copy of the **base image** you used for mm/pixel and bounding boxes.
* Keep the generated CSVs and overlay PNGs with your analysis.

---

## Acknowledgments

* **Segment Anything** model by Meta AI (SAM) is used for automatic mask generation.

---

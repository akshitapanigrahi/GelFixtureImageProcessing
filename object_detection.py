import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.ioff()  
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from load_config import load_config
import argparse
import sys
from matplotlib import patheffects
from pathlib import Path
import pandas as pd
import math
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import colorsys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("-c", "--config", default="config.yaml")
args = parser.parse_args()
cfg = load_config(args.config)

sys.path.append(cfg["segment_anything_dir"])
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

def find_first_white_stripe_x(img_rgb, rect, min_prominence=0.35, smooth_win=31):
    """
    Find x-coordinate (in full image coords in mm) of the first white stripe's left edge,
    detected as the first strong rising edge (black->white) inside the ROI.
    Returns None if not found.
    """
    x1, y1, x2, y2 = map(int, rect)
    roi = img_rgb[y1:y2, x1:x2]

    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (0, 0), 1.0)

    # Normalize for lighting robustness
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Emphasize vertical edges (x-gradient)
    gradx = cv2.Scharr(gray, ddepth=cv2.CV_32F, dx=1, dy=0)

    # Collapse to 1D by averaging rows (keep sign: + means black->white)
    col_signal = gradx.mean(axis=0)

    # Smooth the 1D signal
    k = max(3, smooth_win | 1)  # odd
    kernel = np.ones(k, dtype=np.float32) / k
    smooth = np.convolve(col_signal, kernel, mode='same')

    # Scale-invariant thresholding
    a = float(np.max(np.abs(smooth)) + 1e-6)
    s = smooth / a
    thr = float(min_prominence)

    # Simple local maxima detection on positive side
    # local max if greater than neighbors
    gt_prev = np.r_[False, s[1:] > s[:-1]]
    gt_next = np.r_[s[:-1] > s[1:], False]
    locmax = gt_prev & gt_next

    candidates = np.where((s > thr) & locmax)[0]
    if candidates.size == 0:
        return None

    # Leftmost rising edge
    x_roi = int(candidates[0])
    return (x1 + x_roi)

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

def filter_masks_by_box_and_area(masks, input_box=cfg["bounding_box_meshes"], min_area=cfg["min_area"], max_area=cfg["max_area"]):
    """
    Filters masks based on:
      - Bounding box being inside the input box
      - Area being within [min_area, max_area] range (optional)

    Args:
        masks (list[dict]): list of masks returned by SAM
        input_box (list[int]): [x_min, y_min, x_max, y_max]
        min_area (float or None): minimum area threshold
        max_area (float or None): maximum area threshold

    Returns:
        list[dict]: filtered masks
    """
    x_min, y_min, x_max, y_max = input_box
    filtered_masks = []

    for m in masks:
        bx, by, bw, bh = m['bbox']
        mask_x_min = bx
        mask_y_min = by
        mask_y_max = by + bh

        # Check bounding box containment
        in_box = (mask_x_min >= x_min and mask_y_min >= y_min and mask_y_max <= y_max)

        # Check area thresholds
        area_ok = True
        if min_area is not None and m['area'] < min_area:
            area_ok = False
        if max_area is not None and m['area'] > max_area:
            area_ok = False

        if in_box and area_ok:
            filtered_masks.append(m)

    return filtered_masks

def show_individual_masks_sorted(masks, image=None):
    """
    Displays each mask in its own figure, sorted left-to-right by bbox x_min.

    Args:
        masks (list[dict]): SAM mask objects.
        image (ndarray or None): Optional background image for context.
    """
    # Sort by leftmost x-coordinate
    masks_sorted = sorted(masks, key=lambda m: m['bbox'][0])
    
    for i, m in enumerate(masks_sorted):
        seg = m['segmentation']
        
        plt.figure(figsize=(8, 8))
        if image is not None:
            plt.imshow(image)
            plt.imshow(seg, alpha=0.5, cmap='jet')  # overlay
        else:
            plt.imshow(seg, cmap='gray')  # just the mask
        
        plt.title(f"Mask {i} — x_min={m['bbox'][0]}, area={m['area']}")
        plt.axis('off')
        plt.show()

from matplotlib import patheffects

def show_all_masks_edge_labeled(
    masks, 
    image=None, 
    sort_by="x", 
    alpha=0.45,
    label_format="{idx}", 
    draw_bbox=False, 
    figsize=(12, 12)
):
    """
    Overlay all masks on one image, mark centroids, and label each mask with ID + centroid coordinates.
    Show edge of vertical stripe for object tracking

    Args:
        masks (list[dict]): SAM mask dicts (each has 'segmentation', 'bbox', 'area', ...)
        image (ndarray or None): Optional background image (H,W,3) or (H,W,4)
        sort_by (str): "x" (left→right), "y" (top→bottom), or "area" (desc)
        alpha (float): Mask transparency in [0,1]
        label_format (str): Format string. Available fields:
                            idx, area, x, y, w, h, cx, cy
                            Example: "{idx} (A={area:.0f}) @({cx:.1f},{cy:.1f})"
        draw_bbox (bool): If True, draw each mask's bbox
        figsize (tuple): Matplotlib figure size
    
    Returns:
        x_first (float): x-coordinate of the first white stripe edge, if found; else
    """
    # Sort masks for consistent label order
    if sort_by == "x":
        ordered = sorted(masks, key=lambda m: m["bbox"][0])
    elif sort_by == "y":
        ordered = sorted(masks, key=lambda m: m["bbox"][1])
    elif sort_by == "area":
        ordered = sorted(masks, key=lambda m: m["area"], reverse=True)
    else:
        ordered = list(masks)

    H, W = ordered[0]["segmentation"].shape
    overlay = np.zeros((H, W, 4), dtype=float)
    cmap = plt.get_cmap("tab20", max(20, len(ordered)))

    fig, ax = plt.subplots(figsize=figsize)
    if image is not None:
        ax.imshow(image)

    label_positions = []  # (cx, cy, text, color)
    for idx, m in enumerate(ordered):
        seg = m["segmentation"]
        x, y, w, h = m["bbox"]
        color = cmap(idx % cmap.N)

        # Paint mask
        overlay[seg, :3] = color[:3]
        overlay[seg, 3] = alpha

        # --- Use your centroid_from_segmentation helper ---
        cx, cy = centroid_from_segmentation(seg)  # convert to mm coords
        cx/= cfg["mm_per_pixel"]
        cy/= cfg["mm_per_pixel"]

        # Label text, now supports {cx}, {cy}
        text = label_format.format(idx=idx, area=m["area"], x=x, y=y, w=w, h=h, cx=cx, cy=cy)
        label_positions.append((cx, cy, text, color))

    ax.imshow(overlay)

    # Optional: draw bounding boxes
    if draw_bbox:
        for m in ordered:
            x, y, w, h = m["bbox"]
            rect = plt.Rectangle((x, y), w, h, fill=False, linewidth=2, edgecolor="white")
            ax.add_patch(rect)

    # Add centroid markers + labels
    for cx, cy, text, color in label_positions:
        ax.plot(cx, cy, "o", color=color[:3], markersize=6, 
                markeredgecolor="black", markeredgewidth=1.2)  # centroid dot
        txt = ax.text(cx, cy, f"{text}\n({cx:.1f},{cy:.1f})",
                      ha="left", va="bottom", fontsize=9, color="white", weight="bold",
                      bbox=dict(facecolor="black", alpha=0.4, edgecolor="none", pad=1.5))
        txt.set_path_effects([patheffects.withStroke(linewidth=2, foreground="black")])

    x_first = find_first_white_stripe_x(image, cfg["fixture_bounding_box"], min_prominence=0.35, smooth_win=31)

    if x_first is not None:
        x1, y1, x2, y2 = map(int, cfg["fixture_bounding_box"])
        ax.add_line(Line2D([x_first, x_first], [y1, y2], linewidth=2, color="red"))

    ax.set_axis_off()
    plt.tight_layout()
    plt.show()

def centroid_from_segmentation(seg: np.ndarray):
    """
    seg: boolean mask (H, W). Returns (cx, cy) in mm coords (float).
    """
    ys, xs = np.nonzero(seg)
    mm_px = cfg["mm_per_pixel"]
    if xs.size == 0:
        return np.nan, np.nan
    return float(xs.mean())*mm_px, float(ys.mean()*mm_px)

def px_area_to_mm(area_px: float, mm_per_pixel: float | None):
    """
    Convert pixel area to mm^2
    pixel_size_mm: physical size of 1 pixel (mm/pixel).
    """
    if mm_per_pixel is None:
        return None
    return area_px * (mm_per_pixel ** 2)

import numpy as np

def thickness_profile(side_mask, mm_per_pixel):
    """
    Compute local thickness profile from side silhouette mask.

    Args:
        side_mask (ndarray): 2D binary mask (side view).
        px_to_phys (float): mm/px

    Returns:
        dict with:
          - x_positions (mm)
          - thicknesses (mm)
          - mean_thickness (mm)
    """
    side_bool = side_mask > 0

    thickness_px = side_bool.sum(axis=0)   # count 'on' pixels per column
    W_px = len(thickness_px)
    x_positions = np.arange(W_px) * mm_per_pixel

    thicknesses = thickness_px * mm_per_pixel   # convert pixel counts to physical length
    mean_thickness = np.mean(thicknesses)

    return {
        "x_positions": x_positions,
        "thicknesses": thicknesses,
        "mean_thickness": mean_thickness,
    }

def volume_from_profile(top_area, profile):
    """
    Estimate volume using the thickness profile.

    Args:
        top_area (float): known top footprint area (mm^2)
        profile (dict): output of thickness_profile()

    Returns:
        volume (mm^3)
    """
    t_vals = profile["thicknesses"]
    W_phys = profile["x_positions"][-1] - profile["x_positions"][0]
    # Average thickness over width
    mean_t = profile["mean_thickness"]

    # Volume
    return top_area * mean_t

def generate_gel_timeseries_subplots(master_df, out_dir, cols=3, plot_cm2=True):
    """
    Generate & save per-gel time-series subplots for:
      1) Centroid X (mm) vs Time
      3) Area (mm^2) vs Time (optional; requires 'Area (mm^2)')
      4) Volume (mm^3) vs Time (requires 'Volume (mm^3)')

    Args:
        master_df (pd.DataFrame): master table (one row per gel per image).
        out_dir (str|Path): Output directory where PNGs are saved.
        cols (int): Number of subplot columns.
        plot_cm2 (bool): If True, also plot area in mm^2 when available.

    Saves:
        - gel_centroid_x_over_time_subplots.png
        - gel_area_over_time_subplots_cm2.png (if plot_mm2 and data present)
        - gel_volume_over_time_subplots_cm3.png (if data present)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def _plot(metric_col, ylabel, title, filename):
        if metric_col not in master_df.columns:
            print(f"Skipping '{metric_col}': column not found.")
            return

        df = master_df[["Gel #", "Time (ms)", metric_col]].copy()
        df = df.dropna(subset=[metric_col, "Time (ms)", "Gel #"])
        if df.empty:
            print(f"Skipping '{metric_col}': no valid rows.")
            return

        df = df.sort_values(["Gel #", "Time (ms)"])
        gels = sorted(df["Gel #"].unique())
        if len(gels) == 0:
            print(f"Skipping '{metric_col}': no gels found.")
            return

        rows = math.ceil(len(gels) / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 3.5 * rows), sharex=True)
        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

        for i, g in enumerate(gels):
            ax = axes[i]
            dfg = df[df["Gel #"] == g]
            ax.plot(dfg["Time (ms)"] / 1000.0, dfg[metric_col])  # seconds on x-axis
            ax.set_title(f"Gel {int(g)}", fontsize=11)
            ax.grid(True, alpha=0.25)
            if i % cols == 0:
                ax.set_ylabel(ylabel)
            if i // cols == rows - 1:
                ax.set_xlabel("Time (s)")

        # Hide unused axes if any
        for j in range(i + 1, rows * cols):
            fig.delaxes(axes[j])

        fig.suptitle(title, y=0.995, fontsize=14)
        fig.tight_layout()
        out_path = out_dir / filename
        fig.savefig(out_path, dpi=200)
        plt.close(fig)

    # Generate the separate figures
    _plot("Centroid X (mm)", "Centroid X (mm)",
          "Gel centroid X over time (per-gel subplots)",
          "gel_centroid_x_over_time_subplots.png")

    if plot_cm2:
        _plot("Area (mm^2)", "Area (mm)",
              "Gel area over time (mm)",
              "gel_area_over_time_subplots_mm2.png")

    _plot("Volume (mm^3)", "Volume (mm)",
          "Gel volume over time (mm)",
          "gel_volume_over_time_subplots_mm3.png")

def generate_gel_xfirst_subplots(master_df, out_dir, cols=3, plot_cm2=True):
    """
    Generate & save per-gel subplots where x_first is the x-axis:
      1) Centroid X (px) vs x_first
      2) Area (cm^2) vs x_first (optional)
      3) Volume (cm^3) vs x_first

    Args:
        master_df (pd.DataFrame): Master table; must include 'Gel #', 'x_first', and target metric columns.
        out_dir (str|Path): Output directory where PNGs are saved.
        cols (int): Number of subplot columns.
        plot_cm2 (bool): If True, also plot area in cm^2 when available.

    Saves:
        - gel_centroid_vs_xfirst_subplots.png
        - gel_area_vs_xfirst_subplots_cm2.png (if plot_cm2 and data present)
        - gel_volume_vs_xfirst_subplots_cm3.png (if data present)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def _plot_xfirst(metric_col, ylabel, title, filename):
        required_cols = {"Gel #", "Position (mm)", metric_col}
        missing = [c for c in required_cols if c not in master_df.columns]
        if missing:
            print(f"Skipping '{metric_col}': missing columns: {missing}")
            return

        df = master_df[["Gel #", "Position (mm)", metric_col]].copy()
        df = df.dropna(subset=["Gel #", "Position (mm)", metric_col])
        if df.empty:
            print(f"Skipping '{metric_col}': no valid rows.")
            return

        # Sort within-gel by Position X for a clean line
        df = df.sort_values(["Gel #", "Position (mm)"])
        gels = sorted(df["Gel #"].unique())
        if len(gels) == 0:
            print(f"Skipping '{metric_col}': no gels found.")
            return

        rows = math.ceil(len(gels) / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 3.5 * rows), sharex=False)
        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

        for i, g in enumerate(gels):
            ax = axes[i]
            dfg = df[df["Gel #"] == g]
            # Plot a thin line to show the progression plus scatter to show samples
            ax.plot(dfg["Position (mm)"], dfg[metric_col], linewidth=1.0, alpha=0.8)
            ax.scatter(dfg["Position (mm)"], dfg[metric_col], s=12)

            ax.set_title(f"Gel {int(g)}", fontsize=11)
            ax.grid(True, alpha=0.25)
            if i % cols == 0:
                ax.set_ylabel(ylabel)
            if i // cols == rows - 1:
                ax.set_xlabel("Position (mm)")

        # Hide any unused axes
        for j in range(i + 1, rows * cols):
            fig.delaxes(axes[j])

        fig.suptitle(title, y=0.995, fontsize=14)
        fig.tight_layout()
        out_path = out_dir / filename
        fig.savefig(out_path, dpi=200)
        plt.close(fig)

    # Generate the separate figures
    _plot_xfirst(
        "Centroid X (mm)",
        "Centroid X (mm)",
        "Gel centroid X vs Position (per-gel subplots)",
        "gel_centroid_vs_xposition_subplots.png",
    )

    if plot_cm2:
        _plot_xfirst(
            "Area (mm^2)",
            "Area (mm^2)",
            "Gel area vs Position (mm^2)",
            "gel_area_vs_xposition_subplots_mm2.png",
        )

    _plot_xfirst(
        "Volume (mm^3)",
        "Volume (mm³)",
        "Gel volume vs Position (mm³)",
        "gel_volume_vs_xposition_subplots_mm3.png",
    )

def save_gel_metrics_csvs(master_df, out_dir):
    """
    Save one CSV per gel with its trajectory and metrics over time.
    Always includes:
        - Img #
        - Time (ms)
        - Centroid X (px)
        - Image Name
    Adds when available:
        - Centroid Y (px)
        - Area (cm^2)
        - Volume (cm^3)

    Output: <out_dir>/gel_<id>_trajectory.csv
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Required base cols
    use_cols = ["Img #", "Time (ms)", "Position (mm)", "Centroid X (mm)"]

    # Optional extras (included only if present)
    optional_cols = ["Centroid Y (mm)", "Area (mm^2)", "Volume (mm^3)", "Image Name"]
    use_cols += [c for c in optional_cols if c in master_df.columns]

    df = master_df.copy()
    df = df.dropna(subset=["Gel #", "Time (ms)"]) \
           .sort_values(["Gel #", "Time (ms)"])

    gels = sorted(df["Gel #"].dropna().unique())
    if len(gels) == 0:
        print("No gels found to export.")
        return

    for g in gels:
        dfg = df[df["Gel #"] == g][use_cols]
        out_path = out_dir / f"gel_{int(g)}_trajectory.csv"
        dfg.to_csv(out_path, index=False)

def generate_per_gel_overlay_images(
    ordered_masks_per_image,
    out_root,
    alpha=0.30,
    cmap_name="turbo",     # nice, distinct colors over long sequences
):
    """
    Create ONE PNG per gel. Each PNG shows all timepoint masks (for that gel)
    over a solid background, with a distinct color per timepoint and transparency.

    Args:
        image_files (list[Path]): same order you processed frames
        ordered_masks_per_image (list[list[dict]]): masks sorted L→R for each frame
        out_root (str|Path): output directory for per-gel overlays
        fps (float): frames per second (maps index -> seconds for legend)
        alpha (float): transparency for each mask fill
        draw_edges (bool): optionally outline each timepoint’s mesh
        edge_lw (float): line width for edges
        legend_max (int): show at most this many legend entries (subsample if needed)
        cmap_name (str): matplotlib colormap
        bgcolor (str): "black" or "white"
    """
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # Infer canvas size from first available mask/image
    H = W = None
    for ordered in ordered_masks_per_image:
        if ordered:
            H, W = ordered[0]["segmentation"].shape
            break

    bg = np.zeros((H, W, 3), dtype=np.uint8)

    # Determine max gel index seen across frames
    max_gels = 0
    for ordered in ordered_masks_per_image:
        max_gels = max(max_gels, len(ordered))

    for gel_idx in range(max_gels):
        # Collect (segmentation, frame_index) pairs for this gel across time
        segs = []
        frame_ids = []
        for i, ordered in enumerate(ordered_masks_per_image):
            if gel_idx < len(ordered):
                segs.append(ordered[gel_idx]["segmentation"])
                frame_ids.append(i)

        if not segs:
            continue

        # Set up color map with as many distinct colors as frames we have
        T = len(segs)
        cmap = plt.get_cmap(cmap_name, T)

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(bg)
        ax.set_axis_off()

        # Draw filled transparent masks, in chronological order
        for k, (seg, i) in enumerate(zip(segs, frame_ids)):
            color = cmap(k % cmap.N)  # RGBA in [0,1]
            # Build an RGBA layer for this timepoint
            rgba = np.zeros((H, W, 4), dtype=float)
            rgba[seg, :3] = color[:3]
            rgba[seg, 3] = alpha
            ax.imshow(rgba)

        fig.tight_layout()
        out_path = out_root / f"gel_{gel_idx}_all_time_meshes.png"
        fig.savefig(out_path, dpi=220, bbox_inches="tight")
        plt.close(fig)

import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import colorsys

def _hsv_bright_colors_bgr255(T, s=0.95, v=1.0, start_h=0.0):
    """
    Generate T bright, high-contrast colors by stepping hue around the wheel.
    Returns list of BGR uint8 tuples.
    """
    if T <= 0:
        return []
    hues = np.linspace(start_h, start_h + 1.0, T, endpoint=False)
    bgr_list = []
    for h in hues:
        r, g, b = colorsys.hsv_to_rgb(float(h % 1.0), float(s), float(v))
        bgr_list.append((int(b * 255), int(g * 255), int(r * 255)))
    return bgr_list

def _draw_edges_on_mask(frame_bgr, seg_bool, color_bgr=(255,255,255), edge_lw=1):
    seg_u8 = (seg_bool.astype(np.uint8) * 255)
    cnts, _ = cv2.findContours(seg_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        cv2.drawContours(frame_bgr, cnts, -1, color_bgr, edge_lw, lineType=cv2.LINE_AA)

def generate_per_gel_mp4s(
    ordered_masks_per_image,
    out_root,
    fps_out=10,             # video frame rate to WRITE
    fps_time=None,          # used for timestamps; if None, defaults to fps_out
    mode="current",         # "current" or "trail"
    alpha=0.35,
    draw_edges=True,
    edge_lw=1,
    show_original=False,    # show original image under mask
    image_files=None,       # required if show_original=True
    text=True,
    fourcc_str="mp4v",
):
    """
    Write one MP4 per gel showing shape/motion over time.

    Args:
        ordered_masks_per_image: list[list[dict]] masks (L→R per frame) with 'segmentation'
        out_root: output directory
        fps_out: output video FPS
        fps_time: FPS used to compute timestamp t=i/fps_time (defaults to fps_out)
        mode: "current" (only current timepoint) or "trail" (cumulative history)
        alpha: overlay opacity for colored fill
        draw_edges: draw thin white contours
        edge_lw: contour thickness
        show_original: if True, keep original pixels inside mask, black elsewhere
        image_files: list of paths aligned with frames (needed if show_original=True)
        text: draw "Gel k | t=... | f=..." on frames
        fourcc_str: codec fourcc for cv2.VideoWriter
    """
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # infer canvas size
    H = W = None
    for ordered in ordered_masks_per_image:
        if ordered:
            H, W = ordered[0]["segmentation"].shape
            break
    if H is None or W is None:
        print("No masks found; nothing to export.")
        return

    T = len(ordered_masks_per_image)
    if T == 0:
        print("No frames available; nothing to export.")
        return

    # bright, high-contrast colors: one per frame (no dark tones)
    time_colors_bgr = _hsv_bright_colors_bgr255(T, s=0.95, v=1.0, start_h=0.0)
    # safety fallback (shouldn't happen)
    if len(time_colors_bgr) < T:
        extra = _hsv_bright_colors_bgr255(T - len(time_colors_bgr), s=0.9, v=1.0, start_h=0.123)
        time_colors_bgr += extra
    assert len(time_colors_bgr) == T

    # gel count (max index seen across frames)
    max_gels = 0
    for ordered in ordered_masks_per_image:
        max_gels = max(max_gels, len(ordered))

    # timestamp timing
    if fps_time is None or fps_time <= 0:
        fps_time = fps_out if fps_out and fps_out > 0 else 1.0

    # text overlay settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    font_thickness = 1
    text_color = (255, 255, 255)

    for gel_idx in range(max_gels):
        # collect segmentations per frame for this gel
        segs = []
        frames_idx = []
        for i, ordered in enumerate(ordered_masks_per_image):
            if gel_idx < len(ordered):
                segs.append(ordered[gel_idx]["segmentation"])
                frames_idx.append(i)
        if not segs:
            continue

        # set up writer
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        out_path = out_root / f"gel_{gel_idx}_motion_{mode}.mp4"
        writer = cv2.VideoWriter(str(out_path), fourcc, float(fps_out), (W, H), True)
        if not writer.isOpened():
            print(f"Warning: could not open writer for {out_path}")
            continue

        trail_rgb = np.zeros((H, W, 3), dtype=np.float32)
        trail_mask = np.zeros((H, W), dtype=bool)

        for seg, i in zip(segs, frames_idx):
            # base layer (original-under-mask or black)
            if show_original:
                if image_files is None:
                    writer.release()
                    raise ValueError("show_original=True requires image_files.")
                img_bgr = cv2.imread(str(image_files[i]))  # BGR
                if img_bgr is None or img_bgr.shape[:2] != (H, W):
                    img_bgr = np.zeros((H, W, 3), dtype=np.uint8)
                base = np.zeros_like(img_bgr)
                base[seg] = img_bgr[seg]
            else:
                base = np.zeros((H, W, 3), dtype=np.uint8)

            if mode == "current":
                color_bgr = np.array(time_colors_bgr[i], dtype=np.float32)
                base_seg = base[seg].astype(np.float32)
                base[seg] = (1 - alpha) * base_seg + alpha * color_bgr
                frame_bgr = base.astype(np.uint8)
                if draw_edges:
                    _draw_edges_on_mask(frame_bgr, seg, color_bgr=(255, 255, 255), edge_lw=edge_lw)

            elif mode == "trail":
                color_bgr = np.array(time_colors_bgr[i], dtype=np.float32)
                new_pixels = seg & (~trail_mask)
                if np.any(new_pixels):
                    trail_rgb[new_pixels] = (1 - alpha) * trail_rgb[new_pixels] + alpha * color_bgr
                    trail_mask[new_pixels] = True
                frame_bgr = base.copy()
                frame_bgr[trail_mask] = np.clip(
                    0.5 * frame_bgr[trail_mask].astype(np.float32) + 0.5 * trail_rgb[trail_mask],
                    0, 255
                ).astype(np.uint8)
                if draw_edges:
                    _draw_edges_on_mask(frame_bgr, seg, color_bgr=(255, 255, 255), edge_lw=edge_lw)
            else:
                writer.release()
                raise ValueError("mode must be 'current' or 'trail'")

            if text:
                t_sec = i / float(fps_time)
                cv2.putText(frame_bgr, f"Gel {gel_idx} | t={t_sec:.2f}s | f={i}",
                            (8, 18), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

            writer.write(np.ascontiguousarray(frame_bgr))

        writer.release()
        print(f"Wrote {out_path}")

# Main function to run the SAM model and generate masks, generate metrics
def main():
    # Load the SAM model
    sam = sam_model_registry[cfg["model_type"]](checkpoint=cfg["sam_checkpoint"])
    sam.to(device="cuda" if torch.cuda.is_available() else "cpu")
    mask_generator = SamAutomaticMaskGenerator(sam)

    image_dir = Path(cfg["image_dir"])
    image_files = sorted(image_dir.glob("*.tif"))  # sort for stable Img # ordering

    # Video metadata (kept separate)
    fps = cfg["fps"]

    mm_per_pixel = cfg.get("mm_per_pixel", None)

    # Set up overlay save directory
    overlay_dir = Path(cfg.get("overlay_dir"))
    overlay_dir.mkdir(parents=True, exist_ok=True)

    # Per-image list of masks (if you still want it)
    dir_gel_masks = []

    # ---- NEW: master records (one row per gel per image)
    master_records = []

    for img_idx, image_file in enumerate(tqdm(image_files, desc="Processing images", unit="file")):
        img = cv2.imread(str(image_file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x_first = find_first_white_stripe_x(img, cfg["fixture_bounding_box"], min_prominence=0.35, smooth_win=31)*cfg["mm_per_pixel"]
        
        masks = mask_generator.generate(img)
        filtered_masks = filter_masks_by_box_and_area(masks)

        # Consistent per-image labeling: left-to-right
        ordered = sorted(filtered_masks, key=lambda m: m["bbox"][0])
        dir_gel_masks.append(ordered)

        # Time for this image (ms) from image index & fps
        time_ms = (img_idx / fps) * 1000.0

        # Number of gels in this image
        n_gels = len(ordered)

        # Build one row per gel
        for gel_idx, m in enumerate(ordered):
            seg = m["segmentation"]
            cx, cy = centroid_from_segmentation(seg)   # pixels
            area_px = float(m["area"])                 # pixels^2 (from SAM)
            area_mm2 = px_area_to_mm(area_px, mm_per_pixel)

            thickness_profile_dict = thickness_profile(seg, mm_per_pixel)
            volume = volume_from_profile(area_mm2, thickness_profile_dict)

            master_records.append({
                "Img #": img_idx,                 # zero-based index; change to +1 if you prefer 1-based
                "Time (ms)": time_ms,
                "Position (mm)": x_first,  # x-coordinate of the first white stripe edge
                "# Gels Found": n_gels,
                "Gel #": gel_idx,                 # 0, 1, 2 ... within image after L→R sort
                "Centroid X (mm)": cx,
                "Centroid Y (mm)": cy,
                # Optional real-world fields (NaN if not computable)
                "Area (mm^2)": area_mm2,
                "Volume (mm^3)": volume,
                # (Optional) Keep image name if helpful for debugging/joins
                "Image Name": image_file.name,
            })

        plt.figure(figsize=(10, 10))
        show_all_masks_edge_labeled(
            ordered,
            image=img,
            alpha=0.5,
            label_format="{idx} (A={area})",
            draw_bbox=False
        )

        plt.title(f"Image: {image_file.name}")
        plt.axis('off')
        output_path = overlay_dir / f"{image_file.stem}_overlay.png"
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
        plt.close()

    metrics_dir = Path(cfg.get("metrics_dir"))
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Convert → DataFrames
    master_df = pd.DataFrame(master_records)

    # Save CSVs
    master_csv = metrics_dir / "gel_master_table.csv"
    master_df.to_csv(master_csv, index=False)

    # If you still want the per-image summary from earlier:
    summary_df = (
        master_df
        .groupby("Img #", as_index=False)
        .agg({
            "Time (ms)": "first",       # keep first occurrence
            "Position (mm)": "first",          # add stripe position
            "# Gels Found": "first",
            "Image Name": "first"
        })
    )

    summary_csv = metrics_dir / "gel_detection_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    # After creating master_df and overlay_dir:
    generate_gel_timeseries_subplots(master_df, metrics_dir, cols=3, plot_cm2=True)
    generate_gel_xfirst_subplots(master_df, metrics_dir, cols=3, plot_cm2=True)
    save_gel_metrics_csvs(master_df, metrics_dir)
    
    # One image per gel with all timepoint meshes stacked (colored + transparent)
    per_gel_overlay_dir = Path(cfg.get("overlay_dir")) / "per_gel_all_time_meshes"
    generate_per_gel_overlay_images(
        ordered_masks_per_image=dir_gel_masks,
        out_root=per_gel_overlay_dir,
        alpha=0.30,        # tweak: higher = more opaque
        cmap_name="turbo",
    )

    per_gel_video_dir = Path(cfg.get("overlay_dir")) / "per_gel_videos"
    generate_per_gel_mp4s(
        ordered_masks_per_image=dir_gel_masks,   # make sure you appended the L→R 'ordered' list per frame
        out_root=per_gel_video_dir,
        fps_out=30,                              # output video FPS
        fps_time=cfg["fps"],              # use your source FPS for timestamps (from config)
        mode="current",                          # or "trail"
        alpha=0.35,
        draw_edges=False,
        edge_lw=1,
        show_original=False,                      # mask reveals original pixels; False -> pure silhouettes
        image_files=image_files,
        text=True,
        fourcc_str="mp4v",
    )

    return master_df, summary_df

if __name__ == "__main__":
    main()
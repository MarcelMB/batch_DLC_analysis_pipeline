"""
DLC Behavior Analysis Pipeline
Preprocesses DeepLabCut tracking data and computes behavioral metrics.
Pipeline follows placecell repo: likelihood fallback -> jump removal ->
perspective correction -> clip -> convert to cm -> compute speed.
"""

import os
import sys
import glob
import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.stdout.reconfigure(encoding="utf-8")

# ============================================================================
# CONFIGURATION — all adjustable parameters
# ============================================================================
CONFIG = {
    "fps": 30,
    "arena_size_cm": 30,
    "camera_height_mm": 480,            # 48 cm
    "tracking_height_mm": 25,           # mouse back height
    # Homography corrects camera tilt (arena shape). Radial correction is separate —
    # it accounts for mouse height above floor causing parallax shift near edges.
    "skip_perspective_correction": False,
    "jump_threshold_mm": 100,           # from placecell repo
    "speed_window_frames": 5,           # from placecell repo
    "likelihood_threshold": 0.8,
    "immobility_threshold_cms": 2.0,    # cm/s
    "max_speed_cms": 50.0,              # physiological max — flag as artifact
    "speed_pad_frames": 3,              # expand flagged speed windows by N frames
    "center_zone_fraction": 0.25,       # inner 25% of area
    "coverage_bin_size_cm": 1.0,        # 1 cm grid
    "primary_bodypart": "mouse_center",
    "fallback_bodypart": "tail_base",
    "nose_bodypart": "nose",
    "smooth_sigma_frames": 1,           # Gaussian smooth on positions (frames), 0 to disable
    # Quality flag thresholds
    "warn_fallback_pct": 40.0,         # warn if fallback > this %
    "warn_speed_cleaned_pct": 1.5,     # warn if speed cleaning > this %
    "warn_jump_pct": 0.5,              # warn if jump removal > this %
    "warn_immobility_s": 350.0,        # warn if immobility > this (seconds)
}

# ============================================================================
# PATHS
# ============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE_DIR, "DLC_tracking_summary")
PLOT_DIR = os.path.join(OUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# ============================================================================
# DATA LOADING
# ============================================================================

def load_dlc_csv(csv_path, primary_bp, fallback_bp, nose_bp):
    """Load DLC CSV and extract primary, fallback, and nose body part coordinates."""
    df = pd.read_csv(csv_path, header=[0, 1, 2], index_col=0)
    scorer = df.columns.get_level_values(0)[0]

    out = {}
    for prefix, bp in [("primary", primary_bp), ("fallback", fallback_bp), ("nose", nose_bp)]:
        out[f"{prefix}_x"] = df[(scorer, bp, "x")].values.astype(float)
        out[f"{prefix}_y"] = df[(scorer, bp, "y")].values.astype(float)
        out[f"{prefix}_lh"] = df[(scorer, bp, "likelihood")].values.astype(float)
    out["n_frames"] = len(df)
    return out


def load_arena_corners(yaml_path):
    """Load arena corner YAML and compute bounding box."""
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    corners = data["corners"]
    all_x = [corners[k][0] for k in corners]
    all_y = [corners[k][1] for k in corners]

    return {
        "x_min": min(all_x), "x_max": max(all_x),
        "y_min": min(all_y), "y_max": max(all_y),
        "corners": corners,
    }

# ============================================================================
# PREPROCESSING PIPELINE
# ============================================================================

def apply_likelihood_fallback(primary_x, primary_y, primary_lh,
                              fallback_x, fallback_y, fallback_lh,
                              nose_x, nose_y, nose_lh,
                              threshold):
    """Tiered fallback for low-confidence mouse_center frames.

    Priority:
    1. mouse_center good (>threshold) → use as-is
    2. mouse_center bad, nose AND tail_base good → midpoint(nose, tail_base)
    3. mouse_center bad, only tail_base good → tail_base + median offset
    4. everything bad → keep mouse_center (rare)
    """
    x = primary_x.copy()
    y = primary_y.copy()
    n = len(x)

    low_conf = primary_lh < threshold

    # Compute median offset from tail_base→center on frames where all 3 are good
    all_good = (primary_lh > threshold) & (fallback_lh > threshold)
    if all_good.sum() > 0:
        offset_x = np.median(primary_x[all_good] - fallback_x[all_good])
        offset_y = np.median(primary_y[all_good] - fallback_y[all_good])
    else:
        offset_x, offset_y = 0.0, 0.0

    # Track which fallback method was used (for plotting)
    fallback_method = np.zeros(n, dtype=int)  # 0=primary, 1=midpoint, 2=offset

    for i in range(n):
        if not low_conf[i]:
            continue  # primary is good
        if nose_lh[i] > threshold and fallback_lh[i] > threshold:
            # Tier 1: midpoint of nose and tail_base
            x[i] = (nose_x[i] + fallback_x[i]) / 2.0
            y[i] = (nose_y[i] + fallback_y[i]) / 2.0
            fallback_method[i] = 1
        elif fallback_lh[i] > threshold:
            # Tier 2: tail_base + median offset
            x[i] = fallback_x[i] + offset_x
            y[i] = fallback_y[i] + offset_y
            fallback_method[i] = 2
        else:
            # Tier 3: keep original (nothing reliable)
            fallback_method[i] = 0

    return x, y, low_conf, fallback_method


def remove_position_jumps(x, y, threshold_px):
    """Flag frame-to-frame jumps > threshold, replace with interpolated values."""
    dx = np.diff(x)
    dy = np.diff(y)
    dist = np.sqrt(dx**2 + dy**2)

    bad = np.zeros(len(x), dtype=bool)
    bad[1:] = dist > threshold_px
    n_bad = int(bad.sum())

    x_clean = x.copy()
    y_clean = y.copy()
    if n_bad > 0:
        x_clean[bad] = np.nan
        y_clean[bad] = np.nan
        x_clean = pd.Series(x_clean).interpolate(limit_direction="both").values
        y_clean = pd.Series(y_clean).interpolate(limit_direction="both").values

    return x_clean, y_clean, bad


def clean_by_speed(x, y, fps, window, max_speed_cms, pad_frames,
                   arena_bounds, arena_size_cm):
    """Flag frames where speed exceeds physiological max, expand window, interpolate.

    Works in cm space internally to compare against max_speed_cms, but operates
    on pixel coordinates via temporary conversion.
    """
    # Temporary conversion to cm for speed check
    scale_x = arena_size_cm / (arena_bounds["x_max"] - arena_bounds["x_min"])
    scale_y = arena_size_cm / (arena_bounds["y_max"] - arena_bounds["y_min"])
    x_cm = (x - arena_bounds["x_min"]) * scale_x
    y_cm = (y - arena_bounds["y_min"]) * scale_y

    speed = compute_speed(x_cm, y_cm, fps, window)
    bad = speed > max_speed_cms

    # Expand bad regions by pad_frames in each direction
    if bad.any():
        bad_expanded = bad.copy()
        for shift in range(1, pad_frames + 1):
            bad_expanded[shift:] |= bad[:-shift]
            bad_expanded[:-shift] |= bad[shift:]
        bad = bad_expanded

    n_bad = int(bad.sum())
    x_clean = x.copy()
    y_clean = y.copy()
    if n_bad > 0:
        x_clean[bad] = np.nan
        y_clean[bad] = np.nan
        x_clean = pd.Series(x_clean).interpolate(limit_direction="both").values
        y_clean = pd.Series(y_clean).interpolate(limit_direction="both").values

    return x_clean, y_clean, bad


def correct_perspective(x, y, arena_bounds, camera_height_mm, tracking_height_mm):
    """Radial perspective correction from arena center."""
    cx = (arena_bounds["x_min"] + arena_bounds["x_max"]) / 2.0
    cy = (arena_bounds["y_min"] + arena_bounds["y_max"]) / 2.0
    factor = (camera_height_mm - tracking_height_mm) / camera_height_mm

    x_corr = cx + (x - cx) * factor
    y_corr = cy + (y - cy) * factor
    return x_corr, y_corr


def correct_arena_bounds(arena_bounds, camera_height_mm, tracking_height_mm):
    """Apply the same radial correction to the arena bounds themselves."""
    cx = (arena_bounds["x_min"] + arena_bounds["x_max"]) / 2.0
    cy = (arena_bounds["y_min"] + arena_bounds["y_max"]) / 2.0
    factor = (camera_height_mm - tracking_height_mm) / camera_height_mm

    return {
        "x_min": cx + (arena_bounds["x_min"] - cx) * factor,
        "x_max": cx + (arena_bounds["x_max"] - cx) * factor,
        "y_min": cy + (arena_bounds["y_min"] - cy) * factor,
        "y_max": cy + (arena_bounds["y_max"] - cy) * factor,
        "corners": arena_bounds.get("corners"),
    }


def clip_to_arena(x, y, arena_bounds):
    """Clamp coordinates to arena bounding box."""
    x_clip = np.clip(x, arena_bounds["x_min"], arena_bounds["x_max"])
    y_clip = np.clip(y, arena_bounds["y_min"], arena_bounds["y_max"])
    return x_clip, y_clip


def smooth_positions(x, y, sigma_frames):
    """Apply Gaussian smoothing to position coordinates."""
    if sigma_frames <= 0:
        return x.copy(), y.copy()
    from scipy.ndimage import gaussian_filter1d
    return gaussian_filter1d(x, sigma=sigma_frames), gaussian_filter1d(y, sigma=sigma_frames)


def convert_to_cm(x, y, arena_bounds, arena_size_cm):
    """Convert pixel coordinates to cm."""
    scale_x = arena_size_cm / (arena_bounds["x_max"] - arena_bounds["x_min"])
    scale_y = arena_size_cm / (arena_bounds["y_max"] - arena_bounds["y_min"])
    x_cm = (x - arena_bounds["x_min"]) * scale_x
    y_cm = (y - arena_bounds["y_min"]) * scale_y
    return x_cm, y_cm, scale_x, scale_y

# ============================================================================
# DERIVED METRICS
# ============================================================================

def compute_speed(x_cm, y_cm, fps, window):
    """Windowed speed in cm/s."""
    n = len(x_cm)
    speed = np.zeros(n)
    for i in range(n):
        end = min(i + window, n - 1)
        if end == i:
            speed[i] = speed[i - 1] if i > 0 else 0.0
        else:
            d = np.sqrt((x_cm[end] - x_cm[i])**2 + (y_cm[end] - y_cm[i])**2)
            dt = (end - i) / fps
            speed[i] = d / dt
    return speed


def compute_total_distance(x_cm, y_cm):
    """Sum of frame-to-frame Euclidean distances in cm."""
    dx = np.diff(x_cm)
    dy = np.diff(y_cm)
    return float(np.sum(np.sqrt(dx**2 + dy**2)))


def compute_center_zone_distance(x_cm, y_cm, arena_size_cm, center_fraction):
    """Total distance traveled while inside the center zone."""
    side = arena_size_cm * np.sqrt(center_fraction)
    margin = (arena_size_cm - side) / 2.0
    c_min, c_max = margin, arena_size_cm - margin

    in_center = (x_cm >= c_min) & (x_cm <= c_max) & (y_cm >= c_min) & (y_cm <= c_max)
    # Use frames where BOTH current and next are in center
    both_in = in_center[:-1] & in_center[1:]
    dx = np.diff(x_cm)
    dy = np.diff(y_cm)
    dists = np.sqrt(dx**2 + dy**2)
    return float(np.sum(dists[both_in]))


def compute_arena_coverage(x_cm, y_cm, arena_size_cm, bin_size_cm):
    """Percentage of arena bins visited."""
    n_bins = int(np.ceil(arena_size_cm / bin_size_cm))
    bins_x = np.clip((x_cm / bin_size_cm).astype(int), 0, n_bins - 1)
    bins_y = np.clip((y_cm / bin_size_cm).astype(int), 0, n_bins - 1)
    visited = set(zip(bins_x, bins_y))
    return float(len(visited) / (n_bins * n_bins) * 100)


def compute_immobility_time(speed, fps, threshold_cms):
    """Time in seconds where speed < threshold."""
    return float(np.sum(speed < threshold_cms) / fps)

# ============================================================================
# PLOTTING — per-animal pipeline figure
# ============================================================================

def draw_arena_rect(ax, x_min, x_max, y_min, y_max, color="black", ls="-"):
    """Draw arena boundary rectangle on axis."""
    rect = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                      linewidth=1.5, edgecolor=color, facecolor="none", linestyle=ls)
    ax.add_patch(rect)


def plot_pipeline_figure(stages, arena_bounds, config, metrics, meta, save_path):
    """Create 8-panel pipeline figure for one animal."""
    fig = plt.figure(figsize=(24, 24))
    # 4 rows: row0=3, row1=3, row2=final cm + quality, row3=speed
    gs = fig.add_gridspec(4, 3, height_ratios=[1, 1, 1, 0.4], hspace=0.30, wspace=0.30)

    arena_size = config["arena_size_cm"]
    xmn, xmx = arena_bounds["x_min"], arena_bounds["x_max"]
    ymn, ymx = arena_bounds["y_min"], arena_bounds["y_max"]
    cb = stages["corrected_bounds"]

    # Fixed pixel view: 0–800 for all pixel panels
    px_view_min, px_view_max = 0, 800
    px_ticks = np.arange(px_view_min, px_view_max + 1, 100)

    def setup_px_ax_padded(ax, title):
        """Panels 1-5: fixed 0-800 view with arena rectangle drawn inside."""
        ax.set_xlim(px_view_min, px_view_max)
        ax.set_ylim(px_view_min, px_view_max)
        ax.set_xticks(px_ticks)
        ax.set_yticks(px_ticks)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("x (px)")
        ax.set_ylabel("y (px)")
        draw_arena_rect(ax, xmn, xmx, ymn, ymx, color="black", ls="--")

    def setup_px_ax_tight(ax, title):
        """Panel 6: fixed 0-800 view."""
        ax.set_xlim(px_view_min, px_view_max)
        ax.set_ylim(px_view_min, px_view_max)
        ax.set_xticks(px_ticks)
        ax.set_yticks(px_ticks)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("x (px)")
        ax.set_ylabel("y (px)")

    lw, alpha = 0.3, 0.5

    # Panel (0,0): Raw trajectory
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(stages["raw_x"], stages["raw_y"], linewidth=lw, alpha=alpha, color="C0")
    setup_px_ax_padded(ax, "1. Raw trajectory")

    # Panel (0,1): After likelihood fallback
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(stages["fb_x"], stages["fb_y"], linewidth=lw, alpha=alpha, color="C0")
    fb_method = stages["fallback_method"]
    midpoint_mask = fb_method == 1
    offset_mask = fb_method == 2
    if midpoint_mask.any():
        ax.scatter(stages["fb_x"][midpoint_mask], stages["fb_y"][midpoint_mask],
                   s=1, c="green", alpha=0.3,
                   label=f"midpoint(nose,tail) ({midpoint_mask.sum()})")
    if offset_mask.any():
        ax.scatter(stages["fb_x"][offset_mask], stages["fb_y"][offset_mask],
                   s=1, c="red", alpha=0.3,
                   label=f"tail+offset ({offset_mask.sum()})")
    ax.legend(fontsize=6, loc="upper right")
    setup_px_ax_padded(ax, "2. After likelihood fallback")

    # Panel (0,2): After jump removal
    ax = fig.add_subplot(gs[0, 2])
    ax.plot(stages["jr_x"], stages["jr_y"], linewidth=lw, alpha=alpha, color="C0")
    jmask = stages["jump_mask"]
    if jmask.any():
        ax.scatter(stages["fb_x"][jmask], stages["fb_y"][jmask],
                   s=4, c="red", alpha=0.6, zorder=3,
                   label=f"jumps ({jmask.sum()} fr)")
        ax.legend(fontsize=7, loc="upper right")
    setup_px_ax_padded(ax, "3. After jump removal")

    # Panel (1,0): After speed-based cleaning
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(stages["sc_x"], stages["sc_y"], linewidth=lw, alpha=alpha, color="C0")
    smask = stages["speed_clean_mask"]
    if smask.any():
        ax.scatter(stages["jr_x"][smask], stages["jr_y"][smask],
                   s=2, c="orange", alpha=0.5, zorder=3,
                   label=f"speed artifacts ({smask.sum()} fr)")
        ax.legend(fontsize=7, loc="upper right")
    setup_px_ax_padded(ax, f"4. After speed cleaning (>{config['max_speed_cms']} cm/s)")

    # Panel (1,1): After perspective correction
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(stages["pc_x"], stages["pc_y"], linewidth=lw, alpha=alpha, color="C0")
    setup_px_ax_padded(ax, "5. After perspective correction")
    # Also show corrected bounds in red
    draw_arena_rect(ax, cb["x_min"], cb["x_max"], cb["y_min"], cb["y_max"],
                    color="red", ls="-")

    # Panel (1,2): After clipping
    ax = fig.add_subplot(gs[1, 2])
    ax.plot(stages["clip_x"], stages["clip_y"], linewidth=lw, alpha=alpha, color="C0")
    setup_px_ax_tight(ax, "6. After clipping")

    # Panel (2,0-1): Final trajectory in cm + center zone
    ax = fig.add_subplot(gs[2, 0:2])
    ax.plot(stages["cm_x"], stages["cm_y"], linewidth=lw, alpha=alpha, color="C0")
    ax.set_xlim(0, arena_size)
    ax.set_ylim(0, arena_size)
    ax.set_aspect("equal")
    ax.set_title("7. Final trajectory (cm)", fontsize=10)
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")
    side = arena_size * np.sqrt(config["center_zone_fraction"])
    margin = (arena_size - side) / 2.0
    draw_arena_rect(ax, margin, arena_size - margin, margin, arena_size - margin,
                    color="red", ls="--")
    ax.text(arena_size / 2, arena_size - margin + 0.3, "center zone",
            ha="center", fontsize=7, color="red")

    # Panel (2,2): Quality summary box
    ax = fig.add_subplot(gs[2, 2])
    ax.axis("off")
    n = metrics["n_frames"]
    checks = [
        ("Frames", f"{n}", False),
        ("Duration", f"{n / config['fps'] / 60:.1f} min", False),
        ("", "", False),
        ("Fallback total", f"{metrics['fallback_frames_pct']:.1f}%",
         metrics["fallback_frames_pct"] > config["warn_fallback_pct"]),
        ("  midpoint", f"{metrics['fallback_midpoint_pct']:.1f}%", False),
        ("  tail+offset", f"{metrics['fallback_offset_pct']:.1f}%",
         metrics["fallback_offset_pct"] > 15.0),
        ("Jump removal", f"{metrics['jump_frames_pct']:.2f}%",
         metrics["jump_frames_pct"] > config["warn_jump_pct"]),
        ("Speed clean", f"{metrics['speed_cleaned_pct']:.2f}%",
         metrics["speed_cleaned_pct"] > config["warn_speed_cleaned_pct"]),
        ("", "", False),
        ("Distance", f"{metrics['total_distance_cm']:.0f} cm", False),
        ("Med speed", f"{metrics['median_speed_cms']:.2f} cm/s", False),
        ("Center dist", f"{metrics['center_distance_cm']:.0f} cm", False),
        ("Coverage", f"{metrics['arena_coverage_pct']:.1f}%", False),
        ("Immobility", f"{metrics['immobility_time_s']:.1f} s",
         metrics["immobility_time_s"] > config["warn_immobility_s"]),
    ]
    any_warning = any(bad for _, _, bad in checks)
    box_color = "#ffe0e0" if any_warning else "lightyellow"
    ax.add_patch(plt.Rectangle((0.02, 0.02), 0.96, 0.96, transform=ax.transAxes,
                                facecolor=box_color, edgecolor="gray",
                                linewidth=1, alpha=0.9, zorder=0, clip_on=False))
    y_pos = 0.93
    line_h = 0.06
    if any_warning:
        ax.text(0.5, 0.97, "! QUALITY WARNINGS !",
                transform=ax.transAxes, fontsize=11, ha="center",
                fontweight="bold", color="red")
        y_pos = 0.88
    for label, val, is_bad in checks:
        if label == "" and val == "":
            y_pos -= line_h * 0.4
            continue
        color = "red" if is_bad else "black"
        weight = "bold" if is_bad else "normal"
        marker = " !!!" if is_bad else ""
        ax.text(0.08, y_pos, f"{label}: {val}{marker}",
                transform=ax.transAxes, fontsize=9, verticalalignment="top",
                fontfamily="monospace", color=color, fontweight=weight)
        y_pos -= line_h

    # Panel (3, spanning): Speed trace
    ax = fig.add_subplot(gs[3, :])
    t = np.arange(len(stages["speed"])) / config["fps"]
    ax.plot(t, stages["speed"], linewidth=0.3, alpha=0.6, color="C0")
    ax.axhline(config["immobility_threshold_cms"], color="red", ls="--", lw=1,
               label=f"immobility ({config['immobility_threshold_cms']} cm/s)")
    ax.axhline(config["max_speed_cms"], color="orange", ls="--", lw=1,
               label=f"max speed ({config['max_speed_cms']} cm/s)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Speed (cm/s)")
    ax.set_title("Speed trace (after all cleaning)", fontsize=10)
    ax.legend(fontsize=7)
    ax.set_xlim(0, t[-1])

    fig.suptitle(
        f"{meta['day']} | {meta['animal']}",
        fontsize=13, y=0.99, fontweight="bold"
    )

    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

# ============================================================================
# PLOTTING — summary boxplots
# ============================================================================

def plot_summary_boxplots(results_df, save_path):
    """Boxplots with color-coded animal data points and legend."""
    # (column, label, unit_transform, y_label, ylim)
    metrics_to_plot = [
        ("total_distance_cm", "Total Distance", lambda x: x / 100, "Distance (m)", (0, 200)),
        ("median_speed_cms", "Median Speed", lambda x: x, "Speed (cm/s)", (0, 8)),
        ("center_distance_cm", "Distance in Center", lambda x: x / 100, "Distance (m)", (0, 150)),
        ("arena_coverage_pct", "Arena Coverage", lambda x: x, "Coverage (%)", (0, 100)),
        ("immobility_time_s", "Immobility Time", lambda x: x / 60, "Time (min)", None),
    ]
    days = sorted(results_df["Day"].unique())
    animals = sorted(results_df["Animal"].unique())

    # Assign a distinct color and marker to each animal
    animal_colors = {}
    animal_markers = {}
    cmap_colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628"]
    markers = ["o", "s", "D", "^", "v", "P"]
    for i, a in enumerate(animals):
        animal_colors[a] = cmap_colors[i % len(cmap_colors)]
        animal_markers[a] = markers[i % len(markers)]

    fig, axes = plt.subplots(1, 5, figsize=(24, 5.5))
    for ax, (col, title, transform, ylabel, ylim) in zip(axes, metrics_to_plot):
        data_by_day = [transform(results_df[results_df["Day"] == d][col].values)
                       for d in days]

        ax.boxplot(data_by_day, tick_labels=days, widths=0.5,
                   patch_artist=True, showfliers=False,
                   boxprops=dict(facecolor="lightblue", alpha=0.4),
                   medianprops=dict(color="black", linewidth=1.5))

        # Overlay individual animal points, color-coded
        for i, d in enumerate(days):
            day_df = results_df[results_df["Day"] == d].sort_values("Animal")
            n_animals = len(day_df)
            offsets = np.linspace(-0.15, 0.15, n_animals)
            for j, (_, row) in enumerate(day_df.iterrows()):
                animal = row["Animal"]
                val = transform(np.array([row[col]]))[0]
                ax.scatter(i + 1 + offsets[j], val,
                           c=animal_colors[animal], marker=animal_markers[animal],
                           s=50, alpha=0.9, edgecolors="black", linewidths=0.5,
                           zorder=3)

        ax.set_ylabel(ylabel)
        ax.set_xlabel("Day")
        ax.set_title(title, fontsize=10)

    # Set ylims AFTER all plotting is done (scatter can trigger autoscale)
    for ax, (col, title, transform, ylabel, ylim) in zip(axes, metrics_to_plot):
        if ylim is not None:
            ax.set_ylim(ylim)
        else:
            all_vals = np.concatenate([transform(results_df[col].values)])
            ax.set_ylim(0, all_vals.max() * 1.15)

    # Create shared legend on the right
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker=animal_markers[a], color="w",
               markerfacecolor=animal_colors[a], markeredgecolor="black",
               markersize=8, label=a)
        for a in animals
    ]
    fig.legend(handles=legend_handles, loc="center right",
               title="Animal", fontsize=9, title_fontsize=10,
               bbox_to_anchor=(1.0, 0.5))

    fig.suptitle("Behavior Metrics Summary (D1-D5)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 0.93, 0.95])
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

# ============================================================================
# MAIN PROCESSING
# ============================================================================

def process_one_animal(csv_path, arena_bounds, config):
    """Run full pipeline on one animal. Returns (metrics_dict, stages_dict)."""
    data = load_dlc_csv(csv_path, config["primary_bodypart"],
                        config["fallback_bodypart"], config["nose_bodypart"])
    stages = {}

    # Raw
    stages["raw_x"] = data["primary_x"].copy()
    stages["raw_y"] = data["primary_y"].copy()

    # Step 0: Likelihood fallback (tiered: midpoint(nose,tail) > tail+offset)
    fb_x, fb_y, fb_mask, fb_method = apply_likelihood_fallback(
        data["primary_x"], data["primary_y"], data["primary_lh"],
        data["fallback_x"], data["fallback_y"], data["fallback_lh"],
        data["nose_x"], data["nose_y"], data["nose_lh"],
        config["likelihood_threshold"],
    )
    stages["fb_x"], stages["fb_y"] = fb_x, fb_y
    stages["fallback_mask"] = fb_mask
    stages["fallback_method"] = fb_method

    # Convert jump threshold from mm to px
    avg_span_px = ((arena_bounds["x_max"] - arena_bounds["x_min"]) +
                   (arena_bounds["y_max"] - arena_bounds["y_min"])) / 2.0
    px_per_mm = avg_span_px / (config["arena_size_cm"] * 10)
    jump_threshold_px = config["jump_threshold_mm"] * px_per_mm

    # Step 1: Jump removal
    jr_x, jr_y, jmask = remove_position_jumps(fb_x, fb_y, jump_threshold_px)
    stages["jr_x"], stages["jr_y"], stages["jump_mask"] = jr_x, jr_y, jmask

    # Step 1b: Speed-based cleaning (catch multi-frame teleportation)
    sc_x, sc_y, speed_mask = clean_by_speed(
        jr_x, jr_y, config["fps"], config["speed_window_frames"],
        config["max_speed_cms"], config["speed_pad_frames"],
        arena_bounds, config["arena_size_cm"],
    )
    stages["sc_x"], stages["sc_y"], stages["speed_clean_mask"] = sc_x, sc_y, speed_mask

    # Step 2: Perspective correction
    if config.get("skip_perspective_correction", False):
        pc_x, pc_y = sc_x.copy(), sc_y.copy()
        corrected_bounds = arena_bounds
    else:
        pc_x, pc_y = correct_perspective(sc_x, sc_y, arena_bounds,
                                         config["camera_height_mm"],
                                         config["tracking_height_mm"])
        # Apply same correction to arena bounds
        corrected_bounds = correct_arena_bounds(arena_bounds,
                                                config["camera_height_mm"],
                                                config["tracking_height_mm"])
    stages["pc_x"], stages["pc_y"] = pc_x, pc_y
    stages["corrected_bounds"] = corrected_bounds

    # Step 3: Clip to corrected bounds
    clip_x, clip_y = clip_to_arena(pc_x, pc_y, corrected_bounds)
    stages["clip_x"], stages["clip_y"] = clip_x, clip_y

    # Step 3b: Gaussian smooth (reduces frame-to-frame tracking jitter)
    sm_x, sm_y = smooth_positions(clip_x, clip_y, config["smooth_sigma_frames"])
    # Re-clip after smoothing
    sm_x, sm_y = clip_to_arena(sm_x, sm_y, corrected_bounds)
    stages["sm_x"], stages["sm_y"] = sm_x, sm_y

    # Step 4: Convert to cm using corrected bounds
    cm_x, cm_y, sx, sy = convert_to_cm(sm_x, sm_y, corrected_bounds,
                                        config["arena_size_cm"])
    stages["cm_x"], stages["cm_y"] = cm_x, cm_y

    # Step 5: Speed
    speed = compute_speed(cm_x, cm_y, config["fps"], config["speed_window_frames"])
    stages["speed"] = speed

    # Fallback stats
    n_midpoint = int((fb_method == 1).sum())
    n_offset = int((fb_method == 2).sum())

    # Metrics
    metrics = {
        "total_distance_cm": compute_total_distance(cm_x, cm_y),
        "median_speed_cms": float(np.median(speed)),
        "center_distance_cm": compute_center_zone_distance(
            cm_x, cm_y, config["arena_size_cm"], config["center_zone_fraction"]),
        "arena_coverage_pct": compute_arena_coverage(
            cm_x, cm_y, config["arena_size_cm"], config["coverage_bin_size_cm"]),
        "immobility_time_s": compute_immobility_time(
            speed, config["fps"], config["immobility_threshold_cms"]),
        "fallback_frames_pct": float(fb_mask.sum() / len(fb_mask) * 100),
        "fallback_midpoint_pct": float(n_midpoint / len(fb_mask) * 100),
        "fallback_offset_pct": float(n_offset / len(fb_mask) * 100),
        "jump_frames_pct": float(jmask.sum() / len(jmask) * 100),
        "speed_cleaned_pct": float(speed_mask.sum() / len(speed_mask) * 100),
        "n_frames": data["n_frames"],
    }

    return metrics, stages


def main():
    days = sorted([d for d in os.listdir(BASE_DIR) if d.endswith("_DLC")])
    all_metrics = []

    for day_folder in days:
        day = day_folder.replace("_DLC", "")
        day_dir = os.path.join(BASE_DIR, day_folder)

        yaml_path = os.path.join(day_dir, f"arena_corners_{day}.yaml")
        arena_bounds = load_arena_corners(yaml_path)

        animal_dirs = sorted(glob.glob(os.path.join(day_dir, "M*")))
        for animal_dir in animal_dirs:
            if not os.path.isdir(animal_dir):
                continue
            animal = os.path.basename(animal_dir)
            csv_files = glob.glob(os.path.join(animal_dir, "*.csv"))
            if not csv_files:
                continue
            csv_path = csv_files[0]

            print(f"Processing {day} | {animal} ...")
            metrics, stages = process_one_animal(csv_path, arena_bounds, CONFIG)
            metrics["Day"] = day
            metrics["Animal"] = animal
            all_metrics.append(metrics)

            # Pipeline plot
            plot_path = os.path.join(PLOT_DIR, f"{day}_{animal}_pipeline.png")
            plot_pipeline_figure(stages, arena_bounds, CONFIG, metrics,
                                {"day": day, "animal": animal}, plot_path)
            print(f"  -> saved {plot_path}")

    # Build results table
    results = pd.DataFrame(all_metrics)
    col_order = ["Day", "Animal", "n_frames", "total_distance_cm", "median_speed_cms",
                 "center_distance_cm", "arena_coverage_pct", "immobility_time_s",
                 "fallback_frames_pct", "fallback_midpoint_pct", "fallback_offset_pct",
                 "jump_frames_pct", "speed_cleaned_pct"]
    results = results[col_order]

    # Save CSV
    csv_out = os.path.join(OUT_DIR, "behavior_metrics.csv")
    results.to_csv(csv_out, index=False)
    print(f"\nCSV saved: {csv_out}")

    # Summary boxplot
    summary_path = os.path.join(PLOT_DIR, "summary_boxplots.png")
    plot_summary_boxplots(results, summary_path)
    print(f"Summary plot saved: {summary_path}")

    # Console summary
    print("\n" + "=" * 110)
    print("BEHAVIOR METRICS SUMMARY")
    print("=" * 110)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 150)
    pd.set_option("display.float_format", lambda x: f"{x:.2f}")
    print(results.to_string(index=False))
    print()


if __name__ == "__main__":
    main()

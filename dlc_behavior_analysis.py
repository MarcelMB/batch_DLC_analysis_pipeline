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
    "likelihood_threshold": 0.6,
    "immobility_threshold_cms": 2.0,    # cm/s
    "max_speed_cms": 50.0,              # physiological max — flag as artifact
    "speed_pad_frames": 3,              # expand flagged speed windows by N frames
    "center_zone_fraction": 0.20,       # inner 20% of area
    "coverage_bin_size_cm": 1.0,        # 1 cm grid
    "primary_bodypart": "mouse_center",
    "fallback_bodypart": "tail_base",
    "nose_bodypart": "nose",
    "smooth_sigma_frames": 2,           # Gaussian smooth on positions (frames), 0 to disable
    # Quality flag thresholds
    "warn_fallback_pct": 40.0,         # warn if fallback > this %
    "warn_speed_cleaned_pct": 1.5,     # warn if speed cleaning > this %
    "warn_jump_pct": 0.5,              # warn if jump removal > this %
    "warn_outlier_n_sd": 2,             # flag behavioral metrics beyond median ± N SD
}

# ============================================================================
# PATHS — edit BASE_DIR to point to the folder containing D1_DLC, D2_DLC, etc.
# ============================================================================
BASE_DIR = r"D:\marcel\Miniscope_Zero_2025_11\v4_vs_MSzero_behavior"
OUT_DIR  = os.path.join(BASE_DIR, "DLC_tracking_summary")
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
                   arena_bounds, arena_size_cm, max_passes=10):
    """Flag frames where speed exceeds physiological max, expand window, interpolate.

    Runs iteratively: after interpolating flagged frames, recompute speed and
    check again. This catches entire teleportation episodes where the mouse
    jumps to a wrong location, sits there (low speed), then jumps back —
    each pass peels off the edges until the whole episode is removed.
    """
    scale_x = arena_size_cm / (arena_bounds["x_max"] - arena_bounds["x_min"])
    scale_y = arena_size_cm / (arena_bounds["y_max"] - arena_bounds["y_min"])

    x_clean = x.copy()
    y_clean = y.copy()
    all_bad = np.zeros(len(x), dtype=bool)

    for pass_i in range(max_passes):
        x_cm = (x_clean - arena_bounds["x_min"]) * scale_x
        y_cm = (y_clean - arena_bounds["y_min"]) * scale_y

        speed = compute_speed(x_cm, y_cm, fps, window)
        bad = speed > max_speed_cms

        if not bad.any():
            break

        # Expand bad regions by pad_frames in each direction
        bad_expanded = bad.copy()
        for shift in range(1, pad_frames + 1):
            bad_expanded[shift:] |= bad[:-shift]
            bad_expanded[:-shift] |= bad[shift:]
        bad = bad_expanded

        all_bad |= bad
        x_clean[bad] = np.nan
        y_clean[bad] = np.nan
        x_clean = pd.Series(x_clean).interpolate(limit_direction="both").values
        y_clean = pd.Series(y_clean).interpolate(limit_direction="both").values

    return x_clean, y_clean, all_bad


def remove_out_of_arena_episodes(x, y, arena_bounds, margin_frac=0.05):
    """Find frames far outside the arena and remove entire excursion episodes.

    When DLC tracks to a wrong location, the mouse drifts out slowly over many
    frames, sits there, then drifts back. Individual frames near the arena edge
    look fine, but the episode as a whole is bad. This function:
    1. Finds any frame clearly outside the arena (using a tight 5% margin)
    2. For each such frame, expands backward and forward to find where the
       trajectory left the arena (using a generous 15% margin), capturing
       the entire drift episode
    3. Interpolates the full episode
    """
    span_x = arena_bounds["x_max"] - arena_bounds["x_min"]
    span_y = arena_bounds["y_max"] - arena_bounds["y_min"]

    # Tight bounds to detect clearly-outside frames
    tight_mx = span_x * margin_frac
    tight_my = span_y * margin_frac
    outside = ((x < arena_bounds["x_min"] - tight_mx) |
               (x > arena_bounds["x_max"] + tight_mx) |
               (y < arena_bounds["y_min"] - tight_my) |
               (y > arena_bounds["y_max"] + tight_my))

    if not outside.any():
        return x.copy(), y.copy(), np.zeros(len(x), dtype=bool)

    # Generous bounds to find where the episode starts/ends
    gen_frac = 0.15
    gen_mx = span_x * gen_frac
    gen_my = span_y * gen_frac
    inside_generous = ((x >= arena_bounds["x_min"] - gen_mx) &
                       (x <= arena_bounds["x_max"] + gen_mx) &
                       (y >= arena_bounds["y_min"] - gen_my) &
                       (y <= arena_bounds["y_max"] + gen_my))

    # For each outside frame, expand to the full episode
    bad = np.zeros(len(x), dtype=bool)
    n = len(x)
    for i in np.where(outside)[0]:
        if bad[i]:
            continue  # already part of a flagged episode
        # Expand backward until we find a frame solidly inside
        start = i
        while start > 0 and not inside_generous[start - 1]:
            start -= 1
        # Expand forward
        end = i
        while end < n - 1 and not inside_generous[end + 1]:
            end += 1
        bad[start:end + 1] = True

    x_clean = x.copy()
    y_clean = y.copy()
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


def plot_pipeline_figure(stages, arena_bounds, config, metrics, meta, save_path,
                         pop_bounds=None):
    """Create 9-panel pipeline figure for one animal."""
    fig = plt.figure(figsize=(24, 30))
    # 5 rows: row0-2=pixel panels, row3=final cm + quality, row4=speed
    gs = fig.add_gridspec(5, 3, height_ratios=[1, 1, 1, 1, 0.4], hspace=0.30, wspace=0.30)

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

    lw, alpha = 0.3, 0.25

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
    setup_px_ax_padded(ax, f"4. After speed cleaning (>{config['max_speed_cms']} cm/s, iterative)")

    # Panel (1,1): After out-of-arena episode removal
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(stages["oa_x"], stages["oa_y"], linewidth=lw, alpha=alpha, color="C0")
    oa_mask = stages["out_of_arena_mask"]
    if oa_mask.any():
        ax.scatter(stages["sc_x"][oa_mask], stages["sc_y"][oa_mask],
                   s=2, c="magenta", alpha=0.4, zorder=3,
                   label=f"out-of-arena episodes ({oa_mask.sum()} fr)")
        ax.legend(fontsize=7, loc="upper right")
    setup_px_ax_padded(ax, "5. After out-of-arena episode removal")

    # Panel (1,2): After perspective correction
    ax = fig.add_subplot(gs[1, 2])
    ax.plot(stages["pc_x"], stages["pc_y"], linewidth=lw, alpha=alpha, color="C0")
    setup_px_ax_padded(ax, "6. After perspective correction")
    # Also show corrected bounds in red
    draw_arena_rect(ax, cb["x_min"], cb["x_max"], cb["y_min"], cb["y_max"],
                    color="red", ls="-")

    # Panel (2,0): After clipping
    ax = fig.add_subplot(gs[2, 0])
    ax.plot(stages["clip_x"], stages["clip_y"], linewidth=lw, alpha=alpha, color="C0")
    setup_px_ax_tight(ax, "7. After clipping")

    # Panel (3,0-1): Final trajectory in cm + center zone
    ax = fig.add_subplot(gs[3, 0:2])
    ax.plot(stages["cm_x"], stages["cm_y"], linewidth=lw, alpha=alpha, color="C0")
    ax.set_xlim(0, arena_size)
    ax.set_ylim(0, arena_size)
    ax.set_aspect("equal")
    ax.set_title("8. Final trajectory (cm)", fontsize=10)
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")
    side = arena_size * np.sqrt(config["center_zone_fraction"])
    margin = (arena_size - side) / 2.0
    draw_arena_rect(ax, margin, arena_size - margin, margin, arena_size - margin,
                    color="red", ls="--")
    ax.text(arena_size / 2, arena_size - margin + 0.3, "center zone",
            ha="center", fontsize=7, color="red")

    # Panel (3,2): Quality summary box
    ax = fig.add_subplot(gs[3, 2])
    ax.axis("off")
    n = metrics["n_frames"]

    # Helper: check threshold-based QC metric, return reason string
    def check_threshold(value, threshold, label="threshold"):
        if value > threshold:
            return f"+{value - threshold:.2f}% over {threshold}% {label}"
        return ""

    # Helper: check population-based behavioral metric, return reason string
    def check_outlier(key):
        if pop_bounds is None or key not in pop_bounds:
            return ""
        lo, hi = pop_bounds[key]
        val = metrics[key]
        med = (lo + hi) / 2.0
        sd = (hi - med) / 2.0  # n_sd=2, so half-range = 2*sd
        if sd == 0:
            return ""
        if val > hi:
            n_over = (val - med) / sd
            return f"{n_over:.1f} SD above median"
        elif val < lo:
            n_under = (med - val) / sd
            return f"{n_under:.1f} SD below median"
        return ""

    # Build checks: (label, formatted_value, reason_if_bad)
    checks = [
        ("Frames", f"{n}", ""),
        ("Duration", f"{n / config['fps'] / 60:.1f} min", ""),
        ("", "", ""),
        ("Fallback total", f"{metrics['fallback_frames_pct']:.1f}%",
         check_threshold(metrics["fallback_frames_pct"], config["warn_fallback_pct"])),
        ("  midpoint", f"{metrics['fallback_midpoint_pct']:.1f}%", ""),
        ("  tail+offset", f"{metrics['fallback_offset_pct']:.1f}%",
         check_threshold(metrics["fallback_offset_pct"], 15.0)),
        ("Jump removal", f"{metrics['jump_frames_pct']:.2f}%",
         check_threshold(metrics["jump_frames_pct"], config["warn_jump_pct"])),
        ("Speed clean", f"{metrics['speed_cleaned_pct']:.2f}%",
         check_threshold(metrics["speed_cleaned_pct"], config["warn_speed_cleaned_pct"])),
        ("Out-of-arena", f"{metrics['out_of_arena_pct']:.2f}%",
         check_threshold(metrics["out_of_arena_pct"], 1.0)),
        ("", "", ""),
        ("Distance", f"{metrics['total_distance_cm']:.0f} cm",
         check_outlier("total_distance_cm")),
        ("Med speed", f"{metrics['median_speed_cms']:.2f} cm/s",
         check_outlier("median_speed_cms")),
        ("Med loco speed", f"{metrics['median_locomotion_speed_cms']:.2f} cm/s",
         check_outlier("median_locomotion_speed_cms")),
        ("Center dist", f"{metrics['center_distance_cm']:.0f} cm",
         check_outlier("center_distance_cm")),
        ("Coverage", f"{metrics['arena_coverage_pct']:.1f}%",
         check_outlier("arena_coverage_pct")),
        ("Immobility", f"{metrics['immobility_time_s']:.1f} s",
         check_outlier("immobility_time_s")),
    ]
    any_warning = any(reason for _, _, reason in checks)
    box_color = "#ffe0e0" if any_warning else "lightyellow"
    ax.add_patch(plt.Rectangle((0.02, 0.02), 0.96, 0.96, transform=ax.transAxes,
                                facecolor=box_color, edgecolor="gray",
                                linewidth=1, alpha=0.9, zorder=0, clip_on=False))
    y_pos = 0.93
    line_h = 0.055
    if any_warning:
        ax.text(0.5, 0.97, "! QUALITY WARNINGS !",
                transform=ax.transAxes, fontsize=11, ha="center",
                fontweight="bold", color="red")
        y_pos = 0.88
    for label, val, reason in checks:
        if label == "" and val == "":
            y_pos -= line_h * 0.4
            continue
        is_bad = bool(reason)
        color = "red" if is_bad else "black"
        weight = "bold" if is_bad else "normal"
        txt = f"{label}: {val}"
        if reason:
            txt += f"  ({reason})"
        ax.text(0.08, y_pos, txt,
                transform=ax.transAxes, fontsize=7.5, verticalalignment="top",
                fontfamily="monospace", color=color, fontweight=weight)
        y_pos -= line_h

    # Panel (4, spanning): Speed trace
    ax = fig.add_subplot(gs[4, :])
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

def _p_to_stars(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return "n.s."


def _fmt_p(p):
    if p < 0.001:
        return "p < 0.001"
    elif p < 0.05:
        return f"p = {p:.3f}"
    return f"p = {p:.2f}"


def plot_summary_boxplots(results_df, save_path):
    """Full-experiment summary plot with phase color-coding.

    Top row:  Individual animal data points with median line, color-coded
              by phase. D8+ points filled by miniscope type.
    Bottom row: Boxplots color-coded by phase. D8+ split into MS_Zero /
                v4_MS side-by-side.

    Phases:  D1-D4 = Arena habituation (blue)
             D5-D7 = Naive test (teal)
             D8-D16 = Scope habituation: MS_Zero (green) / v4_MS (purple)
    Note: M04 drops out from D15 onward (scope no longer available).
    """
    from matplotlib.lines import Line2D

    metrics_to_plot = [
        ("total_distance_cm", "Total Distance", lambda x: x / 100, "Distance (m)", (0, 100)),
        ("median_speed_cms", "Median Speed\n(all frames)", lambda x: x, "Speed (cm/s)", (0, 5)),
        ("median_locomotion_speed_cms", "Median Locomotion Speed\n(excl. immobile)", lambda x: x, "Speed (cm/s)", (0, 8)),
        ("center_distance_cm", "Distance in Center", lambda x: x / 100, "Distance (m)", (0, 30)),
        ("arena_coverage_pct", "Arena Coverage", lambda x: x, "Coverage (%)", (0, 100)),
        ("immobility_time_s", "Immobility Time", lambda x: x / 60, "Time (min)", None),
    ]

    # Phase definitions
    color_hab   = "#5b9bd5"  # blue  – habituation
    color_naive = "#2ca89a"  # teal  – naive test
    color_zero  = "#4daf4a"  # green – MS_Zero
    color_v4    = "#984ea3"  # purple – v4_MS

    # Derive day number and miniscope condition
    df = results_df.copy()
    df["day_num"] = df["Day"].str[1:].astype(int)
    df["Miniscope"] = df["Camera"].map({
        "left_camera": "MS_Zero", "right_camera": "v4_MS"})

    all_days = sorted(df["Day"].unique(), key=lambda d: int(d[1:]))
    naive_days = [d for d in all_days if int(d[1:]) <= 7]
    ms_days    = [d for d in all_days if int(d[1:]) >= 8]

    def _phase_color(day):
        dn = int(day[1:])
        if dn <= 4:
            return color_hab
        elif dn <= 7:
            return color_naive
        return "gray"  # shouldn't be used for pooled boxes on D8+

    # Animal identity markers (for top row)
    animals = sorted(df["Animal"].unique())
    cmap_colors = [
        "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
        "#a65628", "#f781bf", "#1b9e77", "#d95f02", "#666666",
    ]
    marker_list = ["o", "s", "D", "^", "v", "P", "X", "h", "*", "p"]
    animal_colors  = {a: cmap_colors[i % len(cmap_colors)]
                      for i, a in enumerate(animals)}
    animal_markers = {a: marker_list[i % len(marker_list)]
                      for i, a in enumerate(animals)}

    n_metrics = len(metrics_to_plot)
    fig, axes = plt.subplots(2, n_metrics, figsize=(30, 11))

    for col_idx, (col, title, transform, ylabel, ylim) in enumerate(
            metrics_to_plot):
        ax_top = axes[0, col_idx]
        ax_bot = axes[1, col_idx]

        # ----- build x-positions -----
        # Naive days get integer positions 1..N_naive.
        # Miniscope days get two sub-positions each (MS_Zero, v4_MS).
        x_tick_pos = []     # tick position for each day label
        x_tick_labels = []  # label strings
        x_pos = 1           # running position counter

        # -- data containers for bottom-row boxplots --
        bot_box_data = []
        bot_box_pos  = []
        bot_box_colors = []

        # -- top-row scatter helpers --
        top_scatter_info = []  # list of (day, x_center, x_left, x_right)

        for day in all_days:
            dn = int(day[1:])
            if dn <= 7:
                # Single pooled box
                x_tick_pos.append(x_pos)
                x_tick_labels.append(day)
                vals = transform(df[df["Day"] == day][col].values)
                bot_box_data.append(vals)
                bot_box_pos.append(x_pos)
                bot_box_colors.append(_phase_color(day))
                top_scatter_info.append((day, x_pos, None, None))
                x_pos += 1
            else:
                # Two side-by-side boxes
                x_left  = x_pos - 0.2
                x_right = x_pos + 0.2
                x_tick_pos.append(x_pos)
                x_tick_labels.append(day)

                vals_z = transform(
                    df[(df["Day"] == day) & (df["Miniscope"] == "MS_Zero")][col].values)
                vals_v = transform(
                    df[(df["Day"] == day) & (df["Miniscope"] == "v4_MS")][col].values)

                if len(vals_z) > 0:
                    bot_box_data.append(vals_z)
                    bot_box_pos.append(x_left)
                    bot_box_colors.append(color_zero)
                if len(vals_v) > 0:
                    bot_box_data.append(vals_v)
                    bot_box_pos.append(x_right)
                    bot_box_colors.append(color_v4)

                top_scatter_info.append((day, x_pos, x_left, x_right))
                x_pos += 1

        # Add a small gap between naive and miniscope phases
        if len(naive_days) > 0 and len(ms_days) > 0:
            gap_idx = len(naive_days)  # index in x_tick_pos where ms starts
            for k in range(gap_idx, len(x_tick_pos)):
                x_tick_pos[k] += 0.5
            for k in range(len(bot_box_pos)):
                if bot_box_pos[k] > len(naive_days):
                    bot_box_pos[k] += 0.5
            for k in range(len(top_scatter_info)):
                day, xc, xl, xr = top_scatter_info[k]
                if int(day[1:]) >= 8:
                    top_scatter_info[k] = (day, xc + 0.5,
                                           xl + 0.5 if xl else None,
                                           xr + 0.5 if xr else None)

        # ── Top row: median line + individual data points ──
        # Median line for naive days only (pooled)
        naive_medians_x = []
        naive_medians_y = []
        for day, xc, xl, xr in top_scatter_info:
            if int(day[1:]) <= 7:
                vals = transform(df[df["Day"] == day][col].values)
                naive_medians_x.append(xc)
                naive_medians_y.append(np.median(vals))
        ax_top.plot(naive_medians_x, naive_medians_y, color="black",
                    linewidth=1.5, zorder=2, marker="_", markersize=12,
                    markeredgewidth=2)

        # Scatter individual points
        for day, xc, xl, xr in top_scatter_info:
            dn = int(day[1:])
            day_df = df[df["Day"] == day].sort_values("Animal")
            if dn <= 7:
                n_animals = len(day_df)
                offsets = np.linspace(-0.25, 0.25, n_animals)
                for j, (_, row) in enumerate(day_df.iterrows()):
                    animal = row["Animal"]
                    val = transform(np.array([row[col]]))[0]
                    ax_top.scatter(
                        xc + offsets[j], val,
                        c=animal_colors[animal],
                        marker=animal_markers[animal],
                        s=55, alpha=0.9, edgecolors="black",
                        linewidths=0.5, zorder=3)
            else:
                # Split by miniscope
                for ms_type, x_sub, ms_color in [
                    ("MS_Zero", xl, color_zero),
                    ("v4_MS",   xr, color_v4),
                ]:
                    if x_sub is None:
                        continue
                    sub = day_df[day_df["Miniscope"] == ms_type]
                    n_sub = len(sub)
                    if n_sub == 0:
                        continue
                    offsets = np.linspace(-0.08, 0.08, n_sub)
                    for j, (_, row) in enumerate(sub.iterrows()):
                        animal = row["Animal"]
                        val = transform(np.array([row[col]]))[0]
                        ax_top.scatter(
                            x_sub + offsets[j], val,
                            c=ms_color,
                            marker=animal_markers[animal],
                            s=55, alpha=0.85, edgecolors="black",
                            linewidths=0.5, zorder=3)

        ax_top.set_xticks(x_tick_pos)
        ax_top.set_xticklabels(x_tick_labels, fontsize=8)
        ax_top.set_ylabel(ylabel)
        ax_top.set_title(title, fontsize=10)

        # Phase background shading (top row)
        for day, xc, xl, xr in top_scatter_info:
            pc = _phase_color(day)
            if int(day[1:]) <= 7:
                ax_top.axvspan(xc - 0.45, xc + 0.45, alpha=0.18,
                               color=pc, zorder=0)

        # ── Bottom row: phase-colored boxplots ──
        box_width = 0.38
        bp = ax_bot.boxplot(
            bot_box_data, positions=bot_box_pos,
            widths=[box_width] * len(bot_box_data),
            patch_artist=True, showfliers=False,
            medianprops=dict(color="black", linewidth=2),
            whiskerprops=dict(color="gray"),
            capprops=dict(color="gray"))
        for patch, fc in zip(bp["boxes"], bot_box_colors):
            patch.set_facecolor(fc)
            patch.set_alpha(0.55)

        ax_bot.set_xticks(x_tick_pos)
        ax_bot.set_xticklabels(x_tick_labels, fontsize=8)
        ax_bot.set_ylabel(ylabel)
        ax_bot.set_xlabel("Day")

        # Phase background shading (bottom row)
        for day, xc, xl, xr in top_scatter_info:
            pc = _phase_color(day)
            if int(day[1:]) <= 7:
                ax_bot.axvspan(xc - 0.45, xc + 0.45, alpha=0.18,
                               color=pc, zorder=0)

        # y-limits
        if ylim is not None:
            ax_top.set_ylim(ylim)
            ax_bot.set_ylim(ylim)
        else:
            all_vals = transform(df[col].values)
            y_max = np.max(all_vals) * 1.10
            ax_top.set_ylim(0, y_max)
            ax_bot.set_ylim(0, y_max)

    # ── Legends ──
    # Animal identity legend (top row)
    animal_handles = [
        Line2D([0], [0], marker=animal_markers[a], color="w",
               markerfacecolor=animal_colors[a], markeredgecolor="black",
               markersize=8, label=a)
        for a in animals
    ]
    axes[0, -1].legend(handles=animal_handles, loc="upper left",
                       title="Animal", fontsize=7, title_fontsize=8,
                       bbox_to_anchor=(1.02, 1.0), borderaxespad=0)

    # Phase legend (bottom row)
    from matplotlib.patches import Patch
    phase_handles = [
        Patch(facecolor=color_hab,   alpha=0.55, edgecolor="black",
              label="Habituation (D1\u2013D4)"),
        Patch(facecolor=color_naive, alpha=0.55, edgecolor="black",
              label="Naive test (D5\u2013D7)"),
        Patch(facecolor=color_zero,  alpha=0.55, edgecolor="black",
              label="MS_Zero (D8\u2013D16, left cam)"),
        Patch(facecolor=color_v4,    alpha=0.55, edgecolor="black",
              label="v4_MS (D8\u2013D16, right cam)"),
    ]
    axes[1, -1].legend(handles=phase_handles, loc="upper left",
                       title="Phase", fontsize=7, title_fontsize=8,
                       bbox_to_anchor=(1.02, 1.0), borderaxespad=0)

    # Row labels
    axes[0, 0].annotate("Individual animals", xy=(0, 0.5),
                        xytext=(-0.35, 0.5), xycoords="axes fraction",
                        fontsize=10, fontweight="bold", rotation=90,
                        va="center", ha="center")
    axes[1, 0].annotate("Day-to-day trend", xy=(0, 0.5),
                        xytext=(-0.35, 0.5), xycoords="axes fraction",
                        fontsize=10, fontweight="bold", rotation=90,
                        va="center", ha="center")

    fig.suptitle("Behavior Metrics Summary", fontsize=14, y=0.98)
    fig.tight_layout(rect=[0.03, 0, 0.93, 0.96])
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_miniscope_comparison(results_df, save_path):
    """Two-row comparison plot for scope habituation days (D8–D16).

    Top row:  Per-day side-by-side boxplots (MS_Zero vs v4_MS) with
              Mann-Whitney U tests per day.
    Bottom row: Overall pooled comparison across all scope habituation days
                with individual animal data points.

    Camera mapping: left_camera = MS_Zero, right_camera = v4_MS.
    Note: M04 drops out from D15 onward (scope no longer available).
    """
    from matplotlib.lines import Line2D
    from scipy.stats import mannwhitneyu

    # Assign miniscope condition based on camera
    df = results_df.copy()
    df["Miniscope"] = df["Camera"].map({
        "left_camera": "MS_Zero",
        "right_camera": "v4_MS",
    })

    # Filter to miniscope days (D8+)
    df["day_num"] = df["Day"].str[1:].astype(int)
    ms_df = df[df["day_num"] >= 8].copy()
    ms_days = sorted(ms_df["Day"].unique(), key=lambda d: int(d[1:]))

    # Naive test days (D5-D7) for pooled baseline reference
    naive_df = df[df["day_num"].between(5, 7)].copy()

    if len(ms_days) == 0:
        return

    metrics_to_plot = [
        ("total_distance_cm", "Total Distance", lambda x: x / 100, "Distance (m)", None),
        ("median_speed_cms", "Median Speed\n(all frames)", lambda x: x, "Speed (cm/s)", None),
        ("median_locomotion_speed_cms", "Median Locomotion Speed\n(excl. immobile)", lambda x: x, "Speed (cm/s)", None),
        ("center_distance_cm", "Distance in Center", lambda x: x / 100, "Distance (m)", None),
        ("arena_coverage_pct", "Arena Coverage", lambda x: x, "Coverage (%)", (0, 100)),
        ("immobility_time_s", "Immobility Time", lambda x: x / 60, "Time (min)", None),
    ]
    n_metrics = len(metrics_to_plot)

    color_naive = "#2ca89a"  # teal  – naive test
    color_zero  = "#4daf4a"  # green – MS_Zero
    color_v4    = "#984ea3"  # purple – v4_MS

    fig, axes = plt.subplots(2, n_metrics, figsize=(28, 11))

    for col_idx, (col, title, transform, ylabel, ylim) in enumerate(metrics_to_plot):
        ax_top = axes[0, col_idx]
        ax_bot = axes[1, col_idx]

        # ── Top row: per-day side-by-side boxplots ──
        positions_zero = []
        positions_v4 = []
        data_zero = []
        data_v4 = []
        box_width = 0.35

        for i, day in enumerate(ms_days):
            x_center = i + 1
            day_data = ms_df[ms_df["Day"] == day]

            vals_zero = transform(day_data[day_data["Miniscope"] == "MS_Zero"][col].values)
            vals_v4   = transform(day_data[day_data["Miniscope"] == "v4_MS"][col].values)

            data_zero.append(vals_zero)
            data_v4.append(vals_v4)
            positions_zero.append(x_center - box_width / 2 - 0.02)
            positions_v4.append(x_center + box_width / 2 + 0.02)

            # Mann-Whitney U per day
            if len(vals_zero) >= 3 and len(vals_v4) >= 3:
                _, p_val = mannwhitneyu(vals_zero, vals_v4, alternative="two-sided")
                stars = _p_to_stars(p_val)
                star_color = "#e67e22" if p_val < 0.05 else "#bbbbbb"
                top_val = max(
                    np.max(vals_zero) if len(vals_zero) > 0 else 0,
                    np.max(vals_v4) if len(vals_v4) > 0 else 0,
                )
                ax_top.text(x_center, top_val * 1.05, stars,
                            ha="center", va="bottom", fontsize=8,
                            color=star_color,
                            fontweight="bold" if p_val < 0.05 else "normal")

        if len(data_zero) > 0:
            ax_top.boxplot(data_zero, positions=positions_zero, widths=box_width,
                           patch_artist=True, showfliers=False,
                           boxprops=dict(facecolor=color_zero, alpha=0.5),
                           medianprops=dict(color="black", linewidth=2),
                           whiskerprops=dict(color="gray"),
                           capprops=dict(color="gray"))
        if len(data_v4) > 0:
            ax_top.boxplot(data_v4, positions=positions_v4, widths=box_width,
                           patch_artist=True, showfliers=False,
                           boxprops=dict(facecolor=color_v4, alpha=0.5),
                           medianprops=dict(color="black", linewidth=2),
                           whiskerprops=dict(color="gray"),
                           capprops=dict(color="gray"))

        ax_top.set_xticks(list(range(1, len(ms_days) + 1)))
        ax_top.set_xticklabels(ms_days)
        ax_top.set_ylabel(ylabel)
        ax_top.set_title(title, fontsize=10)

        # ── Bottom row: pooled 3-group comparison ──
        # Naive test (D5-D7), MS_Zero (D8+), v4_MS (D8+)
        all_naive = transform(naive_df[col].values)
        all_zero  = transform(ms_df[ms_df["Miniscope"] == "MS_Zero"][col].values)
        all_v4    = transform(ms_df[ms_df["Miniscope"] == "v4_MS"][col].values)

        grp_data   = [all_naive, all_zero, all_v4]
        grp_pos    = [1, 2, 3]
        grp_colors = [color_naive, color_zero, color_v4]
        grp_labels = ["Naive\n(D5\u2013D7)", "MS_Zero\n(D8\u2013D16)", "v4_MS\n(D8\u2013D16)"]

        bp_pool = ax_bot.boxplot(
            grp_data, positions=grp_pos, widths=0.5,
            patch_artist=True, showfliers=False,
            medianprops=dict(color="black", linewidth=2),
            whiskerprops=dict(color="gray"),
            capprops=dict(color="gray"),
        )
        for patch, fc in zip(bp_pool["boxes"], grp_colors):
            patch.set_facecolor(fc)
            patch.set_alpha(0.5)

        # Overlay individual points with jitter
        rng = np.random.default_rng(42)
        for gx, gvals, gc in zip(grp_pos, grp_data, grp_colors):
            jitter = rng.uniform(-0.12, 0.12, size=len(gvals))
            ax_bot.scatter(np.full(len(gvals), gx) + jitter, gvals,
                           c=gc, s=30, alpha=0.6, edgecolors="black",
                           linewidths=0.4, zorder=3)

        ax_bot.set_xticks(grp_pos)
        ax_bot.set_xticklabels(grp_labels)
        ax_bot.set_ylabel(ylabel)

        # Pairwise Mann-Whitney U brackets between all three groups
        pairs = [(0, 1), (1, 2), (0, 2)]  # indices into grp_data/grp_pos
        pair_labels = [("Naive", "MS_Zero"), ("MS_Zero", "v4_MS"),
                       ("Naive", "v4_MS")]
        all_max = max(np.max(d) for d in grp_data if len(d) > 0)
        bracket_gap = all_max * 0.07
        tick_h = all_max * 0.015

        for level, (i1, i2) in enumerate(pairs):
            d1, d2 = grp_data[i1], grp_data[i2]
            if len(d1) >= 3 and len(d2) >= 3:
                _, p_val = mannwhitneyu(d1, d2, alternative="two-sided")
                stars = _p_to_stars(p_val)
                sig_color = "#e67e22" if p_val < 0.05 else "#bbbbbb"
                x1, x2 = grp_pos[i1], grp_pos[i2]
                by = all_max * 1.06 + level * bracket_gap
                ax_bot.plot([x1, x1, x2, x2],
                            [by - tick_h, by, by, by - tick_h],
                            color=sig_color, linewidth=1.0, clip_on=False)
                ax_bot.text((x1 + x2) / 2, by + tick_h * 0.5, stars,
                            ha="center", va="bottom", fontsize=7.5,
                            color=sig_color,
                            fontweight="bold" if p_val < 0.05 else "normal",
                            clip_on=False)

        # y-limits — account for brackets
        if ylim is not None:
            ax_top.set_ylim(ylim)
            ax_bot.set_ylim(ylim[0], ylim[1] * 1.15)
        else:
            combined = np.concatenate([d for d in grp_data if len(d) > 0]
                                      + [transform(ms_df[col].values)])
            y_max = np.max(combined) * 1.35
            ax_top.set_ylim(0, np.max(combined) * 1.20)
            ax_bot.set_ylim(0, y_max)

    # Legend
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=color_naive, alpha=0.5, edgecolor="black",
              label="Naive test (D5\u2013D7)"),
        Patch(facecolor=color_zero, alpha=0.5, edgecolor="black",
              label="MS_Zero (D8\u2013D16, left cam)"),
        Patch(facecolor=color_v4, alpha=0.5, edgecolor="black",
              label="v4_MS (D8\u2013D16, right cam)"),
    ]
    axes[0, -1].legend(handles=legend_handles, loc="upper left",
                       title="Condition", fontsize=8, title_fontsize=9,
                       bbox_to_anchor=(1.02, 1.0), borderaxespad=0)

    # Row labels
    axes[0, 0].annotate("Per-day comparison", xy=(0, 0.5),
                        xytext=(-0.35, 0.5), xycoords="axes fraction",
                        fontsize=10, fontweight="bold", rotation=90,
                        va="center", ha="center")
    axes[1, 0].annotate("Pooled comparison", xy=(0, 0.5),
                        xytext=(-0.35, 0.5), xycoords="axes fraction",
                        fontsize=10, fontweight="bold", rotation=90,
                        va="center", ha="center")

    fig.suptitle("Behavior Metrics — Naive (D5\u2013D7) vs Scope Habituation (D8\u2013D16): MS_Zero vs v4_MS",
                 fontsize=14, y=0.98)
    fig.tight_layout(rect=[0.03, 0, 0.93, 0.96])
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_naive_vs_test(results_df, save_path):
    """Publication figure: Naive (D5-D7) vs Scope Test MS_Zero & v4_MS (D14-D16).

    Single row, 4 metrics. Three groups per panel with pairwise
    Mann-Whitney U brackets and sample size (n) annotations.
    """
    from scipy.stats import mannwhitneyu
    from matplotlib.patches import Patch

    df = results_df.copy()
    df["day_num"] = df["Day"].str[1:].astype(int)
    df["Miniscope"] = df["Camera"].map({
        "left_camera": "MS_Zero", "right_camera": "v4_MS"})

    naive_df = df[df["day_num"].between(5, 7)].copy()
    test_df  = df[df["day_num"].between(14, 16)].copy()

    color_naive = "#2ca89a"
    color_zero  = "#4daf4a"
    color_v4    = "#984ea3"

    metrics_to_plot = [
        ("total_distance_cm", "Total Distance",
         lambda x: x / 100, "Distance (m)", None),
        ("median_locomotion_speed_cms", "Locomotion Speed\n(excl. immobile)",
         lambda x: x, "Speed (cm/s)", None),
        ("center_distance_cm", "Distance in Center",
         lambda x: x / 100, "Distance (m)", (0, 30)),
        ("immobility_time_s", "Immobility Time",
         lambda x: x / 60, "Time (min)", None),
    ]
    n_metrics = len(metrics_to_plot)

    fig, axes = plt.subplots(1, n_metrics, figsize=(16, 5))
    rng = np.random.default_rng(42)

    for col_idx, (col, title, transform, ylabel, ylim) in enumerate(metrics_to_plot):
        ax = axes[col_idx]

        # Open axes: remove top and right spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        vals_naive = transform(naive_df[col].values)
        vals_zero  = transform(test_df[test_df["Miniscope"] == "MS_Zero"][col].values)
        vals_v4    = transform(test_df[test_df["Miniscope"] == "v4_MS"][col].values)

        grp_data   = [vals_naive, vals_zero, vals_v4]
        grp_pos    = [1, 2, 3]
        grp_colors = [color_naive, color_zero, color_v4]
        grp_labels = ["Control", "MS_Zero", "v4_MS"]

        bp = ax.boxplot(
            grp_data, positions=grp_pos, widths=0.5,
            patch_artist=True, showfliers=False,
            medianprops=dict(color="black", linewidth=2),
            whiskerprops=dict(color="gray"),
            capprops=dict(color="gray"))
        for patch, fc in zip(bp["boxes"], grp_colors):
            patch.set_facecolor(fc)
            patch.set_alpha(0.5)

        for gx, gvals, gc in zip(grp_pos, grp_data, grp_colors):
            jitter = rng.uniform(-0.12, 0.12, size=len(gvals))
            ax.scatter(np.full(len(gvals), gx) + jitter, gvals,
                       c=gc, s=18, alpha=0.6, edgecolors="black",
                       linewidths=0.3, zorder=3)

        ax.set_xticks(grp_pos)
        ax.set_xticklabels(grp_labels, fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)

        # Pairwise brackets: only show significant comparisons
        pairs = [(0, 1), (1, 2), (0, 2)]
        all_max = max(np.max(d) for d in grp_data if len(d) > 0)
        bracket_gap = all_max * 0.08
        tick_h = all_max * 0.015

        sig_level = 0
        for i1, i2 in pairs:
            d1, d2 = grp_data[i1], grp_data[i2]
            if len(d1) >= 3 and len(d2) >= 3:
                _, p_val = mannwhitneyu(d1, d2, alternative="two-sided")
                if p_val >= 0.05:
                    continue
                stars = _p_to_stars(p_val)
                x1, x2 = grp_pos[i1], grp_pos[i2]
                by = all_max * 1.06 + sig_level * bracket_gap
                ax.plot([x1, x1, x2, x2],
                        [by - tick_h, by, by, by - tick_h],
                        color="#e67e22", linewidth=1.0, clip_on=False)
                ax.text((x1 + x2) / 2, by + tick_h * 0.5,
                        stars,
                        ha="center", va="bottom", fontsize=9,
                        color="#e67e22", fontweight="bold",
                        clip_on=False)
                sig_level += 1

        if ylim is not None:
            ax.set_ylim(ylim)
        else:
            ax.set_ylim(0, all_max * 1.45)

    fig.suptitle("Control vs Scope Test \u2014 MS_Zero vs v4_MS",
                 fontsize=13, y=0.98)
    fig.tight_layout(rect=[0.01, 0, 0.98, 0.94])
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_publication_figure(results_df, save_path):
    """Publication figure: 2 rows x 3 metrics.

    Top row:    Pooled comparison — Naive (D5-D7) vs MS_Zero (D8-D16) vs
                v4_MS (D8-D16) with individual data points and Mann-Whitney U.
    Bottom row: Phase-split comparison — Scope Habituation (D8-D13) and
                Scope Test (D14-D16), each MS_Zero vs v4_MS.
    """
    from scipy.stats import mannwhitneyu
    from matplotlib.patches import Patch

    df = results_df.copy()
    df["day_num"] = df["Day"].str[1:].astype(int)
    df["Miniscope"] = df["Camera"].map({
        "left_camera": "MS_Zero", "right_camera": "v4_MS"})

    naive_df = df[df["day_num"].between(5, 7)].copy()
    ms_df    = df[df["day_num"] >= 8].copy()
    hab_df   = df[df["day_num"].between(8, 13)].copy()
    test_df  = df[df["day_num"].between(14, 16)].copy()

    color_naive = "#2ca89a"
    color_zero  = "#4daf4a"
    color_v4    = "#984ea3"
    # Lighter shades for habituation, saturated for test
    color_zero_hab  = "#a2d99c"
    color_v4_hab    = "#c5a3d9"
    color_zero_test = "#2d8e29"
    color_v4_test   = "#6a2c91"

    metrics_to_plot = [
        ("total_distance_cm", "Total Distance",
         lambda x: x / 100, "Distance (m)"),
        ("median_locomotion_speed_cms", "Median Locomotion Speed\n(excl. immobile)",
         lambda x: x, "Speed (cm/s)"),
        ("center_distance_cm", "Distance in Center",
         lambda x: x / 100, "Distance (m)"),
    ]
    n_metrics = len(metrics_to_plot)

    fig, axes = plt.subplots(2, n_metrics, figsize=(14, 9))
    rng = np.random.default_rng(42)

    def _draw_brackets(ax, grp_data, grp_pos, pairs):
        """Draw Mann-Whitney U brackets between specified pairs."""
        all_max = max(np.max(d) for d in grp_data if len(d) > 0)
        bracket_gap = all_max * 0.08
        tick_h = all_max * 0.015
        for level, (i1, i2) in enumerate(pairs):
            d1, d2 = grp_data[i1], grp_data[i2]
            if len(d1) >= 3 and len(d2) >= 3:
                _, p_val = mannwhitneyu(d1, d2, alternative="two-sided")
                stars = _p_to_stars(p_val)
                sig_color = "#e67e22" if p_val < 0.05 else "#bbbbbb"
                x1, x2 = grp_pos[i1], grp_pos[i2]
                by = all_max * 1.06 + level * bracket_gap
                ax.plot([x1, x1, x2, x2],
                        [by - tick_h, by, by, by - tick_h],
                        color=sig_color, linewidth=1.0, clip_on=False)
                ax.text((x1 + x2) / 2, by + tick_h * 0.5,
                        f"{stars}\n{_fmt_p(p_val)}",
                        ha="center", va="bottom", fontsize=7,
                        color=sig_color,
                        fontweight="bold" if p_val < 0.05 else "normal",
                        clip_on=False)

    for col_idx, (col, title, transform, ylabel) in enumerate(metrics_to_plot):
        ax_top = axes[0, col_idx]
        ax_bot = axes[1, col_idx]

        # ── Top row: pooled 3-group (same as miniscope_comparison bottom) ──
        all_naive = transform(naive_df[col].values)
        all_zero  = transform(ms_df[ms_df["Miniscope"] == "MS_Zero"][col].values)
        all_v4    = transform(ms_df[ms_df["Miniscope"] == "v4_MS"][col].values)

        grp_data   = [all_naive, all_zero, all_v4]
        grp_pos    = [1, 2, 3]
        grp_colors = [color_naive, color_zero, color_v4]
        grp_labels = ["Naive\n(D5\u2013D7)", "MS_Zero\n(D8\u2013D16)",
                       "v4_MS\n(D8\u2013D16)"]

        bp = ax_top.boxplot(
            grp_data, positions=grp_pos, widths=0.5,
            patch_artist=True, showfliers=False,
            medianprops=dict(color="black", linewidth=2),
            whiskerprops=dict(color="gray"),
            capprops=dict(color="gray"))
        for patch, fc in zip(bp["boxes"], grp_colors):
            patch.set_facecolor(fc)
            patch.set_alpha(0.5)

        for gx, gvals, gc in zip(grp_pos, grp_data, grp_colors):
            jitter = rng.uniform(-0.12, 0.12, size=len(gvals))
            ax_top.scatter(np.full(len(gvals), gx) + jitter, gvals,
                           c=gc, s=30, alpha=0.6, edgecolors="black",
                           linewidths=0.4, zorder=3)

        ax_top.set_xticks(grp_pos)
        ax_top.set_xticklabels(grp_labels, fontsize=8)
        ax_top.set_ylabel(ylabel)
        ax_top.set_title(title, fontsize=10)

        _draw_brackets(ax_top, grp_data, grp_pos,
                       [(0, 1), (1, 2), (0, 2)])

        combined_top = np.concatenate([d for d in grp_data if len(d) > 0])
        ax_top.set_ylim(0, np.max(combined_top) * 1.40)

        # ── Bottom row: 5-group phase split (Naive + Hab + Test) ──
        bot_naive = transform(naive_df[col].values)
        hab_zero  = transform(hab_df[hab_df["Miniscope"] == "MS_Zero"][col].values)
        hab_v4    = transform(hab_df[hab_df["Miniscope"] == "v4_MS"][col].values)
        test_zero = transform(test_df[test_df["Miniscope"] == "MS_Zero"][col].values)
        test_v4   = transform(test_df[test_df["Miniscope"] == "v4_MS"][col].values)

        grp5_data   = [bot_naive, hab_zero, hab_v4, test_zero, test_v4]
        grp5_pos    = [1, 2.8, 3.8, 5.6, 6.6]
        grp5_colors = [color_naive, color_zero_hab, color_v4_hab,
                        color_zero_test, color_v4_test]
        grp5_labels = ["Naive", "MS_Zero", "v4_MS", "MS_Zero", "v4_MS"]

        bp5 = ax_bot.boxplot(
            grp5_data, positions=grp5_pos, widths=0.5,
            patch_artist=True, showfliers=False,
            medianprops=dict(color="black", linewidth=2),
            whiskerprops=dict(color="gray"),
            capprops=dict(color="gray"))
        for patch, fc in zip(bp5["boxes"], grp5_colors):
            patch.set_facecolor(fc)
            patch.set_alpha(0.6)

        for gx, gvals, gc in zip(grp5_pos, grp5_data, grp5_colors):
            jitter = rng.uniform(-0.12, 0.12, size=len(gvals))
            ax_bot.scatter(np.full(len(gvals), gx) + jitter, gvals,
                           c=gc, s=30, alpha=0.6, edgecolors="black",
                           linewidths=0.4, zorder=3)

        ax_bot.set_xticks(grp5_pos)
        ax_bot.set_xticklabels(grp5_labels, fontsize=8)
        ax_bot.set_ylabel(ylabel)

        # Phase group labels below x-axis
        ax_bot.text(1.0, -0.18, "Naive\n(D5\u2013D7)",
                    transform=ax_bot.get_xaxis_transform(),
                    ha="center", va="top", fontsize=8, fontweight="bold")
        ax_bot.text(3.3, -0.18, "Scope Habituation\n(D8\u2013D13)",
                    transform=ax_bot.get_xaxis_transform(),
                    ha="center", va="top", fontsize=8, fontweight="bold")
        ax_bot.text(6.1, -0.18, "Scope Test\n(D14\u2013D16)",
                    transform=ax_bot.get_xaxis_transform(),
                    ha="center", va="top", fontsize=8, fontweight="bold")

        # Brackets: MS_Zero vs v4_MS within each phase,
        # plus Naive vs each scope test group
        _draw_brackets(ax_bot, grp5_data, grp5_pos,
                       [(1, 2), (3, 4), (0, 3), (0, 4)])

        combined_bot = np.concatenate([d for d in grp5_data if len(d) > 0])
        ax_bot.set_ylim(0, np.max(combined_bot) * 1.55)

    # ── Legends ──
    top_handles = [
        Patch(facecolor=color_naive, alpha=0.5, edgecolor="black",
              label="Naive (D5\u2013D7)"),
        Patch(facecolor=color_zero, alpha=0.5, edgecolor="black",
              label="MS_Zero (D8\u2013D16)"),
        Patch(facecolor=color_v4, alpha=0.5, edgecolor="black",
              label="v4_MS (D8\u2013D16)"),
    ]
    axes[0, -1].legend(handles=top_handles, loc="upper left",
                       fontsize=7, title="Condition", title_fontsize=8,
                       bbox_to_anchor=(1.02, 1.0), borderaxespad=0)

    bot_handles = [
        Patch(facecolor=color_naive, alpha=0.6, edgecolor="black",
              label="Naive (D5\u2013D7)"),
        Patch(facecolor=color_zero_hab, alpha=0.6, edgecolor="black",
              label="MS_Zero \u2013 Hab (D8\u2013D13)"),
        Patch(facecolor=color_v4_hab, alpha=0.6, edgecolor="black",
              label="v4_MS \u2013 Hab (D8\u2013D13)"),
        Patch(facecolor=color_zero_test, alpha=0.6, edgecolor="black",
              label="MS_Zero \u2013 Test (D14\u2013D16)"),
        Patch(facecolor=color_v4_test, alpha=0.6, edgecolor="black",
              label="v4_MS \u2013 Test (D14\u2013D16)"),
    ]
    axes[1, -1].legend(handles=bot_handles, loc="upper left",
                       fontsize=7, title="Phase", title_fontsize=8,
                       bbox_to_anchor=(1.02, 1.0), borderaxespad=0)

    # Row labels
    axes[0, 0].annotate("Pooled (D8\u2013D16)", xy=(0, 0.5),
                        xytext=(-0.35, 0.5), xycoords="axes fraction",
                        fontsize=10, fontweight="bold", rotation=90,
                        va="center", ha="center")
    axes[1, 0].annotate("Habituation vs Test", xy=(0, 0.5),
                        xytext=(-0.35, 0.5), xycoords="axes fraction",
                        fontsize=10, fontweight="bold", rotation=90,
                        va="center", ha="center")

    fig.suptitle("MS_Zero vs v4_MS \u2014 Behavioral Comparison",
                 fontsize=13, y=0.98)
    fig.tight_layout(rect=[0.04, 0.05, 0.92, 0.96])
    fig.savefig(save_path, dpi=300)
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

    # Step 1c: Remove out-of-arena episodes (entire excursions outside arena)
    oa_x, oa_y, oa_mask = remove_out_of_arena_episodes(sc_x, sc_y, arena_bounds)
    stages["oa_x"], stages["oa_y"], stages["out_of_arena_mask"] = oa_x, oa_y, oa_mask

    # Step 2: Perspective correction
    if config.get("skip_perspective_correction", False):
        pc_x, pc_y = oa_x.copy(), oa_y.copy()
        corrected_bounds = arena_bounds
    else:
        pc_x, pc_y = correct_perspective(oa_x, oa_y, arena_bounds,
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
        "median_locomotion_speed_cms": float(np.median(speed[speed >= config["immobility_threshold_cms"]])) if (speed >= config["immobility_threshold_cms"]).any() else 0.0,
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
        "out_of_arena_pct": float(oa_mask.sum() / len(oa_mask) * 100),
        "n_frames": data["n_frames"],
    }

    return metrics, stages


def find_camera_groups(day_dir, day):
    """Detect camera subfolders or fall back to flat layout.

    Returns list of (camera_label, base_path, yaml_path) tuples.
    Supports both:
      - New layout: D1_DLC/left_camera/arena_corners_D1.yaml + M01/...
      - Old layout: D1_DLC/arena_corners_D1.yaml + M01/...
    """
    groups = []
    # Check for camera subfolders (any folder containing an arena_corners yaml)
    for entry in sorted(os.listdir(day_dir)):
        sub = os.path.join(day_dir, entry)
        if os.path.isdir(sub):
            yaml_path = os.path.join(sub, f"arena_corners_{day}.yaml")
            if os.path.isfile(yaml_path):
                groups.append((entry, sub, yaml_path))

    if groups:
        return groups

    # Fall back to flat layout (no camera subfolders)
    yaml_path = os.path.join(day_dir, f"arena_corners_{day}.yaml")
    if os.path.isfile(yaml_path):
        return [("", day_dir, yaml_path)]

    return []


def compute_population_bounds(all_metrics, n_sd=2):
    """Compute median ± n_sd * std for behavioral metrics across all animals.

    Returns dict of {metric_name: (low, high)} for outlier flagging.
    """
    behavior_keys = ["total_distance_cm", "median_speed_cms",
                     "median_locomotion_speed_cms", "center_distance_cm",
                     "arena_coverage_pct", "immobility_time_s"]
    bounds = {}
    for key in behavior_keys:
        vals = np.array([m[key] for m in all_metrics])
        med = np.median(vals)
        sd = np.std(vals)
        bounds[key] = (med - n_sd * sd, med + n_sd * sd)
    return bounds


EXCLUDE_ANIMALS = {"M04"}  # M04 lost scope from D15; exclude for balanced comparisons


def main():
    days = sorted([d for d in os.listdir(BASE_DIR) if d.endswith("_DLC")])
    all_results = []  # list of (metrics, stages, arena_bounds, meta, plot_path)

    # ── Pass 1: process all animals ──
    for day_folder in days:
        day = day_folder.replace("_DLC", "")
        day_dir = os.path.join(BASE_DIR, day_folder)

        camera_groups = find_camera_groups(day_dir, day)
        if not camera_groups:
            print(f"[{day}] No arena corners YAML found — skipping.")
            continue

        for camera_label, cam_dir, yaml_path in camera_groups:
            arena_bounds = load_arena_corners(yaml_path)
            tag = f"{day}/{camera_label}" if camera_label else day

            animal_dirs = sorted(glob.glob(os.path.join(cam_dir, "M*")))
            for animal_dir in animal_dirs:
                if not os.path.isdir(animal_dir):
                    continue
                animal = os.path.basename(animal_dir)
                if animal in EXCLUDE_ANIMALS:
                    continue
                csv_files = glob.glob(os.path.join(animal_dir, "*.csv"))
                if not csv_files:
                    continue
                csv_path = csv_files[0]

                print(f"Processing {tag} | {animal} ...")
                metrics, stages = process_one_animal(csv_path, arena_bounds, CONFIG)
                metrics["Day"] = day
                metrics["Camera"] = camera_label
                metrics["Animal"] = animal

                prefix = f"{day}_{camera_label}_{animal}" if camera_label else f"{day}_{animal}"
                plot_path = os.path.join(PLOT_DIR, f"{prefix}_pipeline.png")

                all_results.append((metrics, stages, arena_bounds,
                                    {"day": tag, "animal": animal}, plot_path))

    if not all_results:
        print("No data found.")
        return

    # ── Compute population bounds for behavioral outlier flagging ──
    all_metrics = [r[0] for r in all_results]
    pop_bounds = compute_population_bounds(all_metrics)

    # ── Pass 2: generate pipeline plots with population-aware warnings ──
    for metrics, stages, arena_bounds, meta, plot_path in all_results:
        plot_pipeline_figure(stages, arena_bounds, CONFIG, metrics, meta,
                             plot_path, pop_bounds=pop_bounds)
        print(f"  -> saved {plot_path}")

    # Build results table
    results = pd.DataFrame(all_metrics)
    col_order = ["Day", "Camera", "Animal", "n_frames", "total_distance_cm", "median_speed_cms",
                 "median_locomotion_speed_cms",
                 "center_distance_cm", "arena_coverage_pct", "immobility_time_s",
                 "fallback_frames_pct", "fallback_midpoint_pct", "fallback_offset_pct",
                 "jump_frames_pct", "speed_cleaned_pct", "out_of_arena_pct"]
    results = results[col_order]

    # Save CSV
    csv_out = os.path.join(OUT_DIR, "behavior_metrics.csv")
    results.to_csv(csv_out, index=False)
    print(f"\nCSV saved: {csv_out}")

    # Summary boxplot (naive baseline)
    summary_path = os.path.join(PLOT_DIR, "summary_boxplots.png")
    plot_summary_boxplots(results, summary_path)
    print(f"Summary plot saved: {summary_path}")

    # Miniscope comparison (D8+)
    ms_compare_path = os.path.join(PLOT_DIR, "miniscope_comparison.png")
    plot_miniscope_comparison(results, ms_compare_path)
    print(f"Miniscope comparison plot saved: {ms_compare_path}")

    # Publication figure (pooled + habituation vs test)
    pub_path = os.path.join(PLOT_DIR, "publication_figure.png")
    plot_publication_figure(results, pub_path)
    print(f"Publication figure saved: {pub_path}")

    # Publication figure (naive vs scope test)
    naive_test_path = os.path.join(PLOT_DIR, "publication_naive_vs_test.png")
    plot_naive_vs_test(results, naive_test_path)
    print(f"Naive vs test figure saved: {naive_test_path}")

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

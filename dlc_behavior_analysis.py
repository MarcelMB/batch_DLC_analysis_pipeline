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
    "center_zone_fraction": 0.25,       # inner 25% of area
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
    """Two-row summary plot.

    Top row:  Median line + individual animal data points (per-animal view)
    Bottom row: Clean boxplots + Friedman test (repeated measures) with
                Wilcoxon signed-rank pairwise tests between consecutive days
    Uses only D1-D5 where all animals are present.
    """
    from matplotlib.lines import Line2D
    from scipy.stats import friedmanchisquare, wilcoxon

    # (column, label, unit_transform, y_label, ylim)
    metrics_to_plot = [
        ("total_distance_cm", "Total Distance", lambda x: x / 100, "Distance (m)", (0, 100)),
        ("median_speed_cms", "Median Speed\n(all frames)", lambda x: x, "Speed (cm/s)", (0, 5)),
        ("median_locomotion_speed_cms", "Median Locomotion Speed\n(excl. immobile)", lambda x: x, "Speed (cm/s)", (0, 8)),
        ("center_distance_cm", "Distance in Center", lambda x: x / 100, "Distance (m)", (0, 30)),
        ("arena_coverage_pct", "Arena Coverage", lambda x: x, "Coverage (%)", (0, 100)),
        ("immobility_time_s", "Immobility Time", lambda x: x / 60, "Time (min)", None),
    ]

    # Use only D1-D5 for statistics (complete repeated measures)
    stat_days = ["D1", "D2", "D3", "D4", "D5"]
    stat_df = results_df[results_df["Day"].isin(stat_days)]

    # Find animals present on ALL stat_days
    animal_counts = stat_df.groupby("Animal")["Day"].nunique()
    complete_animals = sorted(animal_counts[animal_counts == len(stat_days)].index.tolist())

    # All days for plotting (may include D6 etc.)
    all_days = sorted(results_df["Day"].unique())
    animals = sorted(results_df["Animal"].unique())
    n_metrics = len(metrics_to_plot)

    # Assign a unique color + marker combo to each animal (supports up to 10)
    cmap_colors = [
        "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
        "#a65628", "#f781bf", "#1b9e77", "#d95f02", "#666666",
    ]
    marker_list = ["o", "s", "D", "^", "v", "P", "X", "h", "*", "p"]
    animal_colors = {a: cmap_colors[i % len(cmap_colors)] for i, a in enumerate(animals)}
    animal_markers = {a: marker_list[i % len(marker_list)] for i, a in enumerate(animals)}

    fig, axes = plt.subplots(2, n_metrics, figsize=(28, 11))

    for col_idx, (col, title, transform, ylabel, ylim) in enumerate(metrics_to_plot):
        ax_top = axes[0, col_idx]
        ax_bot = axes[1, col_idx]

        data_by_day = [transform(results_df[results_df["Day"] == d][col].values)
                       for d in all_days]
        day_x = list(range(1, len(all_days) + 1))

        # ── Top row: median line + individual data points ──
        medians = [np.median(d) for d in data_by_day]
        ax_top.plot(day_x, medians, color="black", linewidth=1.5, zorder=2,
                    marker="_", markersize=12, markeredgewidth=2)

        for i, d in enumerate(all_days):
            day_df = results_df[results_df["Day"] == d].sort_values("Animal")
            n_animals = len(day_df)
            offsets = np.linspace(-0.2, 0.2, n_animals)
            for j, (_, row) in enumerate(day_df.iterrows()):
                animal = row["Animal"]
                val = transform(np.array([row[col]]))[0]
                ax_top.scatter(i + 1 + offsets[j], val,
                               c=animal_colors[animal], marker=animal_markers[animal],
                               s=55, alpha=0.9, edgecolors="black", linewidths=0.5,
                               zorder=3)

        ax_top.set_xticks(day_x)
        ax_top.set_xticklabels(all_days)
        ax_top.set_ylabel(ylabel)
        ax_top.set_title(title, fontsize=10)

        # ── Bottom row: clean boxplots + repeated-measures stats (D1-D5) ──
        bp = ax_bot.boxplot(data_by_day, tick_labels=all_days, widths=0.5,
                            patch_artist=True, showfliers=False,
                            boxprops=dict(facecolor="lightblue", alpha=0.5),
                            medianprops=dict(color="black", linewidth=2),
                            whiskerprops=dict(color="gray"),
                            capprops=dict(color="gray"))

        # Build paired data matrix for complete animals across stat_days
        if len(complete_animals) >= 5 and len(stat_days) >= 3:
            paired_data = {}  # day -> array of values in animal order
            for d in stat_days:
                day_vals = []
                for a in complete_animals:
                    row = stat_df[(stat_df["Day"] == d) & (stat_df["Animal"] == a)]
                    day_vals.append(transform(np.array([row[col].values[0]]))[0])
                paired_data[d] = np.array(day_vals)

            # Friedman test (repeated measures)
            friedman_arrays = [paired_data[d] for d in stat_days]
            chi2, p_fried = friedmanchisquare(*friedman_arrays)
            sig_color = "#e67e22" if p_fried < 0.05 else "#555555"
            ax_bot.text(0.98, 0.97,
                        f"Friedman (n={len(complete_animals)})\n"
                        f"\u03c7\u00b2 = {chi2:.1f}, {_fmt_p(p_fried)}",
                        transform=ax_bot.transAxes, fontsize=7, ha="right", va="top",
                        color=sig_color,
                        fontweight="bold" if p_fried < 0.05 else "normal",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                                  edgecolor=sig_color, alpha=0.8))

            # Wilcoxon signed-rank pairwise: consecutive days
            # Place brackets just above the highest whisker of each pair
            all_whisker_tops = []
            for dvals in data_by_day:
                if len(dvals) > 0:
                    q3 = np.percentile(dvals, 75)
                    iqr = q3 - np.percentile(dvals, 25)
                    whisker_top = min(q3 + 1.5 * iqr, np.max(dvals))
                    all_whisker_tops.append(whisker_top)
                else:
                    all_whisker_tops.append(0)

            bracket_gap = (max(all_whisker_tops) - min(all_whisker_tops)) * 0.08
            if bracket_gap == 0:
                bracket_gap = max(all_whisker_tops) * 0.05
            tick_h = bracket_gap * 0.3

            for k in range(len(stat_days) - 1):
                d1, d2 = stat_days[k], stat_days[k + 1]
                x1 = all_days.index(d1) + 1
                x2 = all_days.index(d2) + 1
                idx1 = all_days.index(d1)
                idx2 = all_days.index(d2)
                _, p_wil = wilcoxon(paired_data[d1], paired_data[d2])
                stars = _p_to_stars(p_wil)
                # Position just above the higher whisker of the two days
                top_val = max(all_whisker_tops[idx1], all_whisker_tops[idx2])
                by = top_val + bracket_gap + k * bracket_gap
                star_color = "#e67e22" if p_wil < 0.05 else "#bbbbbb"
                # Draw bracket
                ax_bot.plot([x1, x1, x2, x2], [by - tick_h, by, by, by - tick_h],
                            color=star_color, linewidth=1.0, clip_on=False)
                ax_bot.text((x1 + x2) / 2, by + tick_h * 0.3, stars,
                            ha="center", va="bottom", fontsize=7.5,
                            color=star_color,
                            fontweight="bold" if p_wil < 0.05 else "normal",
                            clip_on=False)

        ax_bot.set_ylabel(ylabel)
        ax_bot.set_xlabel("Day")

        # Apply ylims — top row uses data range, bottom row adds headroom for brackets
        if ylim is not None:
            ax_top.set_ylim(ylim)
            ax_bot.set_ylim(ylim[0], ylim[1] * 1.15)
        else:
            all_vals = transform(results_df[col].values)
            ax_top.set_ylim(0, np.max(all_vals) * 1.15)
            ax_bot.set_ylim(0, np.max(all_vals) * 1.30)

    # Shared legend for top row (animal identity)
    legend_handles = [
        Line2D([0], [0], marker=animal_markers[a], color="w",
               markerfacecolor=animal_colors[a], markeredgecolor="black",
               markersize=8, label=a)
        for a in animals
    ]
    axes[0, -1].legend(handles=legend_handles, loc="upper left",
                       title="Animal", fontsize=7, title_fontsize=8,
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

    fig.suptitle("Behavior Metrics Summary — Naive Baseline", fontsize=14, y=0.98)
    fig.tight_layout(rect=[0.03, 0, 0.93, 0.96])
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

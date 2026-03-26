"""Generate a high-resolution flowchart PNG of the DLC behavior analysis pipeline."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

# ── layout constants ──
FIG_W, FIG_H = 18, 56
DPI = 300

# section colours
C = {
    "batch":   ("#e0e7ff", "#4f46e5"),   # indigo for batch wrapper script
    "input":   ("#dbeafe", "#3b82f6"),
    "preproc": ("#fef3c7", "#f59e0b"),
    "convert": ("#d1fae5", "#10b981"),
    "metrics": ("#ede9fe", "#8b5cf6"),
    "output":  ("#fce7f3", "#ec4899"),
}


def draw_box(ax, x, y, w, h, text, fc="white", ec="#374151", fs=6.5):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                         facecolor=fc, edgecolor=ec, linewidth=1.5, zorder=2)
    ax.add_patch(box)
    lines = text.split("\n")
    spacing = 0.30
    total = len(lines) * spacing
    sy = y + h / 2 - total / 2 + spacing / 2
    for i, line in enumerate(lines):
        bold = (i == 0)
        ax.text(x + w / 2, sy + i * spacing, line,
                ha="center", va="center",
                fontsize=fs + 0.5 if bold else fs,
                fontweight="bold" if bold else "normal",
                fontfamily="sans-serif", zorder=3)


def draw_section(ax, x, y, w, h, label, fill, edge):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2",
                         facecolor=fill, edgecolor=edge, linewidth=2,
                         alpha=0.5, zorder=1)
    ax.add_patch(box)
    ax.text(x + 0.4, y + 0.3, label, fontsize=9,
            fontweight="bold", color=edge, zorder=3)


def arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color="#374151", lw=1.5),
                zorder=4)


def draw_dashed_line(ax, x1, y1, x2, y2, color="#374151"):
    ax.plot([x1, x2], [y1, y2], linestyle="--", color=color, lw=1.5, zorder=4)


def main():
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=DPI)
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, FIG_H)
    ax.invert_yaxis()
    ax.axis("off")

    CX = 9  # centre x for arrows

    # ══════════════════════════════════════════════════════════════════════
    # BATCH WRAPPER — DLC_batch_analysis.py  (separate script)
    # ══════════════════════════════════════════════════════════════════════
    draw_section(ax, 0.5, 0.3, 17, 9.4,
                 "BATCH WRAPPER  (DLC_batch_analysis.py — separate script)", *C["batch"])

    draw_box(ax, 2, 1.1, 14, 1.3,
             "Phase 1: Find Videos\nfind_videos()\nGlob *.mp4 in VIDEO_FOLDER, match by day tag (D1\u2013D5)",
             fc="#eef2ff", ec="#4f46e5")
    arrow(ax, CX, 2.4, CX, 2.8)

    draw_box(ax, 2, 2.8, 14, 1.8,
             "Phase 1: Interactive Corner Selection (per day)\nget_corners_from_user()\nOpen reference frame \u2192 user clicks 4 arena corners\n"
             "(TL, TR, BR, BL) \u2192 compute_homography() \u2192 save YAML",
             fc="#eef2ff", ec="#4f46e5")
    arrow(ax, CX, 4.6, CX, 5.0)

    draw_box(ax, 2, 5.0, 14, 2.2,
             "Phase 2: Warp All Videos (unattended)\nwarp_video() + cv2.warpPerspective()\nCorrects 2D camera-angle distortion: straightens the arena\n"
             "from a trapezoid into a rectangle (image-level transform)\n"
             "Output: {animal}/video_corrected.mp4 per day folder",
             fc="#eef2ff", ec="#4f46e5")
    arrow(ax, CX, 7.2, CX, 7.6)

    draw_box(ax, 2, 7.6, 14, 1.5,
             "Phase 3: Run DeepLabCut (unattended)\ndeeplabcut.analyze_videos()\nBatch inference on all corrected videos \u2192 DLC CSV outputs",
             fc="#eef2ff", ec="#4f46e5")

    # dashed separator between scripts
    ysep = 10.1
    draw_dashed_line(ax, 1, ysep, 17, ysep, color="#6b7280")
    ax.text(CX, ysep + 0.25, "DLC CSVs + arena YAML fed into dlc_behavior_analysis.py",
            ha="center", va="center", fontsize=7, fontstyle="italic", color="#6b7280", zorder=5)
    arrow(ax, CX, 9.1, CX, ysep - 0.15)
    arrow(ax, CX, ysep + 0.5, CX, ysep + 0.8)

    # ══════════════════════════════════════════════════════════════════════
    # ANALYSIS PIPELINE — dlc_behavior_analysis.py  (existing flowchart, shifted down)
    # ══════════════════════════════════════════════════════════════════════
    S = 10.6   # vertical shift for everything below

    # ── START ──
    draw_box(ax, 4, S + 0.3, 10, 1.3,
             "START: main()\nLoop over D1\u2013D5 day folders\nLoop over M01\u2013M09 animal dirs",
             fc="#f0f9ff", ec="#1e40af", fs=7.5)
    arrow(ax, CX, S + 1.6, CX, S + 2.2)

    # ── DATA LOADING ──
    draw_section(ax, 0.5, S + 2.2, 17, 3.6, "DATA LOADING", *C["input"])
    draw_box(ax, 1.2, S + 2.9, 7.2, 2.5,
             "load_dlc_csv()\nRead DLC CSV (multi-level header)\nExtract 3 bodyparts:\n"
             "  \u2022 mouse_center (primary)\n  \u2022 tail_base (fallback)\n  \u2022 nose\n"
             "Each with x, y, likelihood")
    draw_box(ax, 9.6, S + 2.9, 7.2, 2.5,
             "load_arena_corners()\nRead arena corner YAML\nCompute bounding box:\n"
             "  x_min, x_max\n  y_min, y_max")
    arrow(ax, 4.8, S + 5.4, CX, S + 6.2)
    arrow(ax, 13.2, S + 5.4, CX, S + 6.2)

    # ── PREPROCESSING ──
    draw_section(ax, 0.5, S + 6.2, 17, 22.6, "PREPROCESSING PIPELINE  (pixel space)", *C["preproc"])

    # Step 0
    y0 = S + 7.0
    draw_box(ax, 2, y0, 14, 3.0,
             "Step 0: Likelihood Fallback\napply_likelihood_fallback()\n"
             "For each frame where mouse_center likelihood < 0.6:\n"
             "  1. nose AND tail_base good \u2192 midpoint(nose, tail_base)\n"
             "  2. only tail_base good \u2192 tail_base + median offset\n"
             "  3. nothing reliable \u2192 keep original mouse_center\n"
             "Tracks which fallback method was used per frame")
    arrow(ax, CX, y0 + 3.0, CX, y0 + 3.5)

    # Step 1
    y1 = y0 + 3.5
    draw_box(ax, 2, y1, 14, 2.2,
             "Step 1: Jump Removal\nremove_position_jumps()\n"
             "Convert threshold 100 mm \u2192 pixels\n"
             "Flag frame-to-frame jumps > threshold\n"
             "Replace flagged frames with linear interpolation")
    arrow(ax, CX, y1 + 2.2, CX, y1 + 2.7)

    # Step 1b
    y1b = y1 + 2.7
    draw_box(ax, 2, y1b, 14, 2.5,
             "Step 1b: Iterative Speed-Based Cleaning\nclean_by_speed()  \u2014  up to 10 passes\n"
             "Temp convert to cm \u2192 compute speed\n"
             "Flag frames exceeding 50 cm/s (physiological max)\n"
             "Expand flagged windows \u00b13 frames each direction\n"
             "Interpolate, recompute speed, repeat until clean")
    arrow(ax, CX, y1b + 2.5, CX, y1b + 3.0)

    # Step 1c
    y1c = y1b + 3.0
    draw_box(ax, 2, y1c, 14, 2.5,
             "Step 1c: Out-of-Arena Episode Removal\nremove_out_of_arena_episodes()\n"
             "Find frames clearly outside arena (5% margin)\n"
             "Expand each detection to full excursion episode (15% margin)\n"
             "Removes entire drift episodes, not just individual frames\n"
             "Interpolate all flagged episodes")
    arrow(ax, CX, y1c + 2.5, CX, y1c + 3.0)

    # Step 2
    y2 = y1c + 3.0
    draw_box(ax, 2, y2, 14, 2.8,
             "Step 2: 3D Parallax Correction  (complements warp_video above)\ncorrect_perspective()\n"
             "Corrects for mouse body height (~2 cm) above arena floor\n"
             "Radial correction from arena center:\n"
             "  factor = (camera_height \u2013 mouse_height) / camera_height\n"
             "warp_video fixed 2D camera angle; this fixes 3D height parallax\n"
             "Also corrects arena bounds")
    arrow(ax, CX, y2 + 2.8, CX, y2 + 3.3)

    # Step 3
    y3 = y2 + 3.3
    draw_box(ax, 2, y3, 14, 1.4,
             "Step 3: Clip to Arena\nclip_to_arena()\n"
             "Clamp x, y to corrected arena bounding box")
    arrow(ax, CX, y3 + 1.4, CX, y3 + 1.9)

    # Step 3b
    y3b = y3 + 1.9
    draw_box(ax, 2, y3b, 14, 1.8,
             "Step 3b: Gaussian Smooth\nsmooth_positions()\n"
             "\u03c3 = 1 frame (scipy gaussian_filter1d)\n"
             "Reduces tracking jitter; re-clip after smoothing")
    arrow(ax, CX, y3b + 1.8, CX, y3b + 2.4)

    # ── UNIT CONVERSION ──
    yc = y3b + 2.4
    draw_section(ax, 0.5, yc, 17, 2.5, "UNIT CONVERSION", *C["convert"])
    draw_box(ax, 2, yc + 0.6, 14, 1.5,
             "Step 4: Convert to cm\nconvert_to_cm()\n"
             "scale = 30 cm / arena_span_px\n"
             "x_cm = (x \u2013 x_min) \u00d7 scale   y_cm = (y \u2013 y_min) \u00d7 scale")
    arrow(ax, CX, yc + 2.5, CX, yc + 3.1)

    # ── DERIVED METRICS ──
    ym = yc + 3.1
    draw_section(ax, 0.5, ym, 17, 7.4, "DERIVED METRICS", *C["metrics"])
    draw_box(ax, 4.5, ym + 0.7, 9, 1.3,
             "Step 5: Compute Speed\ncompute_speed() \u2014 windowed (5 frames), cm/s")

    ym_top = ym + 2.0
    # fan-out arrows to 6 metrics
    xs = [1.4, 4.2, 7.0, 9.8, 12.6, 15.4]
    for xd in xs:
        arrow(ax, CX, ym_top, xd, ym_top + 0.7)

    metrics = [
        (0.3,  "Total Distance\ncompute_total_\n  distance()\nSum of frame-to-\nframe Euclidean\ndists (cm)"),
        (3.1,  "Median Speed\nnp.median(speed)\nMedian of all\nframes (cm/s)\n(incl. immobile)"),
        (5.9,  "Med. Locomotion\nSpeed\nnp.median(speed\n  [moving])\nExcl. frames\n< 2 cm/s"),
        (8.7,  "Center Zone Dist.\ncompute_center_\n  zone_distance()\nDist. inside inner\n25% of arena"),
        (11.5, "Arena Coverage\ncompute_arena_\n  coverage()\n% of 1 cm \u00d7 1 cm\ngrid bins visited"),
        (14.3, "Immobility Time\ncompute_immobility\n  _time()\nSeconds where\nspeed < 2 cm/s"),
    ]
    for mx, mtxt in metrics:
        draw_box(ax, mx, ym_top + 0.7, 2.8, 3.4, mtxt, fs=5.0)

    # fan-in arrows to output
    yo = ym + 7.4 + 0.6
    ym_bot = ym_top + 0.7 + 3.4
    for mx, _ in metrics:
        arrow(ax, mx + 1.4, ym_bot, mx + 1.4, yo)

    # ── OUTPUT ──
    draw_section(ax, 0.5, yo, 17, 4.8, "OUTPUT", *C["output"])

    draw_box(ax, 1.0, yo + 0.7, 5.2, 3.6,
             "Pipeline Figure (per animal)\n9-panel plot:\n"
             "1. Raw trajectory\n2. After fallback\n"
             "3. After jump removal\n4. After speed cleaning\n"
             "5. After out-of-arena removal\n6. After perspective corr.\n"
             "7. After clipping\n8. Final trajectory (cm) + center zone\n"
             "9. Speed trace + thresholds\n"
             "+ Quality warning box", fs=5.5)

    draw_box(ax, 6.8, yo + 0.7, 4.5, 2.4,
             "behavior_metrics.csv\nAll animals \u00d7 all days\n15 columns:\n6 metrics + QC percentages")

    draw_box(ax, 12.0, yo + 0.7, 5.2, 2.4,
             "Summary Boxplots\n6 metrics \u00d7 D1\u2013D5\nColor-coded per animal\n(M01\u2013M09)")

    # CSV -> boxplots
    arrow(ax, 11.3, yo + 1.9, 12.0, yo + 1.9)

    out = r"D:\marcel\Miniscope_Zero_2025_11\v4_vs_MSzero_behavior\DLC_tracking_summary\dlc_behavior_analysis_flowchart.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white", pad_inches=0.3)
    plt.close(fig)

    from PIL import Image
    img = Image.open(out)
    print(f"Saved: {out}")
    print(f"Resolution: {img.size[0]} x {img.size[1]} px")


if __name__ == "__main__":
    main()

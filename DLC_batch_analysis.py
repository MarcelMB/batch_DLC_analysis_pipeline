"""
Batch DLC analysis pipeline for multiple days.
Steps:
  1. For each day tag, find all matching mp4s
  2. Pop up corner-click UI for ALL days first before any processing
  3. Then warp all videos and run DLC unattended
"""

import os
import re
import glob
import random
import cv2
import numpy as np
import yaml
import datetime
import deeplabcut

# ─────────────────────────────────────────────
# CONFIG — edit these
# ─────────────────────────────────────────────
VIDEO_FOLDER = r"C:\Users\marce\Downloads"
DLC_CONFIG   = r"C:\Users\marce\Documents\DeepLabCut\topmouse_project-marcel-2026-03-24\config.yaml"
DAY_TAGS     = ["D1", "D2", "D3", "D4", "D5"]  # add/remove days as needed
CAP_MINUTES  = 15                                # cap each video to this many minutes
OUTPUT_SIZE  = (1024, 768)                       # keep original frame size, no cropping
TEST_MODE    = False                             # True = 1 video per day only
# ─────────────────────────────────────────────


# ── Step 1: find videos ───────────────────────────────────────────────────────

def find_videos(folder, day_tag):
    all_mp4s = glob.glob(os.path.join(folder, "*.mp4"))
    matched  = [f for f in all_mp4s if re.search(rf"_{day_tag}[_\.]", os.path.basename(f), re.IGNORECASE)
                or os.path.basename(f).upper().endswith(f"_{day_tag}.MP4")]
    return sorted(matched)


# ── Step 2: interactive corner selection ─────────────────────────────────────

clicked_points = []

def _mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(clicked_points) < 4:
        clicked_points.append((x, y))
        print(f"  Corner {len(clicked_points)}: ({x}, {y})")

def get_corners_from_user(video_path, day_tag):
    """
    Opens first frame of video. User clicks 4 arena corners
    in order: TOP-LEFT, TOP-RIGHT, BOTTOM-RIGHT, BOTTOM-LEFT
    Press 'r' to reset, Enter to confirm.
    """
    global clicked_points
    clicked_points = []

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    assert ret, f"Could not read frame from {video_path}"

    display = frame.copy()
    window  = f"[{day_tag}] Click 4 corners: TL -> TR -> BR -> BL  |  r=reset  Enter=confirm"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 900, 700)
    cv2.setMouseCallback(window, _mouse_callback)

    print(f"\n[{day_tag}] Click corners: TOP-LEFT, TOP-RIGHT, BOTTOM-RIGHT, BOTTOM-LEFT")

    while True:
        vis = display.copy()
        for i, pt in enumerate(clicked_points):
            cv2.circle(vis, pt, 8, (0, 255, 0), -1)
            cv2.putText(vis, ["TL","TR","BR","BL"][i], (pt[0]+10, pt[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        if len(clicked_points) == 4:
            pts = np.array(clicked_points, dtype=np.int32)
            cv2.polylines(vis, [pts], isClosed=True, color=(0,255,0), thickness=2)
            cv2.putText(vis, "Press Enter to confirm or r to reset",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
        cv2.imshow(window, vis)
        key = cv2.waitKey(20) & 0xFF
        if key == ord('r'):
            clicked_points = []
            print(f"  [{day_tag}] Reset. Click 4 corners again.")
        elif key in (13, ord('q')) and len(clicked_points) == 4:
            break

    cv2.destroyAllWindows()
    return np.array(clicked_points, dtype=np.float32)


def compute_homography(src_corners):
    xs = src_corners[:, 0]
    ys = src_corners[:, 1]
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    dst = np.array([
        [x_min, y_min],
        [x_max, y_min],
        [x_max, y_max],
        [x_min, y_max],
    ], dtype=np.float32)
    H, _ = cv2.findHomography(src_corners, dst)
    return H


def save_corners(src_corners, day_tag, ref_video, output_folder):
    data = {
        "day_tag"        : day_tag,
        "date_saved"     : datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "reference_video": os.path.basename(ref_video),
        "arena_size_cm"  : 30,
        "corners": {
            "top_left"    : [int(src_corners[0][0]), int(src_corners[0][1])],
            "top_right"   : [int(src_corners[1][0]), int(src_corners[1][1])],
            "bottom_right": [int(src_corners[2][0]), int(src_corners[2][1])],
            "bottom_left" : [int(src_corners[3][0]), int(src_corners[3][1])],
        },
        "notes": "Corners clicked in order: TL, TR, BR, BL on uncorrected frame"
    }
    yaml_path = os.path.join(output_folder, f"arena_corners_{day_tag}.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)
    print(f"  Corners saved -> {yaml_path}")
    return yaml_path


# ── Step 3: warp videos ───────────────────────────────────────────────────────

def get_animal_folder(base_folder, video_path):
    basename  = os.path.basename(video_path)
    match     = re.search(r'_(M\d+)_', basename)
    animal_id = match.group(1) if match else "unknown"
    folder    = os.path.join(base_folder, animal_id)
    os.makedirs(folder, exist_ok=True)
    return folder, animal_id


def warp_video(src_path, dst_path, H):
    cap        = cv2.VideoCapture(src_path)
    fps_src    = cap.get(cv2.CAP_PROP_FPS)
    max_frames = int(CAP_MINUTES * 60 * fps_src)
    total      = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), max_frames)
    print(f"  Warping {os.path.basename(src_path)} ({total} frames @ {fps_src:.1f} fps = {CAP_MINUTES} min cap)...")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(dst_path, cv2.VideoWriter_fourcc(*"mp4v"), fps_src, OUTPUT_SIZE)

    frame_idx = 0
    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(cv2.warpPerspective(frame, H, OUTPUT_SIZE))
        frame_idx += 1
        if frame_idx % 500 == 0:
            print(f"    {frame_idx}/{total} frames done")

    cap.release()
    out.release()
    print(f"  Saved -> {dst_path}")


# ── Step 4: DLC analysis ──────────────────────────────────────────────────────

def run_dlc(video_list):
    print(f"\n── Running DeepLabCut on {len(video_list)} videos ──────────────────")
    deeplabcut.analyze_videos(
        DLC_CONFIG,
        video_list,
        videotype   = "mp4",
        shuffle     = 1,
        save_as_csv = True,
        destfolder  = None,
    )
    print("DLC analysis complete.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():

    # ── Phase 1: collect all videos and do ALL corner clicking upfront ────────
    print("=" * 60)
    print("PHASE 1 — Corner selection for all days")
    print("=" * 60)

    day_data = {}  # day_tag -> {videos, H, output_folder}

    for day_tag in DAY_TAGS:
        output_folder = os.path.join(VIDEO_FOLDER, f"{day_tag}_DLC")
        os.makedirs(output_folder, exist_ok=True)

        videos = find_videos(VIDEO_FOLDER, day_tag)
        if not videos:
            print(f"\n[{day_tag}] No videos found — skipping.")
            continue

        if TEST_MODE:
            videos = [random.choice(videos)]

        print(f"\n[{day_tag}] Found {len(videos)} videos:")
        for v in videos:
            print(f"  {os.path.basename(v)}")

        ref_video   = random.choice(videos)
        print(f"  Reference video: {os.path.basename(ref_video)}")
        src_corners = get_corners_from_user(ref_video, day_tag)
        H           = compute_homography(src_corners)
        save_corners(src_corners, day_tag, ref_video, output_folder)

        day_data[day_tag] = {
            "videos"       : videos,
            "H"            : H,
            "output_folder": output_folder,
        }

    if not day_data:
        print("No valid days found. Check VIDEO_FOLDER and DAY_TAGS.")
        return

    # ── Phase 2: warp all videos unattended ───────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 2 — Warping videos (unattended)")
    print("=" * 60)

    all_corrected = []

    for day_tag, data in day_data.items():
        print(f"\n── {day_tag}: warping {len(data['videos'])} videos ──")
        for vp in data["videos"]:
            animal_folder, _ = get_animal_folder(data["output_folder"], vp)
            stem     = os.path.splitext(os.path.basename(vp))[0]
            dst_path = os.path.join(animal_folder, f"{stem}_corrected.mp4")
            warp_video(vp, dst_path, data["H"])
            all_corrected.append(dst_path)

    # ── Phase 3: DLC on everything ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 3 — DLC analysis (unattended)")
    print("=" * 60)

    run_dlc(all_corrected)

    print("\n✓ All done.")


if __name__ == "__main__":
    main()
"""
Batch DLC analysis pipeline for a single day (e.g. D5)
Steps:
  1. Find all D5 mp4s in the folder
  2. Pick one randomly, pop up corner-click UI to define homography
  3. Warp all D5 videos using that homography -> save corrected videos
  4. Run DLC batch analysis (Faster R-CNN + HRNet-w32) on corrected videos
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
VIDEO_FOLDER   = r"C:\Users\marce\Downloads"
DLC_CONFIG     = r"C:\Users\marce\Documents\DeepLabCut\topmouse_project-marcel-2026-03-24\config.yaml"
DAY_TAG        = "D5"                         # change to D1, D2 etc as needed
OUTPUT_FOLDER  = os.path.join(VIDEO_FOLDER, f"{DAY_TAG}_DLC")
OUTPUT_SIZE    = (1024, 768)                  # keep original frame size, no cropping
CAP_MINUTES    = 15                           # cap each video to this many minutes
TEST_MODE      = True                         # True = run on 1 video only to verify, False = all videos
# ─────────────────────────────────────────────


# ── Step 1: find all videos matching the day tag ──────────────────────────────

def find_videos(folder, day_tag):
    pattern = os.path.join(folder, "*.mp4")
    all_mp4s = glob.glob(pattern)
    matched = [f for f in all_mp4s if re.search(rf"_{day_tag}[_\.]", os.path.basename(f), re.IGNORECASE)
               or os.path.basename(f).upper().endswith(f"_{day_tag}.MP4")]
    return sorted(matched)


# ── Step 2: interactive corner selection ─────────────────────────────────────

clicked_points = []

def _mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(clicked_points) < 4:
        clicked_points.append((x, y))
        print(f"  Corner {len(clicked_points)}: ({x}, {y})")

def get_corners_from_user(video_path):
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
    window  = "Click 4 arena corners: TL -> TR -> BR -> BL   |   r=reset  Enter=confirm"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 900, 700)
    cv2.setMouseCallback(window, _mouse_callback)

    print("\nClick corners in order: TOP-LEFT, TOP-RIGHT, BOTTOM-RIGHT, BOTTOM-LEFT")

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
            print("  Reset. Click 4 corners again.")
        elif key in (13, ord('q')) and len(clicked_points) == 4:
            break

    cv2.destroyAllWindows()
    return np.array(clicked_points, dtype=np.float32)


def compute_homography(src_corners, out_size):
    """
    Straighten the arena while keeping the full original frame size.
    The clicked corners are mapped to a rectangle that preserves their
    relative position in the frame — no cropping, black fill in empty corners.
    """
    xs = src_corners[:, 0]
    ys = src_corners[:, 1]
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    dst = np.array([
        [x_min, y_min],   # TL
        [x_max, y_min],   # TR
        [x_max, y_max],   # BR
        [x_min, y_max],   # BL
    ], dtype=np.float32)

    H, _ = cv2.findHomography(src_corners, dst)
    return H


def save_corners(src_corners, day_tag, ref_video, output_folder):
    """Save clicked corners to a YAML file for future reference."""
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


# ── Step 3: warp a single video ───────────────────────────────────────────────

def get_animal_folder(base_folder, video_path):
    """Extract animal ID from filename and create a subfolder for it.
    Filename format: {ID}_M{mouse#}_D{day}_{condition}.mp4
    Subfolder will be e.g. M01, M03 etc.
    """
    basename = os.path.basename(video_path)
    match = re.search(r'_(M\d+)_', basename)
    animal_id = match.group(1) if match else "unknown"
    folder = os.path.join(base_folder, animal_id)
    os.makedirs(folder, exist_ok=True)
    return folder, animal_id


def warp_video(src_path, dst_path, H, out_size):
    cap = cv2.VideoCapture(src_path)
    fps_src    = cap.get(cv2.CAP_PROP_FPS)
    max_frames = int(CAP_MINUTES * 60 * fps_src)
    total      = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), max_frames)
    print(f"  Warping {os.path.basename(src_path)}  ({total} frames @ {fps_src:.1f} fps = {CAP_MINUTES} min cap)...")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(dst_path, fourcc, fps_src, out_size)

    frame_idx = 0
    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        warped = cv2.warpPerspective(frame, H, out_size)
        out.write(warped)
        frame_idx += 1
        if frame_idx % 500 == 0:
            print(f"    {frame_idx}/{total} frames done")

    cap.release()
    out.release()
    print(f"  Saved -> {dst_path}")


# ── Step 4: DLC batch analysis ────────────────────────────────────────────────

def run_dlc(config_path, video_list):
    print("\n── Running DeepLabCut analysis ──────────────────────")
    deeplabcut.analyze_videos(
        config_path,
        video_list,
        videotype    = "mp4",
        shuffle      = 1,
        save_as_csv  = True,
        destfolder   = None,
    )
    print("DLC analysis complete.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # 1. find videos
    videos = find_videos(VIDEO_FOLDER, DAY_TAG)
    if not videos:
        raise FileNotFoundError(f"No mp4 files found with tag '{DAY_TAG}' in {VIDEO_FOLDER}")
    print(f"Found {len(videos)} videos for {DAY_TAG}:")
    for v in videos:
        print(f"  {os.path.basename(v)}")

    # test mode: only process one video
    if TEST_MODE:
        videos = [random.choice(videos)]
        print(f"\nTEST MODE: only processing {os.path.basename(videos[0])}")

    # 2. pick random video for corner selection
    ref_video = random.choice(videos)
    print(f"\nUsing reference video for corner selection:\n  {os.path.basename(ref_video)}")
    src_corners = get_corners_from_user(ref_video)
    H = compute_homography(src_corners, OUTPUT_SIZE)
    save_corners(src_corners, DAY_TAG, ref_video, OUTPUT_FOLDER)
    print(f"\nHomography matrix:\n{H}")

    # 3. warp all videos
    print(f"\n── Warping {len(videos)} videos ────────────────────────────")
    corrected_videos = []
    for vp in videos:
        animal_folder, animal_id = get_animal_folder(OUTPUT_FOLDER, vp)
        stem     = os.path.splitext(os.path.basename(vp))[0]
        dst_path = os.path.join(animal_folder, f"{stem}_corrected.mp4")
        warp_video(vp, dst_path, H, OUTPUT_SIZE)
        corrected_videos.append(dst_path)

    # 4. DLC
    run_dlc(DLC_CONFIG, corrected_videos)

    print("\n✓ All done.")


if __name__ == "__main__":
    main()
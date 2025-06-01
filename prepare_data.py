import cv2
import os
from sift import ThermalGrayAligner

def get_video_info(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return fps, frame_count, (width, height)

def extract_matched_frames(gray_path, thermal_path, out_gray_dir, out_thermal_dir, zoom_factor, shift, every_nth, prefix_gray="gray", prefix_thermal="thermal"):
    # Get video info
    gray_fps, gray_count, gray_res = get_video_info(gray_path)
    thermal_fps, thermal_count, thermal_res = get_video_info(thermal_path)
    print(f"Grayscale video: {gray_path}")
    print(f"  Frames: {gray_count}, Resolution: {gray_res}, FPS: {gray_fps}")
    print(f"Thermal video: {thermal_path}")
    print(f"  Frames: {thermal_count}, Resolution: {thermal_res}, FPS: {thermal_fps}")

    # Prepare output dirs
    os.makedirs(out_gray_dir, exist_ok=True)
    os.makedirs(out_thermal_dir, exist_ok=True)

    # Open videos
    gray_cap = cv2.VideoCapture(gray_path)
    thermal_cap = cv2.VideoCapture(thermal_path)

    # Build timestamp lists
    gray_timestamps = [i / gray_fps for i in range(gray_count)]
    thermal_timestamps = [i / thermal_fps for i in range(thermal_count)]

    # Match frames by nearest timestamp
    matched_gray_indices = []
    matched_thermal_indices = []
    discarded_gray = []
    discarded_thermal = list(range(thermal_count))  # Start with all, remove matched

    t_idx = 0
    for g_idx, g_time in enumerate(gray_timestamps):
        # Find closest thermal frame
        while t_idx + 1 < thermal_count and abs(thermal_timestamps[t_idx + 1] - g_time) < abs(thermal_timestamps[t_idx] - g_time):
            t_idx += 1
        # Accept match if within 1/(2*max_fps) seconds
        max_fps = max(gray_fps, thermal_fps)
        if abs(thermal_timestamps[t_idx] - g_time) <= 0.5 / max_fps:
            matched_gray_indices.append(g_idx)
            matched_thermal_indices.append(t_idx)
            if t_idx in discarded_thermal:
                discarded_thermal.remove(t_idx)
        else:
            discarded_gray.append(g_idx)

    print(f"Matched {len(matched_gray_indices)} frame pairs.")
    print(f"Discarded grayscale frames: {discarded_gray}")
    print(f"Discarded thermal frames: {discarded_thermal}")

    # Only take every 5000th matched index to minimize training data
    selected_indices = list(range(0, len(matched_gray_indices), every_nth))

    # Extract and save selected matched frames
    for i, idx in enumerate(selected_indices):
        g_idx = matched_gray_indices[idx]
        t_idx = matched_thermal_indices[idx]
        # Set frame positions
        gray_cap.set(cv2.CAP_PROP_POS_FRAMES, g_idx)
        thermal_cap.set(cv2.CAP_PROP_POS_FRAMES, t_idx)
        ret_g, frame_g = gray_cap.read()
        ret_t, frame_t = thermal_cap.read()
        if not (ret_g and ret_t):
            continue
        # Convert grayscale if needed
        if len(frame_g.shape) == 3 and frame_g.shape[2] == 3:
            frame_g = cv2.cvtColor(frame_g, cv2.COLOR_BGR2GRAY)
        if len(frame_t.shape) == 3 and frame_t.shape[2] == 3:
            frame_t = cv2.cvtColor(frame_t, cv2.COLOR_BGR2GRAY)
        
        # Resize grayscale to thermal resolution
        # cv2.imwrite(os.path.join(out_gray_dir, f"init_{prefix_gray}_frame{i:06d}.png"), frame_g)
        # cv2.imwrite(os.path.join(out_thermal_dir, f"init_{prefix_thermal}_frame{i:06d}.png"), frame_t)

        aligned_thermal = ThermalGrayAligner.flexible_crop(frame_t, zoom_factor, shift[0], shift[1], aspect_ratio=(16, 9))

        # 1. Get the shape of the aligned thermal image
        aligned_h, aligned_w = aligned_thermal.shape[:2]
        aligned_gray = cv2.resize(frame_g, (aligned_w, aligned_h), interpolation=cv2.INTER_AREA)

        # Save new frames
        cv2.imwrite(os.path.join(out_gray_dir, f"{i:06d}.png"), aligned_gray)
        cv2.imwrite(os.path.join(out_thermal_dir, f"{i:06d}.png"), aligned_thermal)

    print(f"Resized grayscale frames to resolution: {thermal_res}")
    print(f"Saved every 100th matched frame pair: {len(selected_indices)} pairs.")

    gray_cap.release()
    thermal_cap.release()

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print("Changed working directory to script location:", os.getcwd())

    # Define paths
    grayscale_video = "./data/gray_clipped.mp4"
    thermal_video = "./data/lynred_clipped.mp4"
    out_gray_dir = "./output/grayscale"
    out_thermal_dir = "./output/thermal"
    zoom_factor = 1.45
    shift = (-12, -16)
    every_nth = 60
    extract_matched_frames(grayscale_video, thermal_video, out_gray_dir, out_thermal_dir, zoom_factor, shift, every_nth)
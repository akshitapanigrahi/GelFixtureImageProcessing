from pathlib import Path
import cv2
import argparse
from load_config import load_config

def get_video_metadata(video_path):
    """
    Get total frames and fps from a video file.
    Falls back to `fallback_fps` if metadata cannot be read.

    Args:
        video_path (str or Path): Path to video file
        fallback_fps (int): Default fps if detection fails

    Returns:
        tuple: (total_frames, fps)
    """
    total_frames, fps = None, None
    if video_path and Path(video_path).exists():
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_val = cap.get(cv2.CAP_PROP_FPS)
        if fps_val > 0:
            fps = fps_val
        cap.release()
    return total_frames, fps

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    video = cfg.get("video_mp4", None)
    frames, fps = get_video_metadata(video)

    print(frames, fps)
    return frames, fps

if __name__ == "__main__":
    main()
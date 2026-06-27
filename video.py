"""
video.py — turn a real factory CCTV clip into frames the crew can reason over.

The VLM only accepts IMAGES, never raw video, so we always sample frames first.
extract_frames() walks a clip in time order and writes seqV_NN.jpg files that
the existing discovery logic in run.py / app.py picks up automatically.
"""

import os
import re

try:
    import cv2
    HAVE_CV2 = True
except ImportError:
    HAVE_CV2 = False


class VideoError(RuntimeError):
    pass


def extract_frames(video_path, every_n_seconds=2, out_dir=None, seq_name=None):
    """
    Sample frames from a video every `every_n_seconds` and save them as
    seqV_NN.jpg in `out_dir`. Returns the sorted list of written frame paths.

    video_path     : path to a .mp4 / .mov / .avi clip
    every_n_seconds : sampling interval in seconds of video time
    out_dir         : where to write frames (defaults to ./data)
    seq_name        : sequence prefix e.g. 'seq4' (auto-derived if None)
    """
    if not HAVE_CV2:
        raise VideoError(
            "opencv-python is required for video sampling. "
            "Install it with:  pip install opencv-python")
    if not os.path.exists(video_path):
        raise VideoError("Video not found: %s" % video_path)

    out_dir = out_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(out_dir, exist_ok=True)

    if seq_name is None:
        seq_name = _next_seq_name(out_dir)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise VideoError("Could not open video: %s" % video_path)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(fps * every_n_seconds)))

    written = []
    frame_idx = 0
    saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % step == 0:
            saved += 1
            out_path = os.path.join(out_dir, "%s_%02d.jpg" % (seq_name, saved))
            cv2.imwrite(out_path, frame)
            written.append(out_path)
        frame_idx += 1
    cap.release()

    if not written:
        raise VideoError("No frames extracted from %s" % video_path)
    return sorted(written)


def _next_seq_name(out_dir):
    """Pick the next free seqN prefix in out_dir."""
    nums = []
    for name in os.listdir(out_dir):
        m = re.match(r"seq(\d+)_", name, re.IGNORECASE)
        if m:
            nums.append(int(m.group(1)))
    return "seq%d" % ((max(nums) + 1) if nums else 1)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python3 video.py <clip.mp4> [every_n_seconds]")
        sys.exit(1)
    secs = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
    paths = extract_frames(sys.argv[1], every_n_seconds=secs)
    print("Wrote %d frames:" % len(paths))
    for p in paths:
        print("  " + p)

import io
import uuid
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch

# EchoNet-Dynamic model defaults
TARGET_SIZE = (112, 112)
NUM_FRAMES = 32


# -----------------------------------------------------------
# Utility: Check FFmpeg
# -----------------------------------------------------------
def check_ffmpeg_installed() -> bool:
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return True
    except FileNotFoundError:
        return False


# -----------------------------------------------------------
# FFmpeg Conversion: Normalize to EchoNet-compatible AVI
# -----------------------------------------------------------
def convert_to_echonet_avi(input_path: str) -> str:
    """
    Convert ANY input video into an EchoNet-compatible AVI:
      - RAW video (no compression)
      - Grayscale
      - Constant FPS (30)
      - Uncompressed frames
      - EchoNet-like pixel distribution
    """
    output_path = f"/tmp/{uuid.uuid4().hex}_echonet.avi"

    command = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-vf", "format=gray,fps=30",
        "-vcodec", "rawvideo",
        output_path,
    ]

    try:
        subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg EchoNet normalization failed: {e.stderr.decode()}")

    if not Path(output_path).exists():
        raise RuntimeError("EchoNet conversion produced no output file.")

    return output_path


# -----------------------------------------------------------
# Load Frames from Video (OpenCV)
# -----------------------------------------------------------
def load_video_frames(video_path: str) -> list:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale (EchoNet-style)
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        frame = cv2.resize(frame, TARGET_SIZE)
        frames.append(frame)

    cap.release()
    return frames


# -----------------------------------------------------------
# Normalize: Pad or Trim to 32 Frames
# -----------------------------------------------------------
def pad_or_trim_frames(frames: list) -> np.ndarray:
    if len(frames) == 0:
        raise RuntimeError("No frames extracted from video.")

    if len(frames) >= NUM_FRAMES:
        frames = frames[:NUM_FRAMES]
    else:
        last = frames[-1]
        while len(frames) < NUM_FRAMES:
            frames.append(last)

    return np.stack(frames, axis=0)  # (T, H, W)


# -----------------------------------------------------------
# Convert to Torch Tensor
# -----------------------------------------------------------
def frames_to_tensor(frames: np.ndarray) -> torch.Tensor:
    frames = frames.astype(np.float32) / 255.0
    frames = frames[None, None, ...]  # (1, 1, T, H, W)
    return torch.from_numpy(frames)


# -----------------------------------------------------------
# Public API: Preprocess Video File
# -----------------------------------------------------------
def preprocess_video_file(filepath: str) -> torch.Tensor:
    """
    Preprocess ANY video into EchoNet-compatible model input.
    Full pipeline:
      1. Try direct decode with OpenCV.
      2. If bad/empty/misformatted → normalize with FFmpeg.
      3. Load normalized frames.
      4. Resize, grayscale, normalize to 32 frames.
      5. Convert to model tensor.
    """

    # Step 1 — Try direct decode
    frames = load_video_frames(filepath)

    if len(frames) < 8:
        # Probably wrong codec, color space, VFR, etc.
        print("⚠️ Direct decode failed — normalizing via FFmpeg...")
        if not check_ffmpeg_installed():
            raise RuntimeError("FFmpeg not installed — cannot normalize video.")

        safe_avi = convert_to_echonet_avi(filepath)
        frames = load_video_frames(safe_avi)

    if len(frames) == 0:
        raise RuntimeError("Unable to extract frames even after normalization.")

    frames = pad_or_trim_frames(frames)
    return frames_to_tensor(frames)


# -----------------------------------------------------------
# Public API: Preprocess Uploaded Bytes
# -----------------------------------------------------------
def preprocess_video_bytes(file_bytes: bytes, filename: str) -> torch.Tensor:
    temp_input = Path(f"/tmp/{uuid.uuid4().hex}_{filename}")
    temp_input.write_bytes(file_bytes)
    return preprocess_video_file(str(temp_input))

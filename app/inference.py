import torch
import tempfile
import cv2
import numpy as np
import boto3
from pathlib import Path
from fastapi import UploadFile
from app.config import (
    S3_BUCKET,
    EF_MODEL_KEY,
    LOCAL_MODEL_PATH,
    DEVICE,
    DEFAULT_TARGET_SIZE,
    DEFAULT_MAX_FRAMES,
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    AWS_REGION,
    get_ef_severity
)

# Global model (loaded once at startup)
model = None


def download_model_from_s3():
    """Download model from S3 to local storage"""
    print(f"📥 Downloading model from s3://{S3_BUCKET}/{EF_MODEL_KEY}")
    
    s3_client = boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION
    )
    
    # Create directory if needed
    Path(LOCAL_MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
    
    # Download file
    s3_client.download_file(S3_BUCKET, EF_MODEL_KEY, LOCAL_MODEL_PATH)
    print(f"✅ Model downloaded to {LOCAL_MODEL_PATH}")


def load_model():
    """Load model at startup"""
    global model
    
    # Download from S3 if not exists locally
    if not Path(LOCAL_MODEL_PATH).exists():
        download_model_from_s3()
    
    print(f"🔄 Loading model from {LOCAL_MODEL_PATH}")
    print(f"🖥️  Using device: {DEVICE}")
    
    # Load model
    model = torch.load(LOCAL_MODEL_PATH, map_location=DEVICE)
    model.eval()
    
    print("✅ Model loaded successfully")


def preprocess_video(video_path: str) -> torch.Tensor:
    """Convert video to tensor for model input"""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    frames = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Sample frames evenly
    indices = np.linspace(0, frame_count - 1, DEFAULT_MAX_FRAMES, dtype=int)
    
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Resize to target size
            frame = cv2.resize(frame, DEFAULT_TARGET_SIZE)
            # Normalize to [0, 1]
            frame = frame / 255.0
            frames.append(frame)
    
    cap.release()
    
    if len(frames) == 0:
        raise ValueError("No frames extracted from video")
    
    print(f"📹 Extracted {len(frames)} frames from video")
    
    # Convert to tensor (T, H, W, C) -> (C, T, H, W)
    video_tensor = torch.FloatTensor(np.array(frames))
    video_tensor = video_tensor.permute(3, 0, 1, 2)  # (C, T, H, W)
    
    return video_tensor


async def predict_ef(video: UploadFile) -> dict:
    """Predict ejection fraction from video"""
    
    if model is None:
        raise RuntimeError("Model not loaded")
    
    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(video.filename).suffix) as tmp:
        content = await video.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        # Preprocess video
        video_tensor = preprocess_video(tmp_path)
        video_tensor = video_tensor.to(DEVICE)
        
        # Run inference
        with torch.no_grad():
            output = model(video_tensor.unsqueeze(0))
            ef_value = output.item()
        
        # Classify severity
        severity = get_ef_severity(ef_value)
        
        print(f"✅ Prediction: EF={ef_value:.2f}%, Severity={severity}")
        
        return {
            "ejection_fraction": round(ef_value, 2),
            "severity": severity,
            "filename": video.filename,
            "status": "success"
        }
    
    except Exception as e:
        print(f"❌ Prediction failed: {str(e)}")
        raise
    
    finally:
        # Cleanup temp file
        Path(tmp_path).unlink(missing_ok=True)


def is_model_loaded() -> bool:
    """Check if model is loaded"""
    return model is not None
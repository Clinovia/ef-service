import os
import torch

# -----------------------------
# FastAPI / Service
# -----------------------------
SERVICE_NAME = "Clinovia EF Microservice"
# Railway automatically sets PORT environment variable
PORT = int(os.environ.get("PORT", 8081))

# -----------------------------
# AWS S3 / Model Storage
# -----------------------------
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

S3_BUCKET = os.environ.get("CLINOVIA_S3_BUCKET", "clinovia.ai")
EF_MODEL_KEY = os.environ.get(
    "EF_MODEL_KEY", 
    "models/cardiology/ejection-fraction/v1/ef3dcnn_epoch17.pth"
)

# Local model path (downloaded from S3 at startup)
LOCAL_MODEL_PATH = os.environ.get("LOCAL_MODEL_PATH", "/tmp/ef_model.pth")

# -----------------------------
# Device
# -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Video Processing
# -----------------------------
DEFAULT_TARGET_SIZE = (112, 112)
DEFAULT_MAX_FRAMES = 32
MAX_VIDEO_SIZE_MB = int(os.environ.get("MAX_VIDEO_SIZE_MB", "100"))

# -----------------------------
# Clinical Thresholds
# -----------------------------
EF_THRESHOLDS = {
    "normal": 50,      # EF >= 50%
    "mild": 40,        # 40% <= EF < 50%
    "moderate": 30,    # 30% <= EF < 40%
    # severe: EF < 30%
}

def get_ef_severity(ef_value: float) -> str:
    """Classify EF severity"""
    if ef_value >= EF_THRESHOLDS["normal"]:
        return "normal"
    elif ef_value >= EF_THRESHOLDS["mild"]:
        return "mild_dysfunction"
    elif ef_value >= EF_THRESHOLDS["moderate"]:
        return "moderate_dysfunction"
    else:
        return "severe_dysfunction"
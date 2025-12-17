# app/config.py
import os
import torch
from pathlib import Path
from typing import Tuple, Dict
from dotenv import load_dotenv

# -----------------------------
# Load .env automatically
# -----------------------------
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
    print(f"✅ Loaded environment variables from {env_path}")
else:
    print("ℹ️  No .env file found; using system environment variables")

# -----------------------------
# Settings Class
# -----------------------------
class Settings:
    # Service
    SERVICE_NAME: str = "Clinovia EF Microservice"
    PORT: int = int(os.environ.get("PORT") or 8081)

    # AWS S3
    AWS_ACCESS_KEY_ID: str = os.environ.get("AWS_ACCESS_KEY_ID") or ""
    AWS_SECRET_ACCESS_KEY: str = os.environ.get("AWS_SECRET_ACCESS_KEY") or ""
    AWS_REGION: str = os.environ.get("AWS_DEFAULT_REGION") or "us-east-1"
    S3_BUCKET: str = os.environ.get("CLINOVIA_S3_BUCKET") or "clinovia.ai"

    # Backend + EF Service
    BACKEND_URL: str = os.environ.get("BACKEND_URL") or "http://localhost:8000"
    EF_SERVICE_TOKEN: str = os.environ.get("EF_SERVICE_TOKEN") or ""

    # Model
    LOCAL_MODEL_PATH: str = os.environ.get(
        "EF_LOCAL_MODEL_PATH"
    ) or "/tmp/ef_model.pth"

    # PyTorch device
    DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Video preprocessing
    DEFAULT_TARGET_SIZE: Tuple[int, int] = (112, 112)
    DEFAULT_MAX_FRAMES: int = 32
    MAX_VIDEO_SIZE_MB: int = int(os.environ.get("MAX_VIDEO_SIZE_MB") or 100)

    # EF thresholds
    EF_THRESHOLDS: Dict[str, int] = {
        "normal": 50,
        "mild": 40,
        "moderate": 30
    }

    # -----------------------------
    # EF Severity Helper
    # -----------------------------
    @staticmethod
    def get_ef_severity(ef_value: float) -> str:
        if ef_value >= Settings.EF_THRESHOLDS["normal"]:
            return "normal"
        elif ef_value >= Settings.EF_THRESHOLDS["mild"]:
            return "mild_dysfunction"
        elif ef_value >= Settings.EF_THRESHOLDS["moderate"]:
            return "moderate_dysfunction"
        else:
            return "severe_dysfunction"

    # -----------------------------
    # Validation
    # -----------------------------
    @classmethod
    def validate(cls) -> bool:
        errors = []
        if not cls.LOCAL_MODEL_PATH:
            errors.append("EF_LOCAL_MODEL_PATH not set")
        if not cls.AWS_ACCESS_KEY_ID or not cls.AWS_SECRET_ACCESS_KEY:
            errors.append("AWS credentials missing")
        if errors:
            raise ValueError("Configuration errors:\n" + "\n".join(errors))
        return True

    # -----------------------------
    # Print Config
    # -----------------------------
    @classmethod
    def print_config(cls) -> None:
        masked_aws_key = (
            cls.AWS_ACCESS_KEY_ID[:4] + "*" * 12 + cls.AWS_ACCESS_KEY_ID[-4:]
            if cls.AWS_ACCESS_KEY_ID else "❌ NOT SET"
        )
        print("\n" + "="*50)
        print(f"Service: {cls.SERVICE_NAME}")
        print(f"Port: {cls.PORT}")
        print(f"Device: {cls.DEVICE}")
        print(f"Local model path: {cls.LOCAL_MODEL_PATH}")
        print(f"S3 Bucket: {cls.S3_BUCKET}")
        print(f"AWS Access Key: {masked_aws_key}")
        print("="*50 + "\n")


# -----------------------------
# Singleton instance
# -----------------------------
settings: Settings = Settings()
settings.print_config()
try:
    settings.validate()
    print("✅ Configuration validated")
except ValueError as e:
    print(f"⚠️ Configuration warning:\n{e}")

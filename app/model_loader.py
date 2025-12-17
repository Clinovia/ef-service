# app/model_loader.py
import os
import tempfile
from typing import Optional
import torch
import boto3
from botocore.exceptions import ClientError, NoCredentialsError

from .config import settings
from .models.ef_model import EF3DCNN


def _create_model(device: torch.device) -> EF3DCNN:
    """Initialize EF3DCNN with architecture parameters from settings."""
    H, W = settings.DEFAULT_TARGET_SIZE
    return EF3DCNN(
        in_channels=1,
        T=settings.DEFAULT_MAX_FRAMES,
        H=H,
        W=W
    ).to(device)


def _load_state_dict(model: EF3DCNN, state_dict: dict) -> None:
    """Load model weights from state_dict dict with multiple possible keys."""
    if isinstance(state_dict, dict):
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        elif "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
    model.load_state_dict(state_dict)


def get_s3_client() -> boto3.client:
    """Return a singleton boto3 S3 client configured from settings."""
    if not hasattr(get_s3_client, "_client"):
        if not settings.AWS_ACCESS_KEY_ID or not settings.AWS_SECRET_ACCESS_KEY:
            raise ValueError("AWS credentials missing in settings")
        get_s3_client._client = boto3.client(
            "s3",
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            region_name=settings.AWS_REGION,
        )
    return get_s3_client._client  # type: ignore


def load_ef_model(local_path: Optional[str] = None, device: Optional[torch.device] = None) -> EF3DCNN:
    """
    Load EF3DCNN model weights.
    1️⃣ First tries local_path (defaults to EF_LOCAL_MODEL_PATH)
    2️⃣ Falls back to S3 if local file not found
    """
    device = device or settings.DEVICE
    local_path = local_path or settings.LOCAL_MODEL_PATH

    # -----------------------------
    # Load from local path
    # -----------------------------
    if os.path.exists(local_path):
        print(f"📥 Loading EF model from local path: {local_path}")
        model = _create_model(device)
        state_dict = torch.load(local_path, map_location=device, weights_only=True)
        _load_state_dict(model, state_dict)
        model.eval()
        print(f"✅ EF model loaded successfully from local path on {device}")
        return model

    # -----------------------------
    # Fallback: Load from S3
    # -----------------------------
    if not settings.S3_BUCKET:
        raise ValueError("S3_BUCKET not set in settings; cannot load model from S3")

    print(f"ℹ️ Local model not found; attempting S3 download: s3://{settings.S3_BUCKET}/{settings.EF_MODEL_KEY}")

    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pth")
    tmp_file.close()
    try:
        s3 = get_s3_client()
        s3.download_file(settings.S3_BUCKET, settings.EF_MODEL_KEY, tmp_file.name)
        print(f"⬇️  Downloaded EF model from S3 to temporary path {tmp_file.name}")
        model = _create_model(device)
        state_dict = torch.load(tmp_file.name, map_location=device, weights_only=True)
        _load_state_dict(model, state_dict)
        model.eval()
        print(f"✅ EF model loaded successfully from S3 on {device}")
        return model
    except NoCredentialsError as e:
        raise RuntimeError("AWS credentials missing or invalid") from e
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "Unknown")
        raise RuntimeError(f"S3 download failed ({code}): {e}") from e
    finally:
        try:
            if os.path.exists(tmp_file.name):
                os.remove(tmp_file.name)
                print("🗑️ Temporary file removed")
        except Exception as err:
            print(f"⚠️ Cleanup warning: {err}")

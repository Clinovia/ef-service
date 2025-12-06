# Refactored EF model loader

import os
import tempfile
from typing import Optional

import boto3
import torch
from botocore.exceptions import ClientError, NoCredentialsError

from .config import settings
from .models.ef_model import EF3DCNN

# Singleton S3 client
_s3_client = None


def get_s3_client():
    """
    Returns a singleton boto3 S3 client configured with project AWS credentials.
    """
    global _s3_client

    if _s3_client is None:
        if not settings.AWS_ACCESS_KEY_ID or not settings.AWS_SECRET_ACCESS_KEY:
            raise ValueError(
                "AWS credentials missing. Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY."
            )

        _s3_client = boto3.client(
            's3',
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            region_name=settings.AWS_REGION,
        )

    return _s3_client


def _create_model(device: str):
    """
    Initializes EF3DCNN with correct architecture parameters.
    """
    H, W = settings.DEFAULT_TARGET_SIZE
    return EF3DCNN(
        in_channels=1,
        T=settings.DEFAULT_MAX_FRAMES,
        H=H,
        W=W,
    ).to(device)


def _load_state_dict(model, state_dict):
    """
    Cleanly handle multiple possible checkpoint formats.
    """
    if isinstance(state_dict, dict):
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        elif 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']

    model.load_state_dict(state_dict)


def load_ef_model(model_key: str, device: Optional[str] = None) -> torch.nn.Module:
    """
    Load EF3DCNN model weights from S3 and return initialized model.
    """
    device = device or str(settings.DEVICE)

    if not settings.S3_BUCKET:
        raise ValueError(
            "S3 bucket not configured. Set CLINOVIA_S3_BUCKET environment variable."
        )

    print(f"📥 Loading EF model from S3: s3://{settings.S3_BUCKET}/{model_key}")

    # Initialize model
    model = _create_model(device)

    # Temporary download path
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pth")
    tmp_file.close()

    try:
        s3 = get_s3_client()

        print(f"⬇️  Downloading model to {tmp_file.name}...")
        s3.download_file(settings.S3_BUCKET, model_key, tmp_file.name)

        file_size_mb = os.path.getsize(tmp_file.name) / (1024 * 1024)
        print(f"📊 Downloaded {file_size_mb:.2f} MB")

        state_dict = torch.load(tmp_file.name, map_location=device, weights_only=True)
        print("⚙️  Loading model weights...")

        _load_state_dict(model, state_dict)
        print("✅ Model weights loaded successfully")

    except NoCredentialsError as e:
        raise RuntimeError("AWS credentials missing.") from e

    except ClientError as e:
        code = e.response.get('Error', {}).get('Code', 'Unknown')
        msg = e.response.get('Error', {}).get('Message', str(e))

        if code == 'NoSuchKey':
            raise RuntimeError(
                f"Model not found in S3: s3://{settings.S3_BUCKET}/{model_key}"
            ) from e
        if code in ('403', 'AccessDenied'):
            raise RuntimeError(
                "S3 access denied. Check IAM permissions."
            ) from e
        raise RuntimeError(f"S3 error ({code}): {msg}") from e

    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}") from e

    finally:
        try:
            if os.path.exists(tmp_file.name):
                os.remove(tmp_file.name)
                print("🗑️  Temp file removed")
        except Exception as err:
            print(f"⚠️  Cleanup warning: {err}")

    model.eval()
    print(f"✅ Model ready on {device}")
    return model


def load_ef_model_from_local(local_path: str, device: Optional[str] = None) -> torch.nn.Module:
    """
    Load EF3DCNN model from a local .pth file.
    """
    device = device or str(settings.DEVICE)

    if not os.path.exists(local_path):
        raise FileNotFoundError(f"Local model file not found: {local_path}")

    print(f"📥 Loading EF model from local path: {local_path}")

    try:
        model = _create_model(device)

        state_dict = torch.load(local_path, map_location=device, weights_only=True)
        _load_state_dict(model, state_dict)

        model.eval()
        print(f"✅ Local model loaded successfully on {device}")
        return model

    except Exception as e:
        raise RuntimeError(f"Failed to load local model: {e}") from e


def verify_s3_access(bucket: Optional[str] = None, key: Optional[str] = None) -> bool:
    """
    Verify S3 bucket + key exist and are readable.
    """
    bucket = bucket or settings.S3_BUCKET
    key = key or settings.EF_MODEL_KEY

    try:
        s3 = get_s3_client()
        result = s3.head_object(Bucket=bucket, Key=key)

        size_mb = result['ContentLength'] / (1024 * 1024)
        print(f"✅ S3 object accessible: s3://{bucket}/{key} ({size_mb:.2f} MB)")
        return True

    except ClientError as e:
        code = e.response.get('Error', {}).get('Code', 'Unknown')
        print(f"❌ S3 access failed ({code}): {e}")
        return False

    except Exception as e:
        print(f"❌ Error verifying S3 access: {e}")
        return False

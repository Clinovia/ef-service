import os
import tempfile
import boto3
import torch
from typing import Optional
from botocore.exceptions import ClientError, NoCredentialsError

from .config import settings
from .models.ef_model import EF3DCNN


# Initialize S3 client
_s3_client = None


def get_s3_client():
    """
    Get or create S3 client singleton.
    
    Returns:
        boto3.client: Configured S3 client
    """
    global _s3_client
    
    if _s3_client is None:
        if not settings.AWS_ACCESS_KEY_ID or not settings.AWS_SECRET_ACCESS_KEY:
            raise ValueError(
                "AWS credentials not configured. "
                "Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables."
            )
        
        _s3_client = boto3.client(
            's3',
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            region_name=settings.AWS_REGION
        )
    
    return _s3_client


def load_ef_model(model_key: str, device: Optional[str] = None) -> torch.nn.Module:
    """
    Load EF3DCNN PyTorch model from S3.
    
    Args:
        model_key: S3 key to model file (e.g., "models/cardiology/ef/v1/ef3dcnn_epoch17.pth")
        device: "cpu" or "cuda". If None, uses settings.DEVICE
        
    Returns:
        Loaded and initialized EF3DCNN model in eval mode
        
    Raises:
        ValueError: If S3 bucket or AWS credentials not configured
        RuntimeError: If model loading fails
    """
    # Use device from settings if not provided
    if device is None:
        device = str(settings.DEVICE)
    
    # Validate S3 configuration
    if not settings.S3_BUCKET:
        raise ValueError(
            "S3_BUCKET not configured. Set CLINOVIA_S3_BUCKET environment variable."
        )
    
    print(f"📥 Loading EF model from S3: s3://{settings.S3_BUCKET}/{model_key}")
    
    # Initialize model architecture
    model = EF3DCNN()
    
    # Create temporary file for download
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pth")
    tmp_file.close()
    
    try:
        # Get S3 client
        s3_client = get_s3_client()
        
        # Download model from S3
        print(f"⬇️  Downloading model to {tmp_file.name}...")
        s3_client.download_file(
            Bucket=settings.S3_BUCKET,
            Key=model_key,
            Filename=tmp_file.name
        )
        
        # Check file size
        file_size_mb = os.path.getsize(tmp_file.name) / (1024 * 1024)
        print(f"📊 Downloaded {file_size_mb:.2f} MB")
        
        # Load state dict
        print(f"⚙️  Loading model weights...")
        state_dict = torch.load(
            tmp_file.name,
            map_location=device,
            weights_only=True
        )
        
        # Handle different state dict formats
        if isinstance(state_dict, dict):
            # Check for common checkpoint formats
            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            elif 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
        
        # Load weights into model
        model.load_state_dict(state_dict)
        print("✅ Model weights loaded successfully")
        
    except NoCredentialsError as e:
        raise RuntimeError(
            "AWS credentials not found. Check AWS_ACCESS_KEY_ID and "
            "AWS_SECRET_ACCESS_KEY environment variables."
        ) from e
        
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        error_msg = e.response.get('Error', {}).get('Message', str(e))
        
        if error_code == 'NoSuchKey':
            raise RuntimeError(
                f"Model file not found in S3: s3://{settings.S3_BUCKET}/{model_key}"
            ) from e
        elif error_code == '403' or error_code == 'AccessDenied':
            raise RuntimeError(
                f"Access denied to S3 bucket '{settings.S3_BUCKET}'. "
                "Check AWS credentials and bucket permissions."
            ) from e
        else:
            raise RuntimeError(
                f"S3 error ({error_code}): {error_msg}"
            ) from e
            
    except Exception as e:
        raise RuntimeError(f"Failed to load EF model from S3: {e}") from e
        
    finally:
        # Clean up temporary file
        try:
            if os.path.exists(tmp_file.name):
                os.remove(tmp_file.name)
                print(f"🗑️  Cleaned up temp file")
        except Exception as cleanup_error:
            print(f"⚠️  Warning: Failed to clean up temp file: {cleanup_error}")
    
    # Move model to device and set to evaluation mode
    model.to(device)
    model.eval()
    
    print(f"✅ Model ready on {device}")
    return model


def load_ef_model_from_local(local_path: str, device: Optional[str] = None) -> torch.nn.Module:
    """
    Load EF3DCNN model from local file (useful for testing/development).
    
    Args:
        local_path: Path to local .pth file
        device: "cpu" or "cuda". If None, uses settings.DEVICE
        
    Returns:
        Loaded and initialized EF3DCNN model in eval mode
        
    Raises:
        FileNotFoundError: If local file doesn't exist
        RuntimeError: If model loading fails
    """
    if device is None:
        device = str(settings.DEVICE)
    
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"Model file not found: {local_path}")
    
    print(f"📥 Loading EF model from local file: {local_path}")
    
    try:
        # Initialize model
        model = EF3DCNN()
        
        # Load state dict
        state_dict = torch.load(
            local_path,
            map_location=device,
            weights_only=True
        )
        
        # Handle different formats
        if isinstance(state_dict, dict):
            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            elif 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
        
        # Load weights
        model.load_state_dict(state_dict)
        
        # Move to device and eval mode
        model.to(device)
        model.eval()
        
        print(f"✅ Model loaded successfully on {device}")
        return model
        
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {local_path}: {e}") from e


def verify_s3_access(bucket: Optional[str] = None, key: Optional[str] = None) -> bool:
    """
    Verify that S3 bucket and object are accessible.
    
    Args:
        bucket: S3 bucket name (uses settings.S3_BUCKET if None)
        key: S3 object key (uses settings.EF_MODEL_KEY if None)
        
    Returns:
        bool: True if accessible, False otherwise
    """
    bucket = bucket or settings.S3_BUCKET
    key = key or settings.EF_MODEL_KEY
    
    try:
        s3_client = get_s3_client()
        
        # Try to get object metadata
        response = s3_client.head_object(Bucket=bucket, Key=key)
        
        size_mb = response['ContentLength'] / (1024 * 1024)
        print(f"✅ S3 object accessible: s3://{bucket}/{key} ({size_mb:.2f} MB)")
        return True
        
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        print(f"❌ S3 access failed ({error_code}): {e}")
        return False
        
    except Exception as e:
        print(f"❌ Error verifying S3 access: {e}")
        return False
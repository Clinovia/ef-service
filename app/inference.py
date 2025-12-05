import io
import torch
from typing import Optional, Dict, Any
from fastapi import UploadFile, HTTPException

from .model_loader import load_ef_model
from .config import settings


# -----------------------------
# Global Model State
# -----------------------------
_model: Optional[torch.nn.Module] = None
_model_initialized: bool = False


# -----------------------------
# Model Management
# -----------------------------
def is_model_loaded() -> bool:
    """
    Check if the model has been successfully loaded into memory.
    
    Returns:
        bool: True if model is loaded and ready for inference, False otherwise
    """
    return _model_initialized and _model is not None


def initialize_model() -> None:
    """
    Loads the EF prediction model from S3 or local cache.
    
    Raises:
        RuntimeError: If model fails to load
    """
    global _model, _model_initialized
    
    if _model_initialized:
        print("✅ Model already initialized")
        return
    
    try:
        print("🚀 Starting Clinovia EF Microservice")
        print(f"📥 Loading model: {settings.EF_MODEL_KEY}")
        
        # Load model using model_loader
        _model = load_ef_model(settings.EF_MODEL_KEY)
        
        if _model is None:
            raise RuntimeError("Model loader returned None")
        
        # Move model to appropriate device and set to eval mode
        _model.to(settings.DEVICE)
        _model.eval()
        
        _model_initialized = True
        print(f"✅ Model successfully loaded on {settings.DEVICE}")
        
    except Exception as e:
        _model_initialized = False
        _model = None
        print(f"❌ Model failed to load: {e}")
        raise RuntimeError(f"Failed to initialize model: {e}")


def get_model() -> torch.nn.Module:
    """
    Get the loaded model instance.
    
    Returns:
        torch.nn.Module: The loaded model
        
    Raises:
        RuntimeError: If model is not initialized
    """
    if not is_model_loaded():
        raise RuntimeError(
            "Model not initialized. Call initialize_model() first."
        )
    return _model


# -----------------------------
# Video Preprocessing
# -----------------------------
def preprocess_video(file_bytes: bytes) -> torch.Tensor:
    """
    Preprocess video file bytes into model input tensor.
    
    Args:
        file_bytes: Raw bytes from uploaded video file
        
    Returns:
        torch.Tensor: Preprocessed video tensor ready for inference
                     Shape: (batch, channels, frames, height, width)
                     
    TODO: Implement real preprocessing logic:
        - Decode video frames
        - Sample frames uniformly (DEFAULT_MAX_FRAMES)
        - Resize to DEFAULT_TARGET_SIZE
        - Normalize pixel values
        - Convert to tensor
    """
    # Placeholder implementation
    # Replace with actual video processing pipeline
    tensor = torch.randn(
        1,  # batch size
        3,  # RGB channels
        settings.DEFAULT_MAX_FRAMES,  # temporal dimension
        settings.DEFAULT_TARGET_SIZE[0],  # height
        settings.DEFAULT_TARGET_SIZE[1]   # width
    )
    
    return tensor.to(settings.DEVICE)


# -----------------------------
# Inference
# -----------------------------
def predict_ef(video_tensor: torch.Tensor) -> Dict[str, Any]:
    """
    Run model inference to predict ejection fraction.
    
    Args:
        video_tensor: Preprocessed video tensor
        
    Returns:
        dict: Prediction results containing:
            - ef_value (float): Predicted EF percentage
            - severity (str): Clinical severity classification
            - confidence (float, optional): Model confidence score
            
    Raises:
        RuntimeError: If model is not initialized
        ValueError: If input tensor has invalid shape
    """
    model = get_model()
    
    # Validate input shape
    expected_shape = (
        1,
        3,
        settings.DEFAULT_MAX_FRAMES,
        settings.DEFAULT_TARGET_SIZE[0],
        settings.DEFAULT_TARGET_SIZE[1]
    )
    
    if video_tensor.shape != expected_shape:
        raise ValueError(
            f"Invalid input shape. Expected {expected_shape}, "
            f"got {video_tensor.shape}"
        )
    
    try:
        with torch.no_grad():
            output = model(video_tensor)
            
            # Extract EF value (adjust based on your model's output format)
            if output.dim() == 0:
                ef_value = float(output.item())
            elif output.dim() == 1:
                ef_value = float(output[0].item())
            else:
                ef_value = float(output[0, 0].item())
            
            # Clamp to valid EF range (0-100%)
            ef_value = max(0.0, min(100.0, ef_value))
            
            # Get clinical severity classification
            severity = settings.get_ef_severity(ef_value)
            
            return {
                "ef_value": round(ef_value, 2),
                "severity": severity,
                "confidence": None  # Add if your model provides this
            }
            
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        raise RuntimeError(f"Prediction failed: {e}")


# -----------------------------
# FastAPI Endpoint Handler
# -----------------------------
async def run_inference(file: UploadFile) -> Dict[str, Any]:
    """
    Main inference pipeline for FastAPI endpoint.
    
    Args:
        file: Uploaded video file from FastAPI
        
    Returns:
        dict: Inference results with EF prediction and metadata
        
    Raises:
        HTTPException: If inference fails
    """
    if not is_model_loaded():
        raise HTTPException(
            status_code=503,
            detail="Model not ready. Server is initializing."
        )
    
    # Validate file type
    if not file.content_type.startswith('video/'):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Expected video file."
        )
    
    try:
        # Read file bytes
        file_bytes = await file.read()
        
        # Check file size
        file_size_mb = len(file_bytes) / (1024 * 1024)
        if file_size_mb > settings.MAX_VIDEO_SIZE_MB:
            raise HTTPException(
                status_code=413,
                detail=f"File too large: {file_size_mb:.2f}MB. "
                       f"Max size: {settings.MAX_VIDEO_SIZE_MB}MB"
            )
        
        # Preprocess video
        video_tensor = preprocess_video(file_bytes)
        
        # Run inference
        prediction = predict_ef(video_tensor)
        
        # Add metadata
        result = {
            **prediction,
            "filename": file.filename,
            "file_size_mb": round(file_size_mb, 2),
            "model_version": settings.EF_MODEL_KEY.split('/')[-1],
        }
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Inference pipeline failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {str(e)}"
        )
    finally:
        # Clean up
        await file.close()
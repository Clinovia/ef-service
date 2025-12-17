import io
import torch
from typing import Optional, Dict, Any
from fastapi import UploadFile, HTTPException

from .model_loader import load_ef_model
from .config import settings
from .preprocessing import preprocess_video_bytes


# ============================================================
# Global Model State
# ============================================================

_model: Optional[torch.nn.Module] = None
_model_initialized: bool = False


# ============================================================
# Model Management
# ============================================================

def is_model_loaded() -> bool:
    """Return True if the EF model has been loaded into memory."""
    return _model_initialized and _model is not None


def initialize_model() -> None:
    """
    Load the EF model from local path, place it on the configured device,
    and set it to eval mode.
    """
    global _model, _model_initialized

    if _model_initialized:
        print("⚠️ Model already initialized — skipping reload")
        return

    try:
        print(f"🚀 Initializing EF Model: {settings.LOCAL_MODEL_PATH}")
        model = load_ef_model(settings.LOCAL_MODEL_PATH, device=str(settings.DEVICE))

        if model is None:
            raise RuntimeError("load_ef_model() returned None")

        _model = model
        _model_initialized = True

        print(f"✅ Model loaded successfully on device={settings.DEVICE}")

    except Exception as e:
        _model = None
        _model_initialized = False
        print(f"❌ Model initialization failed: {e}")
        raise RuntimeError(f"Model initialization failed: {e}")


def get_model() -> torch.nn.Module:
    """Return the loaded model or raise if unavailable."""
    if not is_model_loaded():
        raise RuntimeError("Model not initialized.")
    return _model


# ============================================================
# Preprocessing Wrapper
# ============================================================

def preprocess_video(file_bytes: bytes, filename: str) -> torch.Tensor:
    """
    Wrapper around preprocess_video_bytes().
    Produces shape: (1, 1, T=32, 112, 112)
    """
    return preprocess_video_bytes(file_bytes, filename)


# ============================================================
# Inference Logic
# ============================================================

def predict_ef(video_tensor: torch.Tensor) -> Dict[str, Any]:
    """
    Perform EF prediction on a preprocessed video tensor.
    Returns EF in schema format:
        - ef_percent
        - category
        - confidence
    """
    model = get_model()

    expected_shape = (
        1,
        1,
        settings.DEFAULT_MAX_FRAMES,
        settings.DEFAULT_TARGET_SIZE[0],
        settings.DEFAULT_TARGET_SIZE[1],
    )

    if tuple(video_tensor.shape) != expected_shape:
        raise ValueError(
            f"Invalid tensor shape. Expected {expected_shape}, got {tuple(video_tensor.shape)}"
        )

    try:
        with torch.no_grad():
            output = model(video_tensor)

        # Normalize output
        if output.dim() == 0:
            ef_val = output.item()
        elif output.dim() == 1:
            ef_val = output[0].item()
        else:
            ef_val = output[0][0].item()

        ef_val = float(max(0.0, min(100.0, ef_val)))
        ef_val_rounded = round(ef_val, 2)

        return {
            "ef_percent": ef_val_rounded,
            "category": settings.get_ef_severity(ef_val),
            "confidence": None,
        }

    except Exception as e:
        print("❌ Inference failure:", e)
        raise RuntimeError(f"Prediction failed: {e}")


# ============================================================
# FastAPI Handler
# ============================================================

async def run_inference(file: UploadFile) -> Dict[str, Any]:
    """
    Preprocess → predict → return schema-formatted output
    """
    if not is_model_loaded():
        raise HTTPException(
            status_code=503,
            detail="Model is still loading."
        )

    if not file.content_type.startswith("video/"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{file.content_type}'. Expected video/*."
        )

    try:
        file_bytes = await file.read()
        size_mb = len(file_bytes) / (1024 * 1024)

        if size_mb > settings.MAX_VIDEO_SIZE_MB:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"File too large: {size_mb:.2f}MB "
                    f"(max {settings.MAX_VIDEO_SIZE_MB}MB)"
                )
            )

        video_tensor = preprocess_video(file_bytes, file.filename)
        base = predict_ef(video_tensor)

        return {
            **base,
            "filename": file.filename,
            "file_size_mb": round(size_mb, 2),
            "model_name": "echonet_3dcnn",
            "model_version": "1.0.0",
        }

    except HTTPException:
        raise

    except Exception as e:
        print("❌ Pipeline error:", e)
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {e}"
        )

    finally:
        await file.close()

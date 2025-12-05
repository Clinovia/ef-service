from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.inference import initialize_model, run_inference, is_model_loaded
from app.config import settings


# -----------------------------
# FastAPI Application Setup
# -----------------------------
app = FastAPI(
    title=settings.SERVICE_NAME,
    description="Ejection fraction prediction from echocardiogram videos",
    version="1.0.0"
)

# CORS settings — configure origins in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# Lifecycle Events
# -----------------------------
@app.on_event("startup")
async def startup_event():
    """Load EF model at startup."""
    print(f"🚀 Starting {settings.SERVICE_NAME}")
    try:
        initialize_model()
        print("✅ EF model loaded — service ready")
    except Exception as e:
        print(f"❌ Model load failure: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    print(f"👋 Shutting down {settings.SERVICE_NAME}")


# -----------------------------
# API Endpoints
# -----------------------------
@app.get("/")
def root():
    """Root endpoint with service information."""
    return {
        "service": settings.SERVICE_NAME,
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "predict": "/predict-ef",
            "docs": "/docs"
        }
    }


@app.get("/health")
def health_check():
    """
    Health check endpoint for Railway / AWS / Kubernetes.
    Returns service status and model readiness.
    """
    model_loaded = is_model_loaded()
    
    return {
        "status": "healthy" if model_loaded else "starting",
        "model_loaded": model_loaded,
        "service": settings.SERVICE_NAME,
        "device": str(settings.DEVICE)
    }


@app.post("/predict-ef")
async def predict_ef_endpoint(video: UploadFile = File(...)):
    """
    Upload an echocardiogram video and return EF prediction.
    
    Args:
        video: Video file (.avi, .mp4, .mov)
        
    Returns:
        dict: Prediction results with EF value, severity, and metadata
        
    Raises:
        HTTPException: If validation fails or prediction errors occur
    """
    # Check if model is ready
    if not is_model_loaded():
        raise HTTPException(
            status_code=503,
            detail="Service is starting. Model not ready yet."
        )
    
    # Validate file size
    if video.size and video.size > settings.MAX_VIDEO_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"Video exceeds {settings.MAX_VIDEO_SIZE_MB}MB limit. "
                   f"Uploaded file is {video.size / (1024 * 1024):.2f}MB"
        )
    
    # Validate mime type
    allowed_types = [
        "video/x-msvideo",  # .avi
        "video/avi",
        "video/mp4",
        "video/quicktime",  # .mov
        "video/x-matroska"  # .mkv (optional)
    ]
    
    if video.content_type and video.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {video.content_type}. "
                   f"Allowed formats: .avi, .mp4, .mov"
        )
    
    try:
        # Run inference pipeline
        result = await run_inference(video)
        return result
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
        
    except ValueError as e:
        # Validation errors
        raise HTTPException(status_code=400, detail=str(e))
        
    except Exception as e:
        # Unexpected errors
        print(f"❌ Prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


# -----------------------------
# Additional Utility Endpoints
# -----------------------------
@app.get("/model-info")
def model_info():
    """
    Get information about the loaded model.
    """
    if not is_model_loaded():
        raise HTTPException(
            status_code=503,
            detail="Model not loaded yet"
        )
    
    return {
        "model_key": settings.EF_MODEL_KEY,
        "model_path": settings.LOCAL_MODEL_PATH,
        "device": str(settings.DEVICE),
        "input_size": {
            "frames": settings.DEFAULT_MAX_FRAMES,
            "height": settings.DEFAULT_TARGET_SIZE[0],
            "width": settings.DEFAULT_TARGET_SIZE[1]
        },
        "thresholds": settings.EF_THRESHOLDS
    }


@app.get("/version")
def version():
    """
    Get service version information.
    """
    return {
        "service": settings.SERVICE_NAME,
        "version": "1.0.0",
        "model_version": settings.EF_MODEL_KEY.split('/')[-1] if settings.EF_MODEL_KEY else "unknown"
    }
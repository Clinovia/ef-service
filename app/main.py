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
    
    # Check FFmpeg availability
    from app.preprocessing import check_ffmpeg_installed
    if not check_ffmpeg_installed():
        print("⚠️  Warning: FFmpeg not found. Video format conversion will fail.")
        print("   Install FFmpeg: https://ffmpeg.org/download.html")
    else:
        print("✅ FFmpeg is available")
    
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
    Schema fields returned:
        - ef_percent
        - category
        - confidence
        - model_name
        - model_version
    """
    if not is_model_loaded():
        raise HTTPException(
            status_code=503,
            detail="Service is starting. Model not ready yet."
        )

    allowed_types = [
        "video/x-msvideo",
        "video/avi",
        "video/mp4",
        "video/quicktime",
        "video/x-matroska"
    ]

    if video.content_type and video.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {video.content_type}. "
                   f"Allowed formats: .avi, .mp4, .mov"
        )

    try:
        result = await run_inference(video)
        return result

    except HTTPException:
        raise

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

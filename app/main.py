import os
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse  # <-- FIXED missing import

from app.inference import initialize_model, run_inference, is_model_loaded
from app.config import settings

# Load environment variables
load_dotenv()

# -----------------------------
# FastAPI App
# -----------------------------
app = FastAPI(
    title=settings.SERVICE_NAME,
    description="Ejection fraction prediction from echocardiogram videos",
    version="1.0.0",
)

# -----------------------------
# CORS
# -----------------------------
origins = [os.getenv("BACKEND_URL")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins if origins else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Token middleware (optional)
# -----------------------------
@app.middleware("http")
async def verify_token(request: Request, call_next):
    """Optional auth placeholder: skip auth for /health/docs."""
    if request.url.path in ["/health", "/docs", "/openapi.json"]:
        return await call_next(request)

    auth = request.headers.get("authorization")
    if not auth:
        return JSONResponse(status_code=401, content={"detail": "Unauthorized"})

    return await call_next(request)

# -----------------------------
# Startup & Shutdown Events
# -----------------------------
@app.on_event("startup")
async def startup_event():
    print(f"🚀 Starting {settings.SERVICE_NAME}")

    from app.preprocessing import check_ffmpeg_installed

    if not check_ffmpeg_installed():
        print("⚠️  FFmpeg not found. Video conversion may fail.")
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
    print(f"👋 Shutting down {settings.SERVICE_NAME}")

# -----------------------------
# Endpoints
# -----------------------------
@app.get("/")
def root():
    return {
        "service": settings.SERVICE_NAME,
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "predict": "/predict-ef",
            "docs": "/docs",
        },
    }

@app.get("/health")
def health_check():
    model_ready = is_model_loaded()
    return {
        "status": "healthy" if model_ready else "starting",
        "model_loaded": model_ready,
        "service": settings.SERVICE_NAME,
        "device": str(settings.DEVICE),
    }

@app.post("/ejection-fraction", response_model=dict)
async def predict_ef_endpoint(video: UploadFile = File(...)):
    """Predict EF from uploaded video."""
    if not is_model_loaded():
        raise HTTPException(status_code=503, detail="Model not ready yet.")

    allowed_types = [
        "video/x-msvideo",
        "video/avi",
        "video/mp4",
        "video/quicktime",
        "video/x-matroska",
    ]
    if video.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {video.content_type}. Allowed: .avi, .mp4, .mov",
        )

    try:
        return await run_inference(video)

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.inference import load_model, predict_ef, is_model_loaded
from app.config import SERVICE_NAME, MAX_VIDEO_SIZE_MB

app = FastAPI(
    title=SERVICE_NAME,
    description="Ejection fraction prediction from echocardiogram videos",
    version="1.0.0"
)

# CORS middleware - allows your main backend to call this service
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Load model when service starts"""
    print(f"🚀 Starting {SERVICE_NAME}")
    try:
        load_model()
        print("✅ Service ready")
    except Exception as e:
        print(f"❌ Failed to start service: {e}")
        raise


@app.get("/")
def root():
    """Root endpoint"""
    return {
        "service": SERVICE_NAME,
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
def health_check():
    """Health check endpoint for Railway"""
    return {
        "status": "healthy" if is_model_loaded() else "starting",
        "model_loaded": is_model_loaded()
    }


@app.post("/predict-ef")
async def predict_ef_endpoint(video: UploadFile = File(...)):
    """
    Predict ejection fraction from echocardiogram video
    
    Args:
        video: Video file upload (.avi, .mp4)
    
    Returns:
        Dictionary with EF prediction and severity classification
    """
    
    # Validate file size
    if video.size and video.size > MAX_VIDEO_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail=f"Video size exceeds {MAX_VIDEO_SIZE_MB}MB limit"
        )
    
    # Validate file type
    allowed_types = ["video/x-msvideo", "video/avi", "video/mp4", "video/quicktime"]
    if video.content_type and video.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {video.content_type}. Allowed: .avi, .mp4"
        )
    
    try:
        result = await predict_ef(video)
        return result
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )
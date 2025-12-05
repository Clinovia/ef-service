import os
import torch
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    
    # Find .env file in project root
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
        print(f"✅ Loaded environment variables from {env_path}")
    else:
        print("ℹ️  No .env file found, using system environment variables")
except ImportError:
    print("ℹ️  python-dotenv not installed, using system environment variables")


class Settings:
    # -----------------------------
    # FastAPI / Service
    # -----------------------------
    SERVICE_NAME = "Clinovia EF Microservice"
    PORT = int(os.environ.get("PORT", 8081))
    
    # -----------------------------
    # AWS S3 / Model Storage
    # -----------------------------
    AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
    AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
    
    S3_BUCKET = os.environ.get("CLINOVIA_S3_BUCKET", "clinovia.ai")
    EF_MODEL_KEY = os.environ.get(
        "EF_MODEL_KEY",
        "models/cardiology/ejection-fraction/v1/ef3dcnn_epoch17.pth"
    )
    LOCAL_MODEL_PATH = os.environ.get("LOCAL_MODEL_PATH", "/tmp/ef_model.pth")
    
    # -----------------------------
    # PyTorch Device
    # -----------------------------
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # -----------------------------
    # Video Processing
    # -----------------------------
    DEFAULT_TARGET_SIZE = (112, 112)
    DEFAULT_MAX_FRAMES = 32
    MAX_VIDEO_SIZE_MB = int(os.environ.get("MAX_VIDEO_SIZE_MB", "100"))
    
    # -----------------------------
    # EF Severity Thresholds
    # -----------------------------
    EF_THRESHOLDS = {
        "normal": 50,
        "mild": 40,
        "moderate": 30,
    }
    
    @staticmethod
    def get_ef_severity(ef_value: float) -> str:
        """Classify EF severity based on clinically accepted thresholds."""
        if ef_value >= Settings.EF_THRESHOLDS["normal"]:
            return "normal"
        elif ef_value >= Settings.EF_THRESHOLDS["mild"]:
            return "mild_dysfunction"
        elif ef_value >= Settings.EF_THRESHOLDS["moderate"]:
            return "moderate_dysfunction"
        else:
            return "severe_dysfunction"
    
    @classmethod
    def validate(cls):
        """Validate critical configuration values."""
        errors = []
        
        if not cls.S3_BUCKET:
            errors.append("CLINOVIA_S3_BUCKET environment variable is required")
        
        if not cls.AWS_ACCESS_KEY_ID:
            errors.append("AWS_ACCESS_KEY_ID environment variable is required")
        
        if not cls.AWS_SECRET_ACCESS_KEY:
            errors.append("AWS_SECRET_ACCESS_KEY environment variable is required")
        
        if errors:
            error_msg = "\n".join([f"  ❌ {error}" for error in errors])
            raise ValueError(f"Configuration errors:\n{error_msg}")
        
        return True
    
    @classmethod
    def print_config(cls):
        """Print current configuration (masking sensitive values)."""
        print("\n" + "="*50)
        print("Configuration Summary")
        print("="*50)
        print(f"Service Name: {cls.SERVICE_NAME}")
        print(f"Port: {cls.PORT}")
        print(f"Device: {cls.DEVICE}")
        print(f"S3 Bucket: {cls.S3_BUCKET}")
        print(f"Model Key: {cls.EF_MODEL_KEY}")
        print(f"Local Model Path: {cls.LOCAL_MODEL_PATH}")
        
        # Mask credentials
        if cls.AWS_ACCESS_KEY_ID:
            masked_key = cls.AWS_ACCESS_KEY_ID[:4] + "*" * 12 + cls.AWS_ACCESS_KEY_ID[-4:]
            print(f"AWS Access Key: {masked_key}")
        else:
            print("AWS Access Key: ❌ NOT SET")
        
        if cls.AWS_SECRET_ACCESS_KEY:
            print(f"AWS Secret Key: {'*' * 20} (hidden)")
        else:
            print("AWS Secret Key: ❌ NOT SET")
        
        print(f"AWS Region: {cls.AWS_REGION}")
        print("="*50 + "\n")


# Create singleton instance
settings = Settings()

# Print configuration on import
settings.print_config()

# Validate configuration (comment out if you want to start without AWS credentials)
try:
    settings.validate()
    print("✅ Configuration validated successfully\n")
except ValueError as e:
    print(f"\n⚠️  Configuration Warning:\n{e}\n")
    print("💡 To fix this:")
    print("   1. Create a .env file in your project root")
    print("   2. Add your AWS credentials:")
    print("      AWS_ACCESS_KEY_ID=your-key-id")
    print("      AWS_SECRET_ACCESS_KEY=your-secret-key")
    print("   3. Or set them as environment variables\n")
    # Don't raise - allow server to start for health checks
    # raise

print(f"🖥️  Using compute device: {settings.DEVICE}\n")
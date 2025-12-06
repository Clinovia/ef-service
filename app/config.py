import os
import torch
from pathlib import Path

# -----------------------------
# Load environment variables
# -----------------------------
try:
    from dotenv import load_dotenv

    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
        print(f"✅ Loaded environment variables from {env_path}")
    else:
        print("ℹ️  No .env file found; using system environment variables")

except ImportError:
    print("ℹ️  python-dotenv not installed; using system environment variables")


class Settings:
    # -----------------------------
    # FastAPI Service Config
    # -----------------------------
    SERVICE_NAME = "Clinovia EF Microservice"
    PORT = int(os.environ.get("PORT", 8081))

    # -----------------------------
    # AWS S3 (Model Storage)
    # -----------------------------
    AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
    AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

    S3_BUCKET = os.environ.get("CLINOVIA_S3_BUCKET", "clinovia.ai")

    EF_MODEL_KEY = os.environ.get(
        "EF_MODEL_KEY",
        "models/cardiology/ejection-fraction/v1/ef3dcnn_epoch17.pth"
    )

    LOCAL_MODEL_PATH = os.environ.get(
        "LOCAL_MODEL_PATH",
        "/tmp/ef_model.pth"
    )

    # -----------------------------
    # PyTorch Device
    # -----------------------------
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -----------------------------
    # Video Preprocessing Settings
    # -----------------------------
    # ✔ MUST match your EchoNet training input
    DEFAULT_TARGET_SIZE = (112, 112)   # (H, W)
    DEFAULT_MAX_FRAMES = 32            # number of frames your model expects
    MAX_VIDEO_SIZE_MB = int(os.environ.get("MAX_VIDEO_SIZE_MB", "100"))

    # -----------------------------
    # EF Severity Thresholds
    # -----------------------------
    EF_THRESHOLDS = {
        "normal": 50,     # >= 50%
        "mild": 40,       # 40–49%
        "moderate": 30    # 30–39%
        # <30 → severe
    }

    @staticmethod
    def get_ef_severity(ef_value: float) -> str:
        """
        Assign EF severity class.
        """
        if ef_value >= Settings.EF_THRESHOLDS["normal"]:
            return "normal"
        elif ef_value >= Settings.EF_THRESHOLDS["mild"]:
            return "mild_dysfunction"
        elif ef_value >= Settings.EF_THRESHOLDS["moderate"]:
            return "moderate_dysfunction"
        else:
            return "severe_dysfunction"

    # -----------------------------
    # Validation Helpers
    # -----------------------------
    @classmethod
    def validate(cls):
        """Validate that required environment variables exist."""
        errors = []

        if not cls.S3_BUCKET:
            errors.append("CLINOVIA_S3_BUCKET environment variable is required")

        if not cls.AWS_ACCESS_KEY_ID:
            errors.append("AWS_ACCESS_KEY_ID environment variable is required")

        if not cls.AWS_SECRET_ACCESS_KEY:
            errors.append("AWS_SECRET_ACCESS_KEY environment variable is required")

        if errors:
            error_msg = "\n".join([f"  ❌ {e}" for e in errors])
            raise ValueError(f"Configuration errors:\n{error_msg}")

        return True

    @classmethod
    def print_config(cls):
        """Print a summary of current configuration with masked secrets."""
        print("\n" + "=" * 50)
        print("Configuration Summary")
        print("=" * 50)
        print(f"Service Name: {cls.SERVICE_NAME}")
        print(f"Port: {cls.PORT}")
        print(f"Device: {cls.DEVICE}")
        print(f"S3 Bucket: {cls.S3_BUCKET}")
        print(f"Model Key: {cls.EF_MODEL_KEY}")
        print(f"Local Model Path: {cls.LOCAL_MODEL_PATH}")

        # Mask secret values
        if cls.AWS_ACCESS_KEY_ID:
            masked = cls.AWS_ACCESS_KEY_ID[:4] + "*" * 12 + cls.AWS_ACCESS_KEY_ID[-4:]
            print(f"AWS Access Key: {masked}")
        else:
            print("AWS Access Key: ❌ NOT SET")

        if cls.AWS_SECRET_ACCESS_KEY:
            print("AWS Secret Key: " + "*" * 20)
        else:
            print("AWS Secret Key: ❌ NOT SET")

        print(f"AWS Region: {cls.AWS_REGION}")
        print("=" * 50 + "\n")


# -----------------------------
# Create settings singleton
# -----------------------------
settings = Settings()

# Print config & validate
settings.print_config()

try:
    settings.validate()
    print("✅ Configuration validated successfully\n")
except ValueError as e:
    print(f"\n⚠️  Configuration Warning:\n{e}\n")
    print("💡 To fix:")
    print("   Create .env and set your AWS credentials.")
    print("   Server will still start for health checks.\n")

print(f"🖥️  Using compute device: {settings.DEVICE}\n")

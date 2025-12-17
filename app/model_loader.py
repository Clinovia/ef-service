# app/model_loader.py
import os
import torch
from .config import settings
from .models.ef_model import EF3DCNN

def _create_model(device: str):
    H, W = settings.DEFAULT_TARGET_SIZE
    return EF3DCNN(
        in_channels=1,
        T=settings.DEFAULT_MAX_FRAMES,
        H=H,
        W=W
    ).to(device)

def _load_state_dict(model, state_dict):
    if isinstance(state_dict, dict):
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        elif "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
    model.load_state_dict(state_dict)

def load_ef_model(local_path: str = None, device: str = None) -> torch.nn.Module:
    """
    Load EF3DCNN model. Defaults to EF_LOCAL_MODEL_PATH from .env.
    """
    device = device or str(settings.DEVICE)
    local_path = local_path or settings.LOCAL_MODEL_PATH

    if not os.path.exists(local_path):
        raise FileNotFoundError(f"Local model not found: {local_path}")

    print(f"📥 Loading EF model from local path: {local_path}")
    model = _create_model(device)
    state_dict = torch.load(local_path, map_location=device, weights_only=True)
    _load_state_dict(model, state_dict)
    model.eval()
    print(f"✅ EF model loaded successfully on {device}")
    return model

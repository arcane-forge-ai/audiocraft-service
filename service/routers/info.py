from fastapi import APIRouter
# from service.config import settings
import torch

router = APIRouter()


@router.get("/ping")
async def ping():
    return {"ping": "pong"}

@router.get("/info")
async def info():
    return {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
    }


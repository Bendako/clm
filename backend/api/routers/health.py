from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any
import psutil
import os
import time

router = APIRouter()

@router.get("/")
async def health_check() -> Dict[str, Any]:
    """
    Basic health check endpoint to verify API is running.
    Returns system metrics and uptime information.
    """
    return {
        "status": "ok",
        "timestamp": time.time(),
        "uptime": time.time() - psutil.boot_time(),
        "cpu_usage": psutil.cpu_percent(interval=0.1),
        "memory_usage": psutil.virtual_memory().percent,
        "disk_usage": psutil.disk_usage('/').percent,
        "api_version": "0.1.0"
    }

@router.get("/readiness")
async def readiness_check() -> Dict[str, str]:
    """
    Readiness probe for Kubernetes.
    Checks if all required services are available.
    """
    # TODO: Implement actual database and service checks
    return {"status": "ready"}

@router.get("/liveness")
async def liveness_check() -> Dict[str, str]:
    """
    Liveness probe for Kubernetes.
    Verifies the API is responsive.
    """
    return {"status": "alive"} 
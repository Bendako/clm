from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import Dict, Any
import psutil
import os
import time

from db.database import get_db
from db.utils import verify_db_connection, get_table_names

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
async def readiness_check(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """
    Readiness probe for Kubernetes.
    Checks if all required services are available.
    """
    services_status = {
        "api": "ready",
        "database": "ready" if verify_db_connection() else "not_ready"
    }
    
    # Check if database has tables
    try:
        tables = get_table_names()
        services_status["database_tables"] = len(tables)
    except Exception as e:
        services_status["database"] = "error"
        services_status["database_error"] = str(e)
    
    # Determine overall status
    overall_status = "ready" if all(s == "ready" for s in 
                                    [v for k, v in services_status.items() 
                                     if isinstance(v, str)]) else "not_ready"
    
    return {
        "status": overall_status,
        "services": services_status,
        "timestamp": time.time()
    }

@router.get("/liveness")
async def liveness_check() -> Dict[str, str]:
    """
    Liveness probe for Kubernetes.
    Verifies the API is responsive.
    """
    return {"status": "alive"} 
from fastapi import APIRouter, Depends, HTTPException, Body, UploadFile, File, Form, status
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
import uuid

# Models for request/response
class ModelBase(BaseModel):
    name: str
    description: Optional[str] = None
    model_type: str
    framework: str = "pytorch"
    tags: Optional[List[str]] = None

class ModelCreate(ModelBase):
    pass

class Model(ModelBase):
    id: str
    created_at: datetime
    updated_at: datetime
    version: str
    status: str = "registered"
    metrics: Optional[Dict[str, float]] = None
    
    class Config:
        orm_mode = True

class ModelVersion(BaseModel):
    model_id: str
    version: str
    created_at: datetime
    status: str
    metrics: Optional[Dict[str, float]] = None
    
    class Config:
        orm_mode = True

# Initialize router
router = APIRouter()

# Mock database for development
mock_models = {}

@router.post("/", response_model=Model, status_code=status.HTTP_201_CREATED)
async def create_model(model: ModelCreate) -> Model:
    """
    Register a new model in the registry.
    """
    model_id = str(uuid.uuid4())
    now = datetime.now()
    
    new_model = Model(
        id=model_id,
        created_at=now,
        updated_at=now,
        version="0.1.0",
        **model.dict()
    )
    
    mock_models[model_id] = new_model
    return new_model

@router.get("/", response_model=List[Model])
async def list_models(skip: int = 0, limit: int = 100) -> List[Model]:
    """
    List all registered models.
    """
    return list(mock_models.values())[skip:skip+limit]

@router.get("/{model_id}", response_model=Model)
async def get_model(model_id: str) -> Model:
    """
    Get details of a specific model.
    """
    if model_id not in mock_models:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model with ID {model_id} not found"
        )
    return mock_models[model_id]

@router.post("/{model_id}/versions", response_model=ModelVersion)
async def create_model_version(
    model_id: str,
    version: str = Form(...),
    metrics: str = Form("{}"),
    model_file: UploadFile = File(...)
) -> ModelVersion:
    """
    Register a new version of an existing model.
    """
    if model_id not in mock_models:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model with ID {model_id} not found"
        )
    
    # In a real implementation, we would save the model file
    # and register the version in MLflow or similar system
    
    return ModelVersion(
        model_id=model_id,
        version=version,
        created_at=datetime.now(),
        status="registered"
    )

@router.get("/{model_id}/versions", response_model=List[ModelVersion])
async def list_model_versions(model_id: str) -> List[ModelVersion]:
    """
    List all versions of a specific model.
    """
    if model_id not in mock_models:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model with ID {model_id} not found"
        )
    
    # Mock response for development
    return [
        ModelVersion(
            model_id=model_id,
            version="0.1.0",
            created_at=mock_models[model_id].created_at,
            status="registered"
        )
    ] 
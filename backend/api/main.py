from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
import os

# Import routers
from api.routers import models, data, training, deployments, health

# Initialize FastAPI app
app = FastAPI(
    title="CLM - Continuous Learning for LLMs",
    description="API for managing continuous learning of LLMs",
    version="0.1.0"
)

# Setup CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this in production to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, prefix="/api/health", tags=["Health"])
app.include_router(models.router, prefix="/api/models", tags=["Models"])
app.include_router(data.router, prefix="/api/data", tags=["Data"])
app.include_router(training.router, prefix="/api/training", tags=["Training"])
app.include_router(deployments.router, prefix="/api/deployments", tags=["Deployments"])

@app.on_event("startup")
async def startup_event():
    logger.info("Starting the CLM Backend API")
    # Initialize services and connections
    # TODO: Add database initialization
    # TODO: Add model registry connection
    # TODO: Add authentication setup

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down CLM Backend API")
    # Clean up resources
    # TODO: Close database connections
    # TODO: Close other resources

@app.get("/")
async def root():
    """Root endpoint providing API information"""
    return {
        "name": "CLM - Continuous Learning for LLMs",
        "version": "0.1.0",
        "docs": "/docs",
        "status": "operational"
    } 
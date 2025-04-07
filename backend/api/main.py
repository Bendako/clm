from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
import os

# Import routers
from api.routers import models, data, training, deployments, health

# Import database
from db.utils import init_db, verify_db_connection
from db.database import get_db, engine, Base

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
    
    # Initialize database
    logger.info("Initializing database connection")
    try:
        # Verify database connection
        if not verify_db_connection():
            logger.error("Failed to connect to database")
            raise Exception("Database connection failed")
        
        # Initialize database tables (only in development)
        if os.getenv("ENVIRONMENT", "development") == "development":
            logger.info("Initializing database tables")
            init_db()
        
        logger.success("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization error: {str(e)}")
        # In production, we might want to exit here instead of continuing
        # sys.exit(1)

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down CLM Backend API")
    # Close any connections or resources
    logger.info("Closing database connections")
    # SQLAlchemy engine manages connection pool cleanup automatically
    # If we have other resources to close, add them here

@app.get("/")
async def root():
    """Root endpoint providing API information"""
    return {
        "name": "CLM - Continuous Learning for LLMs",
        "version": "0.1.0",
        "docs": "/docs",
        "status": "operational"
    } 
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv
from loguru import logger

# Load environment variables
load_dotenv()

# Get database URL from environment variable or use default
SQLALCHEMY_DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "postgresql://postgres:postgres@localhost:5432/clm"
)

# Create database engine with connection pooling
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    pool_pre_ping=True,  # Test connections before using them
    pool_size=5,  # Number of connections to keep permanently
    max_overflow=10  # Maximum number of connections to use in addition to pool_size
)

# Create a session factory that will generate new sessions
SessionLocal = sessionmaker(
    autocommit=False, 
    autoflush=False, 
    bind=engine
)

# Create a Base class for declarative models
Base = declarative_base()

# Dependency to get a database session
def get_db():
    """
    Creates a database session for a request and closes it when the request is done.
    This function should be used as a FastAPI dependency.
    """
    db = SessionLocal()
    try:
        yield db
        logger.debug("Database session provided")
    finally:
        db.close()
        logger.debug("Database session closed") 
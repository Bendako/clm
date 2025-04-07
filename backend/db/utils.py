from sqlalchemy import inspect
from sqlalchemy.orm import Session
import os
from loguru import logger

from .database import engine, Base
from .models import *  # Import all models to register them with the Base class


def init_db():
    """
    Initialize the database with all defined models.
    This creates all tables that don't exist yet.
    """
    logger.info("Initializing database")
    Base.metadata.create_all(bind=engine)
    logger.info("Database initialized successfully")


def drop_db():
    """
    Drop all database tables.
    WARNING: This will delete all data in the database.
    """
    logger.warning("Dropping all database tables")
    Base.metadata.drop_all(bind=engine)
    logger.warning("All database tables dropped")


def get_table_names():
    """
    Get a list of all table names in the database.
    """
    inspector = inspect(engine)
    return inspector.get_table_names()


def verify_db_connection():
    """
    Verify that the database connection works.
    Returns True if connection is successful, False otherwise.
    """
    try:
        # Try to get table names to verify connection
        inspector = inspect(engine)
        inspector.get_table_names()
        return True
    except Exception as e:
        logger.error(f"Database connection failed: {str(e)}")
        return False


def count_records(db: Session, model_class):
    """
    Count the number of records in a table.
    
    Args:
        db: SQLAlchemy session
        model_class: The model class to count records for
        
    Returns:
        int: Number of records in the table
    """
    return db.query(model_class).count()


def clear_table(db: Session, model_class):
    """
    Delete all records from a table.
    
    Args:
        db: SQLAlchemy session
        model_class: The model class to clear
        
    Returns:
        int: Number of records deleted
    """
    result = db.query(model_class).delete()
    db.commit()
    return result 
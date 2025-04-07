#!/usr/bin/env python
"""
Database seeding script for CLM backend.
This script initializes the database and populates it with sample data for development.
"""

import os
import sys
import json
from datetime import datetime, timedelta
import uuid
from loguru import logger
from sqlalchemy.orm import Session

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.database import SessionLocal
from db.utils import init_db, drop_db, verify_db_connection
from db.models import (
    RegisteredModel, ModelVersion, ModelTask,
    Dataset, DatasetVersion, DatasetTask, DataDrift,
    TrainingJob, ContinualLearningRun, Deployment, DeploymentLog
)


def seed_models(db: Session):
    """Seed the database with sample models."""
    logger.info("Seeding model registry tables")
    
    # Create a sample registered model
    model = RegisteredModel(
        name="MNIST Classifier",
        description="A simple classifier trained on MNIST dataset",
        model_type="classifier",
        framework="pytorch",
        tags=["computer-vision", "classification", "demo"]
    )
    db.add(model)
    db.flush()  # Flush to get the ID
    
    # Create a few versions of the model
    version1 = ModelVersion(
        model_id=model.id,
        version="0.1.0",
        artifact_path="/models/mnist/0.1.0",
        metrics={"accuracy": 0.92, "f1": 0.91},
        params={"learning_rate": 0.001, "epochs": 10},
        status="archived"
    )
    
    version2 = ModelVersion(
        model_id=model.id,
        version="0.2.0",
        artifact_path="/models/mnist/0.2.0",
        metrics={"accuracy": 0.95, "f1": 0.94},
        params={"learning_rate": 0.0005, "epochs": 15},
        status="deployed"
    )
    
    db.add_all([version1, version2])
    db.flush()
    
    # Add model tasks
    task1 = ModelTask(
        model_version_id=version1.id,
        task_name="digit_recognition_base",
        task_type="classification",
        performance={"accuracy": 0.92, "f1": 0.91}
    )
    
    task2 = ModelTask(
        model_version_id=version2.id,
        task_name="digit_recognition_base",
        task_type="classification",
        performance={"accuracy": 0.95, "f1": 0.94}
    )
    
    task3 = ModelTask(
        model_version_id=version2.id,
        task_name="digit_recognition_rotated",
        task_type="classification",
        performance={"accuracy": 0.87, "f1": 0.86}
    )
    
    db.add_all([task1, task2, task3])
    
    # Create a second model
    model2 = RegisteredModel(
        name="Sentiment Analyzer",
        description="A sentiment analysis model for text",
        model_type="nlp",
        framework="pytorch",
        tags=["nlp", "sentiment", "text"]
    )
    db.add(model2)
    db.flush()
    
    version3 = ModelVersion(
        model_id=model2.id,
        version="0.1.0",
        artifact_path="/models/sentiment/0.1.0",
        metrics={"accuracy": 0.88, "f1": 0.87},
        params={"learning_rate": 0.001, "epochs": 5},
        status="registered"
    )
    
    db.add(version3)
    db.flush()
    
    task4 = ModelTask(
        model_version_id=version3.id,
        task_name="sentiment_analysis",
        task_type="classification",
        performance={"accuracy": 0.88, "f1": 0.87}
    )
    
    db.add(task4)
    db.commit()
    
    logger.info(f"Created {db.query(RegisteredModel).count()} models with {db.query(ModelVersion).count()} versions")


def seed_datasets(db: Session):
    """Seed the database with sample datasets."""
    logger.info("Seeding dataset tables")
    
    # Create sample datasets
    mnist = Dataset(
        name="MNIST",
        description="Handwritten digit recognition dataset",
        data_type="image",
        format="numpy",
        location="s3://clm-datasets/mnist",
        size_bytes=11594722,
        num_samples=70000,
        stats={"mean": 0.1307, "std": 0.3081}
    )
    
    sentiment = Dataset(
        name="IMDb Reviews",
        description="Movie reviews for sentiment analysis",
        data_type="text",
        format="json",
        location="s3://clm-datasets/imdb",
        size_bytes=128457398,
        num_samples=50000
    )
    
    db.add_all([mnist, sentiment])
    db.flush()
    
    # Create dataset versions
    mnist_v1 = DatasetVersion(
        dataset_id=mnist.id,
        version="1.0.0",
        change_description="Initial version"
    )
    
    mnist_v2 = DatasetVersion(
        dataset_id=mnist.id,
        version="1.1.0",
        change_description="Added rotated digits"
    )
    
    sentiment_v1 = DatasetVersion(
        dataset_id=sentiment.id,
        version="1.0.0",
        change_description="Initial version"
    )
    
    db.add_all([mnist_v1, mnist_v2, sentiment_v1])
    db.flush()
    
    # Create dataset tasks
    dt1 = DatasetTask(
        dataset_version_id=mnist_v1.id,
        task_name="digit_recognition_base",
        split="train"
    )
    
    dt2 = DatasetTask(
        dataset_version_id=mnist_v2.id,
        task_name="digit_recognition_rotated",
        split="train"
    )
    
    dt3 = DatasetTask(
        dataset_version_id=sentiment_v1.id,
        task_name="sentiment_analysis",
        split="train"
    )
    
    db.add_all([dt1, dt2, dt3])
    
    # Create data drift record
    drift = DataDrift(
        dataset_id=mnist.id,
        reference_version_id=mnist_v1.id,
        current_version_id=mnist_v2.id,
        drift_score=0.32,
        drift_type="feature",
        detection_method="statistical",
        feature_scores={"pixel_mean_shift": 0.15, "rotation_angle": 0.78}
    )
    
    db.add(drift)
    db.commit()
    
    logger.info(f"Created {db.query(Dataset).count()} datasets with {db.query(DatasetVersion).count()} versions")


def seed_training_and_deployments(db: Session):
    """Seed the database with sample training jobs and deployments."""
    logger.info("Seeding training and deployment tables")
    
    # Get models and datasets for reference
    models = db.query(RegisteredModel).all()
    model_versions = db.query(ModelVersion).all()
    
    # Create training jobs
    now = datetime.utcnow()
    
    job1 = TrainingJob(
        name="Initial MNIST Training",
        description="Initial training of MNIST classifier",
        model_id=models[0].id,
        status="completed",
        start_time=now - timedelta(days=7),
        end_time=now - timedelta(days=7, hours=1),
        hyperparameters={"learning_rate": 0.001, "batch_size": 64, "epochs": 10},
        metrics={"accuracy": 0.92, "f1": 0.91},
        logs="Training completed successfully...",
        artifacts_path="/artifacts/jobs/1",
        strategy="EWC",
        strategy_params={"lambda": 1.0},
        task_id="digit_recognition_base",
        previous_tasks=[]
    )
    
    job2 = TrainingJob(
        name="MNIST Continual Learning - Rotated",
        description="Continual learning on rotated digits",
        model_id=models[0].id,
        base_version_id=model_versions[0].id,
        status="completed",
        start_time=now - timedelta(days=3),
        end_time=now - timedelta(days=3, hours=2),
        hyperparameters={"learning_rate": 0.0005, "batch_size": 64, "epochs": 15},
        metrics={"accuracy": 0.95, "f1": 0.94},
        logs="Training completed successfully...",
        artifacts_path="/artifacts/jobs/2",
        strategy="EWC",
        strategy_params={"lambda": 1.0},
        task_id="digit_recognition_rotated",
        previous_tasks=["digit_recognition_base"]
    )
    
    job3 = TrainingJob(
        name="Sentiment Analysis Training",
        description="Initial training of sentiment analyzer",
        model_id=models[1].id,
        status="completed",
        start_time=now - timedelta(days=1),
        end_time=now - timedelta(days=1, hours=3),
        hyperparameters={"learning_rate": 0.001, "batch_size": 32, "epochs": 5},
        metrics={"accuracy": 0.88, "f1": 0.87},
        logs="Training completed successfully...",
        artifacts_path="/artifacts/jobs/3",
        strategy="LwF",
        strategy_params={"alpha": 1.0, "temperature": 2.0},
        task_id="sentiment_analysis",
        previous_tasks=[]
    )
    
    db.add_all([job1, job2, job3])
    db.flush()
    
    # Create continual learning runs
    cl_run = ContinualLearningRun(
        job_id=job2.id,
        strategy="EWC",
        strategy_params={"lambda": 1.0},
        task_id="digit_recognition_rotated",
        task_metrics={"accuracy": 0.87, "f1": 0.86},
        previous_task_metrics={"digit_recognition_base": {"accuracy": 0.94, "f1": 0.93}},
        forgetting_metrics={"digit_recognition_base": -0.01},
        knowledge_transfer_metrics={"forward_transfer": 0.05}
    )
    
    db.add(cl_run)
    
    # Create deployments
    deployment1 = Deployment(
        name="mnist-prod",
        description="Production deployment of MNIST classifier",
        model_version_id=model_versions[1].id,
        environment="production",
        status="running",
        endpoint_url="https://api.example.com/v1/models/mnist",
        deployment_platform="kubernetes",
        config={"replicas": 3, "resources": {"cpu": "1", "memory": "2Gi"}},
        metrics={"latency_ms": 15, "throughput": 250},
        health_status="healthy",
        replicas=3
    )
    
    db.add(deployment1)
    db.flush()
    
    # Create deployment logs
    log1 = DeploymentLog(
        deployment_id=deployment1.id,
        event_type="created",
        message="Deployment created successfully",
        created_at=now - timedelta(days=2)
    )
    
    log2 = DeploymentLog(
        deployment_id=deployment1.id,
        event_type="updated",
        message="Scaled deployment to 3 replicas",
        created_at=now - timedelta(days=1)
    )
    
    db.add_all([log1, log2])
    db.commit()
    
    logger.info(f"Created {db.query(TrainingJob).count()} training jobs and {db.query(Deployment).count()} deployments")


def main():
    """Main function to initialize and seed the database."""
    # Check database connection
    if not verify_db_connection():
        logger.error("Database connection failed. Make sure the database is running.")
        sys.exit(1)
    
    # Reset database (dev/test only)
    reset = os.environ.get("RESET_DB", "false").lower() == "true"
    if reset:
        logger.warning("Resetting database")
        drop_db()
    
    # Initialize database
    init_db()
    
    # Seed database with sample data
    db = SessionLocal()
    try:
        seed_models(db)
        seed_datasets(db)
        seed_training_and_deployments(db)
        logger.success("Database seeded successfully")
    except Exception as e:
        logger.error(f"Error seeding database: {str(e)}")
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    main() 
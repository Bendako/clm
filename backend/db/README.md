# CLM Database Module

This module handles database connections and models for the CLM (Continuous Learning for Models) backend.

## Setup

The database uses SQLAlchemy with PostgreSQL. To set up the database:

1. Make sure PostgreSQL is installed and running.
2. Create a database for CLM:
   ```sql
   CREATE DATABASE clm;
   ```
3. Set the database URL in your environment:
   ```bash
   export DATABASE_URL=postgresql://username:password@localhost:5432/clm
   ```
   Or set it in a `.env` file in the project root.

## Models

The database includes the following main models:

- **Model Registry Models**
  - `RegisteredModel`: Base model information
  - `ModelVersion`: Versioned instances of models
  - `ModelTask`: Tasks associated with model versions

- **Dataset Models**
  - `Dataset`: Base dataset information
  - `DatasetVersion`: Versioned snapshots of datasets
  - `DatasetTask`: Tasks associated with dataset versions
  - `DataDrift`: Records of detected data drift

- **Training Models**
  - `TrainingJob`: Training jobs for continual learning
  - `ContinualLearningRun`: Detailed metrics for continual learning training

- **Deployment Models**
  - `Deployment`: Model deployment information
  - `DeploymentLog`: Logs of deployment events

## Migrations

Database migrations are managed with Alembic. To create a new migration:

```bash
cd backend/db/migrations
python create_migration.py --autogenerate -m "description of changes"
```

To upgrade the database to the latest version:

```bash
cd backend/db/migrations
alembic upgrade head
```

## Seeding

To populate the database with sample data for development:

```bash
cd backend
python -m db.seed
```

To reset the database before seeding:

```bash
RESET_DB=true python -m db.seed
```

## Usage in API

The database session is provided as a FastAPI dependency:

```python
from db.database import get_db
from sqlalchemy.orm import Session
from fastapi import Depends

@app.get("/some-endpoint")
async def some_endpoint(db: Session = Depends(get_db)):
    # Use the database session
    result = db.query(SomeModel).all()
    return result
``` 
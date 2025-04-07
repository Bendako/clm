"""
Unit tests for the ContinualTrainer class.
"""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import shutil
import tempfile
from unittest.mock import MagicMock, patch
import numpy as np

from ml.training.continual import ContinualTrainer
from ml.continual_learning.ewc import EWC
from ml.replay_buffers.reservoir_sampling import ReservoirBuffer, TensorReservoirBuffer

# Define a simple model for testing
class SimpleModel(nn.Module):
    def __init__(self, input_size=10, hidden_size=50, output_size=2):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Fixtures
@pytest.fixture
def model():
    return SimpleModel(input_size=10, hidden_size=20, output_size=2)

@pytest.fixture
def config():
    return {
        "optimizer": "adam",
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "epochs": 2,
        "continual_strategies": ["ewc", "replay"],
        "ewc": {
            "importance": 1000.0,
            "fisher_sample_size": 10
        },
        "replay": {
            "buffer_size": 100,
            "task_balanced": True,
            "buffer_type": "tensor"
        },
        "save_checkpoints": True
    }

@pytest.fixture
def temp_dir():
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Clean up after test
    shutil.rmtree(temp_dir)
    
@pytest.fixture
def mock_mlflow():
    with patch('ml.training.continual.trainer.mlflow') as mock:
        # Mock the MLflow methods
        mock.start_run.return_value.__enter__.return_value = None
        mock.start_run.return_value.__exit__.return_value = None
        yield mock

@pytest.fixture
def train_data():
    # Create synthetic data for task "A"
    x = torch.randn(50, 10)
    y = torch.randint(0, 2, (50,))
    return TensorDataset(x, y)

@pytest.fixture
def val_data():
    # Create synthetic data for validation
    x = torch.randn(20, 10)
    y = torch.randint(0, 2, (20,))
    return TensorDataset(x, y)

@pytest.fixture
def train_dataloader(train_data):
    return DataLoader(train_data, batch_size=8, shuffle=True)

@pytest.fixture
def val_dataloader(val_data):
    return DataLoader(val_data, batch_size=8, shuffle=False)

# Tests
def test_trainer_init(model, config, temp_dir, mock_mlflow):
    """Test the initialization of the ContinualTrainer."""
    trainer = ContinualTrainer(
        model=model,
        config=config,
        device="cpu",
        experiment_name="test_exp",
        checkpoint_dir=temp_dir
    )
    
    # Check that the model was moved to the device
    assert next(model.parameters()).device.type == "cpu"
    
    # Check that the trainer attributes were correctly set
    assert trainer.model == model
    assert trainer.config == config
    assert trainer.device == "cpu"
    assert trainer.experiment_name == "test_exp"
    assert trainer.checkpoint_dir == temp_dir
    
    # Check that the optimizer was correctly initialized
    assert isinstance(trainer.optimizer, optim.Adam)
    
    # Check continual learning strategies
    assert isinstance(trainer.ewc, EWC)
    assert isinstance(trainer.replay_buffer, TensorReservoirBuffer)
    
    # Check that the training history was initialized
    assert trainer.training_history == {"tasks": [], "metrics": {}}
    
    # Check that MLflow was initialized
    mock_mlflow.set_experiment.assert_called_once_with("test_exp")

def test_init_optimizer(model, config):
    """Test the initialization of different optimizers."""
    # Test Adam optimizer
    config["optimizer"] = "adam"
    trainer = ContinualTrainer(model=model, config=config, device="cpu")
    assert isinstance(trainer.optimizer, optim.Adam)
    
    # Test AdamW optimizer
    config["optimizer"] = "adamw"
    trainer = ContinualTrainer(model=model, config=config, device="cpu")
    assert isinstance(trainer.optimizer, optim.AdamW)
    
    # Test SGD optimizer
    config["optimizer"] = "sgd"
    config["momentum"] = 0.9
    trainer = ContinualTrainer(model=model, config=config, device="cpu")
    assert isinstance(trainer.optimizer, optim.SGD)
    
    # Test invalid optimizer
    config["optimizer"] = "invalid"
    with pytest.raises(ValueError):
        ContinualTrainer(model=model, config=config, device="cpu")
        
    # Test linear scheduler
    config["optimizer"] = "adam"
    config["scheduler"] = "linear"
    config["scheduler_steps"] = 100
    trainer = ContinualTrainer(model=model, config=config, device="cpu")
    assert isinstance(trainer.scheduler, optim.lr_scheduler.LinearLR)
    
    # Test cosine scheduler
    config["scheduler"] = "cosine"
    trainer = ContinualTrainer(model=model, config=config, device="cpu")
    assert isinstance(trainer.scheduler, optim.lr_scheduler.CosineAnnealingLR)

def test_setup_continual_strategies(model, config):
    """Test setting up different continual learning strategies."""
    # Test with both EWC and replay
    config["continual_strategies"] = ["ewc", "replay"]
    trainer = ContinualTrainer(model=model, config=config, device="cpu")
    assert isinstance(trainer.ewc, EWC)
    assert isinstance(trainer.replay_buffer, TensorReservoirBuffer)
    
    # Test with only EWC
    config["continual_strategies"] = ["ewc"]
    trainer = ContinualTrainer(model=model, config=config, device="cpu")
    assert isinstance(trainer.ewc, EWC)
    assert trainer.replay_buffer is None
    
    # Test with only replay
    config["continual_strategies"] = ["replay"]
    trainer = ContinualTrainer(model=model, config=config, device="cpu")
    assert trainer.ewc is None
    assert isinstance(trainer.replay_buffer, TensorReservoirBuffer)
    
    # Test with non-tensor replay buffer
    config["replay"]["buffer_type"] = "standard"
    trainer = ContinualTrainer(model=model, config=config, device="cpu")
    assert isinstance(trainer.replay_buffer, ReservoirBuffer)
    assert not isinstance(trainer.replay_buffer, TensorReservoirBuffer)
    
    # Test with no strategies
    config["continual_strategies"] = []
    trainer = ContinualTrainer(model=model, config=config, device="cpu")
    assert trainer.ewc is None
    assert trainer.replay_buffer is None

@patch('ml.training.continual.trainer.mlflow')
def test_train_task_single_epoch(mock_mlflow, model, config, train_dataloader, val_dataloader):
    """Test the train_task method for a single task and epoch."""
    trainer = ContinualTrainer(model=model, config=config, device="cpu")
    
    # Mock the _train_epoch and _validate methods to avoid actual training
    trainer._train_epoch = MagicMock(return_value={"train_loss": 0.5, "epoch": 1})
    trainer._validate = MagicMock(return_value={"val_loss": 0.6})
    trainer._save_checkpoint = MagicMock()
    
    # Train on a single task for a single epoch
    task_id = "task_A"
    result = trainer.train_task(
        task_id=task_id,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        num_epochs=1
    )
    
    # Check that the task was tracked
    assert task_id in trainer.training_history["tasks"]
    assert len(trainer.training_history["metrics"][task_id]) == 1
    
    # Check that methods were called correctly
    trainer._train_epoch.assert_called_once()
    trainer._validate.assert_called_once_with(task_id, val_dataloader, None)
    trainer._save_checkpoint.assert_called_with(task_id, 1, is_final=True)
    
    # Check that the result contains the expected data
    assert result["task_id"] == task_id
    assert len(result["epochs"]) == 1
    assert result["epochs"][0]["train_loss"] == 0.5
    assert result["epochs"][0]["val_loss"] == 0.6

@patch('ml.training.continual.trainer.mlflow')
def test_train_epoch(mock_mlflow, model, config, train_dataloader):
    """Test the _train_epoch method with actual forward and backward passes."""
    # Use a simple config for this test
    simple_config = {
        "optimizer": "adam",
        "learning_rate": 0.001,
        "epochs": 1,
        "continual_strategies": []
    }
    
    trainer = ContinualTrainer(model=model, config=simple_config, device="cpu")
    
    # Define a loss function
    loss_fn = nn.CrossEntropyLoss()
    
    # Train for one epoch
    metrics = trainer._train_epoch(0, "task_A", train_dataloader, loss_fn)
    
    # Check that metrics were computed
    assert "train_loss" in metrics
    assert metrics["epoch"] == 1
    assert isinstance(metrics["train_loss"], float)

@patch('ml.training.continual.trainer.mlflow')
def test_validate(mock_mlflow, model, config, val_dataloader):
    """Test the _validate method with actual forward passes."""
    trainer = ContinualTrainer(model=model, config=config, device="cpu")
    
    # Define a loss function
    loss_fn = nn.CrossEntropyLoss()
    
    # Validate
    metrics = trainer._validate("task_A", val_dataloader, loss_fn)
    
    # Check that metrics were computed
    assert "val_loss" in metrics
    assert isinstance(metrics["val_loss"], float)

@patch('ml.training.continual.trainer.mlflow')
def test_save_and_load_checkpoint(mock_mlflow, model, config, temp_dir):
    """Test saving and loading a checkpoint."""
    trainer = ContinualTrainer(
        model=model,
        config=config,
        device="cpu",
        checkpoint_dir=temp_dir
    )
    
    # Save a checkpoint
    task_id = "task_A"
    epoch = 2
    trainer._save_checkpoint(task_id, epoch, is_final=True)
    
    # Check that the checkpoint file exists
    checkpoint_path = os.path.join(temp_dir, f"{task_id}_final.pt")
    assert os.path.exists(checkpoint_path)
    
    # Create a new model and trainer
    new_model = SimpleModel(input_size=10, hidden_size=20, output_size=2)
    new_trainer = ContinualTrainer(
        model=new_model,
        config=config,
        device="cpu",
        checkpoint_dir=temp_dir
    )
    
    # Load the checkpoint
    new_trainer.load_checkpoint(checkpoint_path)
    
    # Check that the model parameters were loaded correctly
    for p1, p2 in zip(model.parameters(), new_model.parameters()):
        assert torch.all(torch.eq(p1.data, p2.data))
    
    # Check that loading a non-existent checkpoint raises an error
    with pytest.raises(FileNotFoundError):
        new_trainer.load_checkpoint("nonexistent.pt")

@patch('ml.training.continual.trainer.mlflow')
def test_ewc_integration(mock_mlflow, model, config, train_dataloader, val_dataloader):
    """Test the integration with EWC."""
    # Mock EWC methods
    with patch('ml.training.continual.trainer.EWC') as mock_ewc:
        mock_ewc_instance = MagicMock()
        mock_ewc.return_value = mock_ewc_instance
        mock_ewc_instance.register_task.return_value = None
        mock_ewc_instance.apply_ewc_training.return_value = nn.CrossEntropyLoss()
        
        trainer = ContinualTrainer(model=model, config=config, device="cpu")
        
        # Check that EWC was initialized
        mock_ewc.assert_called_once()
        
        # Mock trainer methods
        trainer._train_epoch = MagicMock(return_value={"train_loss": 0.5, "epoch": 1})
        trainer._validate = MagicMock(return_value={"val_loss": 0.6})
        trainer._save_checkpoint = MagicMock()
        
        # Train on first task
        task_id_1 = "task_A"
        trainer.train_task(
            task_id=task_id_1,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            num_epochs=1
        )
        
        # EWC should register the first task
        mock_ewc_instance.register_task.assert_called_once()
        
        # Train on second task
        task_id_2 = "task_B"
        trainer.train_task(
            task_id=task_id_2,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            num_epochs=1
        )
        
        # EWC should apply regularization for the second task
        mock_ewc_instance.apply_ewc_training.assert_called_once()

@patch('ml.training.continual.trainer.mlflow')
def test_replay_integration(mock_mlflow, model, config, train_dataloader):
    """Test the integration with replay buffer."""
    # Set replay-specific config
    config["replay_save_freq"] = 1
    config["replay_freq"] = 2
    config["replay_batch_size"] = 4
    
    # Create trainer with mocked replay buffer
    with patch('ml.training.continual.trainer.TensorReservoirBuffer') as mock_buffer:
        mock_buffer_instance = MagicMock()
        mock_buffer.return_value = mock_buffer_instance
        mock_buffer_instance.is_empty.return_value = False
        mock_buffer_instance.sample.return_value = [torch.randn(10) for _ in range(4)]
        
        trainer = ContinualTrainer(model=model, config=config, device="cpu")
        
        # Check that replay buffer was initialized
        mock_buffer.assert_called_once()
        
        # Call _train_epoch directly to test replay integration
        loss_fn = nn.CrossEntropyLoss()
        trainer._train_epoch(0, "task_A", train_dataloader, loss_fn)
        
        # Check that add_batch was called
        assert mock_buffer_instance.add_batch.call_count > 0
        
        # Check that sample was called
        assert mock_buffer_instance.sample.call_count > 0

@patch('ml.training.continual.trainer.mlflow')
def test_evaluate_all_tasks(mock_mlflow, model, config):
    """Test evaluating all tasks."""
    trainer = ContinualTrainer(model=model, config=config, device="cpu")
    
    # Mock _validate method
    trainer._validate = MagicMock(side_effect=[
        {"val_loss": 0.5},  # task_A
        {"val_loss": 0.7}   # task_B
    ])
    
    # Create dummy dataloaders
    task_dataloaders = {
        "task_A": MagicMock(),
        "task_B": MagicMock()
    }
    
    # Evaluate tasks
    results = trainer.evaluate_all_tasks(task_dataloaders)
    
    # Check results
    assert "task_A" in results
    assert "task_B" in results
    assert results["task_A"]["val_loss"] == 0.5
    assert results["task_B"]["val_loss"] == 0.7
    assert trainer._validate.call_count == 2

@patch('ml.training.continual.trainer.mlflow')
def test_compute_forgetting(mock_mlflow, model, config):
    """Test computing forgetting metrics."""
    trainer = ContinualTrainer(model=model, config=config, device="cpu")
    
    # Mock evaluate_all_tasks method
    current_metrics = {
        "task_A": {"val_loss": 0.7},
        "task_B": {"val_loss": 0.8}
    }
    trainer.evaluate_all_tasks = MagicMock(return_value=current_metrics)
    
    # Define reference metrics
    reference_metrics = {
        "task_A": {"val_loss": 0.5},
        "task_B": {"val_loss": 0.6}
    }
    
    # Create dummy dataloaders
    task_dataloaders = {
        "task_A": MagicMock(),
        "task_B": MagicMock()
    }
    
    # Compute forgetting
    forgetting = trainer.compute_forgetting(
        task_dataloaders, reference_metrics=reference_metrics)
    
    # Check forgetting metrics
    assert "task_A" in forgetting
    assert "task_B" in forgetting
    assert forgetting["task_A"] == 0.2  # 0.7 - 0.5
    assert forgetting["task_B"] == 0.2  # 0.8 - 0.6
    
    # Test without reference metrics
    result = trainer.compute_forgetting(task_dataloaders)
    
    # Should return current metrics
    assert "task_A" in result
    assert "task_B" in result
    assert result["task_A"]["val_loss"] == 0.7
    assert result["task_B"]["val_loss"] == 0.8 
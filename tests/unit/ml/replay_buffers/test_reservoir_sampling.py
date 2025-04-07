"""
Unit tests for the Reservoir Sampling Replay Buffer.
"""

import pytest
import random
import numpy as np
import torch
from ml.replay_buffers.reservoir_sampling import ReservoirBuffer, TensorReservoirBuffer

# Ensure tests are reproducible
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

class TestReservoirBuffer:
    """Test suite for the generic ReservoirBuffer class."""
    
    def test_initialization(self):
        """Test that the buffer initializes correctly."""
        buffer = ReservoirBuffer(capacity=100)
        assert buffer.capacity == 100
        assert len(buffer) == 0
        assert buffer.is_empty()
        assert not buffer.is_full()
        
    def test_add_items(self):
        """Test adding items to the buffer."""
        buffer = ReservoirBuffer(capacity=5)
        
        # Add items
        for i in range(5):
            buffer.add(f"item{i}", f"task{i%2}")
            
        assert len(buffer) == 5
        assert buffer.is_full()
        assert not buffer.is_empty()
        
        # Check task distributions
        task_dist = buffer.get_task_distribution()
        assert "task0" in task_dist
        assert "task1" in task_dist
        
    def test_reservoir_sampling_property(self):
        """Test that the buffer maintains reservoir sampling properties."""
        capacity = 10
        buffer = ReservoirBuffer(capacity=capacity, seed=SEED)
        
        # Add more items than capacity
        for i in range(100):
            buffer.add(i, f"task{i%4}")
            
        # Buffer should be at capacity
        assert len(buffer) == capacity
        assert buffer.items_seen == 100
        
        # Get all items
        items = buffer.get_all_items()
        assert len(items) == capacity
        
        # Items should be a subset of the original items
        for item in items:
            assert 0 <= item < 100
            
    def test_task_balanced_sampling(self):
        """Test that task-balanced sampling works correctly."""
        buffer = ReservoirBuffer(capacity=100, task_balanced=True, seed=SEED)
        
        # Add items from different tasks
        for i in range(100):
            buffer.add(i, f"task{i%4}")
            
        # Sample with task balancing
        samples = buffer.sample(40)
        
        # Count samples per task
        task_counts = {}
        for i in range(len(samples)):
            # Get the task for this item
            task_id = buffer.task_ids[buffer.buffer.index(samples[i])]
            task_counts[task_id] = task_counts.get(task_id, 0) + 1
            
        # Since we have 4 tasks and 40 samples, we expect ~10 samples per task
        for task, count in task_counts.items():
            assert 8 <= count <= 12, f"Task {task} has {count} samples, expected close to 10"
            
    def test_uniform_sampling(self):
        """Test uniform sampling from the buffer."""
        buffer = ReservoirBuffer(capacity=100, task_balanced=False, seed=SEED)
        
        # Add items with skewed task distribution
        for i in range(80):
            buffer.add(i, "task0")
        for i in range(80, 100):
            buffer.add(i, "task1")
            
        # Sample uniformly
        samples = buffer.sample(50)
        assert len(samples) == 50
        
        # Count samples per task
        task_counts = {}
        for i in range(len(samples)):
            # Get the task for this item
            task_id = buffer.task_ids[buffer.buffer.index(samples[i])]
            task_counts[task_id] = task_counts.get(task_id, 0) + 1
            
        # With uniform sampling, we expect distribution similar to buffer contents
        assert task_counts.get("task0", 0) > task_counts.get("task1", 0)
            
    def test_sample_task(self):
        """Test sampling from a specific task."""
        buffer = ReservoirBuffer(capacity=100, seed=SEED)
        
        # Add items from different tasks
        for i in range(100):
            buffer.add(i, f"task{i%4}")
            
        # Sample from a specific task
        task_samples = buffer.sample_task("task2", 10)
        assert len(task_samples) == 10
        
        # Verify all samples are from task2
        for item in task_samples:
            # Get the task for this item
            task_id = buffer.task_ids[buffer.buffer.index(item)]
            assert task_id == "task2"
            
    def test_clear(self):
        """Test clearing the buffer."""
        buffer = ReservoirBuffer(capacity=10)
        
        # Add some items
        for i in range(5):
            buffer.add(i, "task0")
            
        assert len(buffer) == 5
        
        # Clear the buffer
        buffer.clear()
        assert len(buffer) == 0
        assert buffer.is_empty()
        assert buffer.get_task_distribution() == {}


class TestTensorReservoirBuffer:
    """Test suite for the TensorReservoirBuffer class."""
    
    def test_initialization(self):
        """Test that the tensor buffer initializes correctly."""
        buffer = TensorReservoirBuffer(capacity=100)
        assert buffer.capacity == 100
        assert len(buffer) == 0
        
    def test_add_tensors(self):
        """Test adding tensors to the buffer."""
        buffer = TensorReservoirBuffer(capacity=5, device="cpu")
        
        # Add tensor items
        for i in range(5):
            buffer.add(torch.tensor([i, i+1, i+2], dtype=torch.float32), f"task{i%2}")
            
        assert len(buffer) == 5
        assert buffer.is_full()
        
        # Check task distributions
        task_dist = buffer.get_task_distribution()
        assert "task0" in task_dist
        assert "task1" in task_dist
        
    def test_sample_tensors(self):
        """Test sampling tensors as a batch."""
        buffer = TensorReservoirBuffer(capacity=20, seed=SEED)
        
        # Add tensors
        for i in range(20):
            buffer.add(torch.tensor([i, i+1, i+2], dtype=torch.float32), f"task{i%4}")
            
        # Sample as tensor batch
        batch = buffer.sample_tensors(10)
        
        # Check batch properties
        assert isinstance(batch, torch.Tensor)
        assert batch.shape == (10, 3)  # 10 samples of 3-element vectors
        
    def test_sample_task_tensors(self):
        """Test sampling tensors from a specific task."""
        buffer = TensorReservoirBuffer(capacity=20, seed=SEED)
        
        # Add tensors
        for i in range(20):
            buffer.add(torch.tensor([i, i+1, i+2], dtype=torch.float32), f"task{i%4}")
            
        # Sample from a specific task
        batch = buffer.sample_task_tensors("task2", 5)
        
        # Check batch properties
        assert isinstance(batch, torch.Tensor)
        assert batch.shape == (5, 3)  # 5 samples of 3-element vectors
        
        # Verify all samples are from task2 by checking the original indices
        for i in range(batch.shape[0]):
            sample = batch[i]
            # Find this tensor in the buffer
            for j, tensor in enumerate(buffer.buffer):
                if torch.all(tensor == sample):
                    assert buffer.task_ids[j] == "task2"
                    break 
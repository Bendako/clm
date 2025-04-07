#!/usr/bin/env python
"""
Unit tests for the PackNet strategy.

These tests verify that the PackNet strategy correctly implements pruning,
masking, and parameter freezing for continual learning.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pytest

from ml.continual_strategies.packnet import PackNet


class SimpleModel(nn.Module):
    """Simple model for testing PackNet."""
    
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 10)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class TestPackNet:
    """Test class for PackNet strategy."""
    
    @pytest.fixture
    def model(self):
        """Create a simple model for testing."""
        return SimpleModel()
    
    @pytest.fixture
    def packnet(self, model):
        """Create a PackNet instance for testing."""
        return PackNet(
            model=model,
            prune_percentage=0.5,
            prune_threshold=0.001,
            use_magnitude_pruning=True
        )
    
    def test_initialization(self, packnet):
        """Test that PackNet is initialized correctly."""
        assert packnet.prune_percentage == 0.5
        assert packnet.prune_threshold == 0.001
        assert packnet.use_magnitude_pruning is True
        assert packnet.task_masks == {}
        assert packnet.current_task_id is None
    
    def test_prepare_model_first_task(self, packnet, model):
        """Test model preparation for the first task."""
        # Prepare model for first task
        packnet.prepare_model_for_task(task_id=0)
        
        # Check that task_id is set
        assert packnet.current_task_id == 0
        
        # Check that no masks exist yet
        assert 0 not in packnet.task_masks
        
        # Check model is in training mode
        assert model.training
        
        # Check all parameters require grad
        for param in model.parameters():
            assert param.requires_grad
    
    def test_prepare_model_second_task(self, packnet, model):
        """Test model preparation for the second task after masking."""
        # Prepare for first task
        packnet.prepare_model_for_task(task_id=0)
        
        # Create some dummy gradients and prune
        for param in model.parameters():
            param.grad = torch.randn_like(param)
        
        # Prune first task
        packnet._create_mask_for_task(task_id=0)
        
        # Prepare for second task
        packnet.prepare_model_for_task(task_id=1)
        
        # Check that task_id is updated
        assert packnet.current_task_id == 1
        
        # Check that mask exists for first task
        assert 0 in packnet.task_masks
        
        # Check model is in training mode
        assert model.training
    
    def test_create_mask(self, packnet, model):
        """Test mask creation based on weight magnitudes."""
        # Prepare for task
        packnet.prepare_model_for_task(task_id=0)
        
        # Set weights to known values for predictable pruning
        with torch.no_grad():
            for param in model.parameters():
                param.data = torch.linspace(0, 1, param.numel()).reshape(param.shape)
                param.grad = torch.ones_like(param)
        
        # Create mask
        packnet._create_mask_for_task(task_id=0)
        
        # Check mask exists
        assert 0 in packnet.task_masks
        
        # Check mask shape matches parameters
        param_shapes = [p.shape for p in model.parameters()]
        mask_shapes = [m.shape for m in packnet.task_masks[0]]
        
        assert len(param_shapes) == len(mask_shapes)
        for ps, ms in zip(param_shapes, mask_shapes):
            assert ps == ms
            
        # Check pruning percentage is approximately correct
        total_params = sum(p.numel() for p in model.parameters())
        masked_params = sum(m.sum().item() for m in packnet.task_masks[0])
        
        # Should have kept approximately 50% of parameters
        assert 0.45 <= masked_params / total_params <= 0.55
    
    def test_apply_mask(self, packnet, model):
        """Test applying a task-specific mask."""
        # Prepare for task 0
        packnet.prepare_model_for_task(task_id=0)
        
        # Set weights to known values
        with torch.no_grad():
            for param in model.parameters():
                param.data = torch.ones_like(param)
                param.grad = torch.ones_like(param)
        
        # Create mask for task 0
        packnet._create_mask_for_task(task_id=0)
        
        # Set weights to different values
        with torch.no_grad():
            for param in model.parameters():
                param.data = 2 * torch.ones_like(param)
        
        # Apply mask for task 0
        packnet._apply_task_specific_mask(task_id=0)
        
        # Check that masked weights are zero and unmasked weights are 2
        for param_idx, param in enumerate(model.parameters()):
            mask = packnet.task_masks[0][param_idx]
            
            # Masked elements should be 0
            assert torch.all((param.data[mask == 0] == 0))
            
            # Unmasked elements should still be 2
            assert torch.all((param.data[mask == 1] == 2))
    
    def test_compute_loss(self, packnet, model):
        """Test loss computation with regularization."""
        # Create dummy inputs and targets
        x = torch.randn(5, 10)
        y = torch.randn(5, 10)
        
        # Criterion
        criterion = nn.MSELoss()
        
        # Prepare model and create mask
        packnet.prepare_model_for_task(task_id=0)
        with torch.no_grad():
            for param in model.parameters():
                param.grad = torch.ones_like(param)
        packnet._create_mask_for_task(task_id=0)
        
        # Compute loss for task 0
        loss = packnet.compute_loss(criterion, model(x), y)
        
        # Loss should be the standard criterion loss for task 0
        expected_loss = criterion(model(x), y)
        assert loss.item() == expected_loss.item()
        
        # Prepare for task 1
        packnet.prepare_model_for_task(task_id=1)
        
        # Compute loss for task 1 (should be the same since no regularization is applied yet)
        loss = packnet.compute_loss(criterion, model(x), y)
        assert loss.item() == expected_loss.item()
    
    def test_after_training(self, packnet, model):
        """Test behavior after training a task."""
        # Prepare model for task 0
        packnet.prepare_model_for_task(task_id=0)
        
        # Set gradients
        for param in model.parameters():
            param.grad = torch.ones_like(param)
        
        # After training task 0
        packnet.after_training(task_id=0)
        
        # Check mask was created
        assert 0 in packnet.task_masks
        
        # Prepare for task 1
        packnet.prepare_model_for_task(task_id=1)
        
        # Set all weights to 1
        with torch.no_grad():
            for param in model.parameters():
                param.data = torch.ones_like(param)
                param.grad = torch.ones_like(param)
        
        # After training task 1
        packnet.after_training(task_id=1)
        
        # Check mask was created for task 1
        assert 1 in packnet.task_masks
        
        # Apply mask for task 0 and check that appropriate weights are preserved
        packnet._apply_task_specific_mask(task_id=0)
        for param_idx, param in enumerate(model.parameters()):
            mask = packnet.task_masks[0][param_idx]
            # Masked elements should be 0
            assert torch.all(param.data[mask == 0] == 0)
            # Unmasked elements should be 1
            assert torch.all(param.data[mask == 1] == 1)
    
    def test_before_backward(self, packnet, model):
        """Test before_backward method."""
        # Nothing much happens here for PackNet, just check it doesn't crash
        packnet.prepare_model_for_task(task_id=0)
        packnet.before_backward()
        assert True  # If we get here, it didn't crash
    
    def test_before_update(self, packnet, model):
        """Test before_update method."""
        # Nothing much happens here for PackNet, just check it doesn't crash
        packnet.prepare_model_for_task(task_id=0)
        packnet.before_update()
        assert True  # If we get here, it didn't crash
    
    def test_after_update(self, packnet, model):
        """Test after_update method which should apply masks for previous tasks."""
        # Prepare model for task 0
        packnet.prepare_model_for_task(task_id=0)
        
        # Set gradients and create mask
        for param in model.parameters():
            param.grad = torch.ones_like(param)
        packnet.after_training(task_id=0)
        
        # Prepare for task 1
        packnet.prepare_model_for_task(task_id=1)
        
        # Set all weights to 1
        with torch.no_grad():
            for param in model.parameters():
                param.data = torch.ones_like(param)
        
        # After update should apply previous masks
        packnet.after_update()
        
        # Check that task 0 mask was applied
        for param_idx, param in enumerate(model.parameters()):
            mask = packnet.task_masks[0][param_idx]
            # Weights for task 0 should be preserved
            assert torch.all(param.data[mask == 1] == 1)
            
            # Weights not used by task 0 should still be 1
            assert torch.all(param.data[mask == 0] == 1)
    
    def test_state_dict(self, packnet, model):
        """Test saving and loading state dict."""
        # Prepare model for task 0
        packnet.prepare_model_for_task(task_id=0)
        
        # Set gradients and create mask
        for param in model.parameters():
            param.grad = torch.ones_like(param)
        packnet.after_training(task_id=0)
        
        # Save state dict
        state_dict = packnet.state_dict()
        
        # Check state dict contains expected keys
        assert "task_masks" in state_dict
        assert "current_task_id" in state_dict
        assert "prune_percentage" in state_dict
        assert "prune_threshold" in state_dict
        assert "use_magnitude_pruning" in state_dict
        
        # Create new PackNet with different params
        new_packnet = PackNet(
            model=model,
            prune_percentage=0.75,  # Different from original
            prune_threshold=0.002,  # Different from original
            use_magnitude_pruning=False  # Different from original
        )
        
        # Load state dict
        new_packnet.load_state_dict(state_dict)
        
        # Check values were loaded correctly
        assert new_packnet.prune_percentage == 0.5
        assert new_packnet.prune_threshold == 0.001
        assert new_packnet.use_magnitude_pruning is True
        assert new_packnet.current_task_id == 0
        assert 0 in new_packnet.task_masks
        
        # Check masks are identical
        for orig_mask, new_mask in zip(packnet.task_masks[0], new_packnet.task_masks[0]):
            assert torch.all(orig_mask == new_mask)
    
    def test_count_remaining_params(self, packnet, model):
        """Test counting remaining parameters after pruning."""
        # Prepare for task 0
        packnet.prepare_model_for_task(task_id=0)
        
        # Set gradients and create mask
        for param in model.parameters():
            param.grad = torch.ones_like(param)
        packnet.after_training(task_id=0)
        
        # Count total parameters
        total_params = sum(p.numel() for p in model.parameters())
        
        # Count non-zero parameters after pruning
        non_zero_params = packnet._count_remaining_params()
        
        # Should have roughly 50% of parameters remaining
        assert 0.45 <= non_zero_params / total_params <= 0.55
    
    def test_load_task_masks(self, packnet, model):
        """Test loading task masks from state dict."""
        # Prepare model for task 0
        packnet.prepare_model_for_task(task_id=0)
        
        # Set gradients and create mask
        for param in model.parameters():
            param.grad = torch.ones_like(param)
        packnet.after_training(task_id=0)
        
        # Save task masks
        task_masks = packnet.task_masks.copy()
        
        # Clear task masks
        packnet.task_masks = {}
        
        # Load task masks
        packnet.task_masks = task_masks
        
        # Check masks are loaded
        assert 0 in packnet.task_masks
        
        # Apply mask to verify it works
        packnet._apply_task_specific_mask(task_id=0)
        
        # Check that masked parameters are zero
        for param_idx, param in enumerate(model.parameters()):
            mask = packnet.task_masks[0][param_idx]
            assert torch.all(param.data[mask == 0] == 0)


if __name__ == "__main__":
    pytest.main(["-xvs", "test_packnet.py"])
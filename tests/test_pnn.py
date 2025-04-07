"""
Unit tests for Progressive Neural Networks (PNN) continual learning strategy.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pytest
from typing import Dict, List, Tuple

from ml.continual_strategies import ProgressiveNetworks, ProgressiveNeuralNetwork
from ml.continual_strategies.pnn import ColumnLayer, ProgressiveColumn

# Simple test model
class SimpleModel(nn.Module):
    def __init__(self, input_size=10, hidden_size=20, output_size=5):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)


class TestColumnLayer:
    """Tests for the ColumnLayer class."""
    
    def test_init(self):
        """Test initialization of ColumnLayer."""
        layer = ColumnLayer(10, 20)
        assert layer.main.in_features == 10
        assert layer.main.out_features == 20
        assert len(layer.adapters) == 0
    
    def test_init_with_previous_layers(self):
        """Test initialization with previous layers."""
        prev_layer = ColumnLayer(10, 15)
        layer = ColumnLayer(10, 20, previous_layers=[prev_layer])
        assert len(layer.adapters) == 1
        assert layer.adapters[0].in_features == 15
        assert layer.adapters[0].out_features == 20
    
    def test_forward(self):
        """Test forward pass without previous outputs."""
        layer = ColumnLayer(10, 20)
        x = torch.randn(5, 10)
        output = layer(x)
        assert output.shape == (5, 20)
    
    def test_forward_with_prev_outputs(self):
        """Test forward pass with previous outputs."""
        prev_layer = ColumnLayer(10, 15)
        layer = ColumnLayer(10, 20, previous_layers=[prev_layer])
        
        x = torch.randn(5, 10)
        prev_output = torch.randn(5, 15)
        
        # Without previous outputs
        output1 = layer(x)
        assert output1.shape == (5, 20)
        
        # With previous outputs
        output2 = layer(x, [prev_output])
        assert output2.shape == (5, 20)
        
        # The outputs should be different due to lateral connections
        assert not torch.allclose(output1, output2)


class TestProgressiveColumn:
    """Tests for the ProgressiveColumn class."""
    
    def test_init(self):
        """Test initialization of ProgressiveColumn."""
        column = ProgressiveColumn([10, 20, 5])
        assert len(column.layers) == 2  # 10->20, 20->5
    
    def test_init_with_previous_columns(self):
        """Test initialization with previous columns."""
        prev_column = ProgressiveColumn([10, 20, 5])
        column = ProgressiveColumn([10, 20, 5], previous_columns=[prev_column])
        assert len(column.layers) == 2
    
    def test_forward(self):
        """Test forward pass without previous columns."""
        column = ProgressiveColumn([10, 20, 5])
        x = torch.randn(5, 10)
        output = column(x)
        assert output.shape == (5, 5)
        assert len(column.activations) == 2
    
    def test_forward_with_prev_columns(self):
        """Test forward pass with previous columns."""
        prev_column = ProgressiveColumn([10, 20, 5])
        column = ProgressiveColumn([10, 20, 5], previous_columns=[prev_column])
        
        x = torch.randn(5, 10)
        
        # Run through previous column first to get activations
        prev_column(x)
        
        # Run through current column with previous column's activations
        output = column(x, [prev_column])
        assert output.shape == (5, 5)
        assert len(column.activations) == 2


class TestProgressiveNeuralNetwork:
    """Tests for the ProgressiveNeuralNetwork class."""
    
    def test_init(self):
        """Test initialization of ProgressiveNeuralNetwork."""
        pnn = ProgressiveNeuralNetwork(10, [20, 15], 5)
        assert pnn.input_size == 10
        assert pnn.hidden_sizes == [20, 15]
        assert pnn.output_size == 5
        assert len(pnn.columns) == 0
        assert len(pnn.task_to_column) == 0
    
    def test_add_column(self):
        """Test adding a column."""
        pnn = ProgressiveNeuralNetwork(10, [20], 5)
        
        # Add first column
        pnn.add_column(0)
        assert len(pnn.columns) == 1
        assert pnn.task_to_column == {0: 0}
        assert pnn.active_column == 0
        
        # Add second column
        pnn.add_column(1)
        assert len(pnn.columns) == 2
        assert pnn.task_to_column == {0: 0, 1: 1}
        assert pnn.active_column == 1
    
    def test_set_active_column(self):
        """Test setting the active column."""
        pnn = ProgressiveNeuralNetwork(10, [20], 5)
        pnn.add_column(0)
        pnn.add_column(1)
        
        # Set active column to first column
        pnn.set_active_column(0)
        assert pnn.active_column == 0
        
        # Set active column to second column
        pnn.set_active_column(1)
        assert pnn.active_column == 1
        
        # Test with invalid task ID
        with pytest.raises(ValueError):
            pnn.set_active_column(2)
    
    def test_forward(self):
        """Test forward pass."""
        pnn = ProgressiveNeuralNetwork(10, [20], 5)
        
        # Test with no columns
        with pytest.raises(ValueError):
            pnn(torch.randn(5, 10))
        
        # Add a column and test forward pass
        pnn.add_column(0)
        x = torch.randn(5, 10)
        output = pnn(x)
        assert output.shape == (5, 5)
        
        # Add another column and test forward pass
        pnn.add_column(1)
        output = pnn(x)  # Should use the second column
        assert output.shape == (5, 5)
        
        # Switch to first column and test forward pass
        pnn.set_active_column(0)
        output = pnn(x)
        assert output.shape == (5, 5)


class TestProgressiveNetworks:
    """Tests for the ProgressiveNetworks strategy."""
    
    def test_init_from_scratch(self):
        """Test initialization with parameters for a new PNN."""
        model = SimpleModel()
        pnn_strategy = ProgressiveNetworks(
            model=model,
            input_size=10,
            hidden_sizes=[20],
            output_size=5
        )
        assert isinstance(pnn_strategy.model, ProgressiveNeuralNetwork)
        assert pnn_strategy.model.input_size == 10
        assert pnn_strategy.model.hidden_sizes == [20]
        assert pnn_strategy.model.output_size == 5
    
    def test_init_with_existing_pnn(self):
        """Test initialization with an existing PNN model."""
        pnn_model = ProgressiveNeuralNetwork(10, [20], 5)
        pnn_strategy = ProgressiveNetworks(model=pnn_model)
        assert pnn_strategy.model is pnn_model
    
    def test_before_training(self):
        """Test before_training hook."""
        pnn_strategy = ProgressiveNetworks(
            model=SimpleModel(),
            input_size=10,
            hidden_sizes=[20],
            output_size=5
        )
        
        # First task should add a column
        pnn_strategy.before_training(0)
        assert len(pnn_strategy.model.columns) == 1
        assert pnn_strategy.model.active_column == 0
        
        # Second task should add another column
        pnn_strategy.before_training(1)
        assert len(pnn_strategy.model.columns) == 2
        assert pnn_strategy.model.active_column == 1
        
        # Revisiting first task should not add a column
        pnn_strategy.before_training(0)
        assert len(pnn_strategy.model.columns) == 2
        assert pnn_strategy.model.active_column == 0
    
    def test_compute_loss(self):
        """Test compute_loss method."""
        pnn_strategy = ProgressiveNetworks(
            model=SimpleModel(),
            input_size=10,
            hidden_sizes=[20],
            output_size=5
        )
        
        # Setup for loss computation
        outputs = torch.randn(5, 5)
        targets = torch.randint(0, 5, (5,))
        criterion = nn.CrossEntropyLoss()
        
        # Regular loss should be returned
        loss = pnn_strategy.compute_loss(outputs, targets, 0, criterion)
        expected_loss = criterion(outputs, targets)
        assert torch.isclose(loss, expected_loss)
    
    def test_save_and_load_state(self, tmpdir):
        """Test saving and loading the PNN state."""
        # Create a PNN strategy and add some columns
        pnn_strategy = ProgressiveNetworks(
            model=SimpleModel(),
            input_size=10,
            hidden_sizes=[20],
            output_size=5
        )
        pnn_strategy.before_training(0)
        pnn_strategy.before_training(1)
        
        # Save state
        save_path = tmpdir.join("pnn_state.pt")
        pnn_strategy.save_state(save_path)
        
        # Create a new strategy and load the state
        new_strategy = ProgressiveNetworks(
            model=SimpleModel(),
            input_size=10,
            hidden_sizes=[20],
            output_size=5
        )
        new_strategy.load_state(save_path)
        
        # Check the loaded state
        assert len(new_strategy.model.columns) == 2
        assert new_strategy.model.task_to_column == {0: 0, 1: 1}
        
        # Test forward pass with loaded model
        x = torch.randn(5, 10)
        output = new_strategy.model(x)
        assert output.shape == (5, 5) 
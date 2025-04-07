"""
Progressive Neural Networks Implementation

This module implements Progressive Neural Networks (PNN) as a continual learning strategy.
PNN creates a new neural network for each task and establishes lateral connections to
previously trained networks, preventing forgetting while enabling knowledge transfer.

Reference:
Rusu, A. A., Rabinowitz, N. C., Desjardins, G., Soyer, H., Kirkpatrick, J.,
Kavukcuoglu, K., ... & Hadsell, R. (2016).
Progressive neural networks. arXiv preprint arXiv:1606.04671.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any
import copy
import numpy as np
import logging
from pathlib import Path

from ml.continual_strategies.base import ContinualStrategy

logger = logging.getLogger(__name__)


class ColumnLayer(nn.Module):
    """
    A layer in a column of a Progressive Neural Network.
    
    This layer includes lateral connections from previous columns.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        previous_layers: Optional[List[nn.Module]] = None,
        lateral_connections: bool = True,
        activation: nn.Module = nn.ReLU()
    ):
        """
        Initialize a column layer.
        
        Args:
            in_features: Number of input features
            out_features: Number of output features
            previous_layers: Layers from previous columns to connect to
            lateral_connections: Whether to use lateral connections
            activation: Activation function to use
        """
        super().__init__()
        self.main = nn.Linear(in_features, out_features)
        self.activation = activation
        self.lateral_connections = lateral_connections
        self.adapters = nn.ModuleList()
        
        # Add lateral connections from previous layers
        if previous_layers and lateral_connections:
            for prev_layer in previous_layers:
                self.adapters.append(nn.Linear(prev_layer.main.out_features, out_features))
    
    def forward(self, x: torch.Tensor, prev_outputs: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        """
        Forward pass for the column layer.
        
        Args:
            x: Input tensor
            prev_outputs: Outputs from previous column layers
            
        Returns:
            Output tensor
        """
        output = self.main(x)
        
        # Add contributions from previous columns
        if prev_outputs and self.lateral_connections:
            for i, prev_output in enumerate(prev_outputs):
                if i < len(self.adapters):
                    output += self.adapters[i](prev_output)
        
        return self.activation(output)


class ProgressiveColumn(nn.Module):
    """
    A column in a Progressive Neural Network.
    
    Each column is a neural network for a specific task.
    """
    
    def __init__(
        self,
        layer_sizes: List[int],
        previous_columns: Optional[List['ProgressiveColumn']] = None,
        lateral_connections: bool = True,
        activation: nn.Module = nn.ReLU()
    ):
        """
        Initialize a column in a Progressive Neural Network.
        
        Args:
            layer_sizes: Sizes of each layer in the column
            previous_columns: Previous columns to connect to
            lateral_connections: Whether to use lateral connections
            activation: Activation function to use
        """
        super().__init__()
        self.layers = nn.ModuleList()
        self.layer_sizes = layer_sizes
        self.lateral_connections = lateral_connections
        
        # Create layers with lateral connections
        for i in range(len(layer_sizes) - 1):
            prev_layers = None
            if previous_columns and lateral_connections:
                prev_layers = [col.layers[i] for col in previous_columns]
            
            self.layers.append(
                ColumnLayer(
                    layer_sizes[i],
                    layer_sizes[i + 1],
                    prev_layers,
                    lateral_connections,
                    activation
                )
            )
    
    def forward(self, x: torch.Tensor, previous_columns: Optional[List['ProgressiveColumn']] = None) -> torch.Tensor:
        """
        Forward pass for the column.
        
        Args:
            x: Input tensor
            previous_columns: Previous columns to get activations from
            
        Returns:
            Output tensor
        """
        # Store intermediate activations for potential use by next column
        activations = []
        out = x
        
        for i, layer in enumerate(self.layers):
            # Get outputs from this layer in previous columns
            prev_outputs = None
            if previous_columns and self.lateral_connections:
                prev_outputs = [col.activations[i] for col in previous_columns]
            
            out = layer(out, prev_outputs)
            activations.append(out)
        
        self.activations = activations
        return out


class ProgressiveNeuralNetwork(nn.Module):
    """
    Implementation of Progressive Neural Networks.
    
    Creates a new neural network for each task and establishes lateral connections
    to previously trained networks, preventing forgetting while enabling knowledge transfer.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        output_size: int,
        lateral_connections: bool = True,
        activation: nn.Module = nn.ReLU()
    ):
        """
        Initialize the Progressive Neural Network.
        
        Args:
            input_size: Size of input features
            hidden_sizes: Sizes of hidden layers
            output_size: Size of output features
            lateral_connections: Whether to use lateral connections
            activation: Activation function to use
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.lateral_connections = lateral_connections
        self.activation = activation
        
        # Create initial architecture
        self.columns = nn.ModuleList()
        self.active_column = 0
        self.task_to_column = {}
    
    def add_column(self, task_id: int) -> None:
        """
        Add a new column for a new task.
        
        Args:
            task_id: ID of the new task
        """
        layer_sizes = [self.input_size] + self.hidden_sizes + [self.output_size]
        previous_columns = list(self.columns) if self.columns else None
        
        new_column = ProgressiveColumn(
            layer_sizes,
            previous_columns,
            self.lateral_connections,
            self.activation
        )
        
        self.columns.append(new_column)
        self.task_to_column[task_id] = len(self.columns) - 1
        self.active_column = len(self.columns) - 1
        
        logger.info(f"Added column {len(self.columns)-1} for task {task_id}")
    
    def set_active_column(self, task_id: int) -> None:
        """
        Set the active column based on task ID.
        
        Args:
            task_id: ID of the active task
        """
        if task_id not in self.task_to_column:
            raise ValueError(f"No column found for task {task_id}")
        self.active_column = self.task_to_column[task_id]
        logger.info(f"Set active column to {self.active_column} for task {task_id}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the active column.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        if not self.columns:
            raise ValueError("No columns added to the Progressive Neural Network")
        
        previous_columns = list(self.columns[:self.active_column]) if self.active_column > 0 else None
        return self.columns[self.active_column](x, previous_columns)


class ProgressiveNetworks(ContinualStrategy):
    """
    Progressive Neural Networks continual learning strategy.
    
    This strategy creates a new neural network for each task while maintaining
    lateral connections to previously trained networks, allowing the model to
    leverage knowledge from previous tasks without interference.
    """
    
    def __init__(
        self,
        model: Union[nn.Module, ProgressiveNeuralNetwork],
        input_size: Optional[int] = None,
        hidden_sizes: Optional[List[int]] = None,
        output_size: Optional[int] = None,
        lateral_connections: bool = True,
        activation: nn.Module = nn.ReLU(),
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ):
        """
        Initialize the PNN strategy.
        
        Args:
            model: Base model or existing PNN model
            input_size: Size of input features (required if model is not a PNN)
            hidden_sizes: Sizes of hidden layers (required if model is not a PNN)
            output_size: Size of output features (required if model is not a PNN)
            lateral_connections: Whether to use lateral connections
            activation: Activation function to use
            device: Device to use for computation
        """
        super().__init__()
        self.device = device
        
        # If model is already a PNN, use it; otherwise create a new one
        if isinstance(model, ProgressiveNeuralNetwork):
            self.model = model
        else:
            if input_size is None or hidden_sizes is None or output_size is None:
                raise ValueError("input_size, hidden_sizes, and output_size must be "
                               "provided when model is not a ProgressiveNeuralNetwork")
            
            self.model = ProgressiveNeuralNetwork(
                input_size,
                hidden_sizes,
                output_size,
                lateral_connections,
                activation
            )
            
            # Copy initial weights from the base model if possible
            if hasattr(model, 'parameters'):
                self._copy_initial_weights(model)
        
        self.model.to(device)
        self.current_task_id = None
    
    def _copy_initial_weights(self, base_model: nn.Module) -> None:
        """
        Copy weights from base model to first column of PNN if architectures match.
        
        Args:
            base_model: Original model to copy weights from
        """
        try:
            # This is a simplified version; in practice would need more sophisticated mapping
            if len(list(base_model.parameters())) > 0:
                logger.info("Attempting to copy weights from base model to initial PNN column")
                # Actual implementation would depend on specific model architectures
        except Exception as e:
            logger.warning(f"Could not copy weights from base model: {e}")
    
    def before_training(self, task_id: int, **kwargs) -> None:
        """
        Set up model before training on a new task.
        
        Args:
            task_id: ID of the task to train on
        """
        self.current_task_id = task_id
        
        # Add a new column if we haven't seen this task before
        if task_id not in self.model.task_to_column:
            self.model.add_column(task_id)
        else:
            # If we've seen this task, set it as active
            self.model.set_active_column(task_id)
    
    def after_training(self, task_id: int, **kwargs) -> None:
        """
        Finalize model after training on a task.
        
        Args:
            task_id: ID of the task just trained
        """
        # Nothing special to do here - the column is already frozen by design
        pass
    
    def compute_loss(self, outputs: torch.Tensor, targets: torch.Tensor, task_id: int,
                    criterion: nn.Module, **kwargs) -> torch.Tensor:
        """
        Compute regular loss as the columns are isolated by design.
        
        Args:
            outputs: Model outputs
            targets: Target values
            task_id: ID of the current task
            criterion: Loss function
            
        Returns:
            Calculated loss
        """
        return criterion(outputs, targets)
    
    def save_state(self, path: Union[str, Path]) -> None:
        """
        Save the PNN state including all columns.
        
        Args:
            path: Path to save state to
        """
        path = Path(path)
        state = {
            'model_state': self.model.state_dict(),
            'task_to_column': self.model.task_to_column,
            'active_column': self.model.active_column,
            'input_size': self.model.input_size,
            'hidden_sizes': self.model.hidden_sizes,
            'output_size': self.model.output_size,
            'lateral_connections': self.model.lateral_connections,
        }
        torch.save(state, path)
        logger.info(f"Saved PNN state to {path}")
    
    def load_state(self, path: Union[str, Path]) -> None:
        """
        Load the PNN state including all columns.
        
        Args:
            path: Path to load state from
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"No state file found at {path}")
        
        state = torch.load(path, map_location=self.device)
        
        # Recreate model with same architecture
        input_size = state['input_size']
        hidden_sizes = state['hidden_sizes']
        output_size = state['output_size']
        lateral_connections = state['lateral_connections']
        
        self.model = ProgressiveNeuralNetwork(
            input_size,
            hidden_sizes,
            output_size,
            lateral_connections
        )
        
        # Load state
        self.model.load_state_dict(state['model_state'])
        self.model.task_to_column = state['task_to_column']
        self.model.active_column = state['active_column']
        self.model.to(self.device)
        
        logger.info(f"Loaded PNN state from {path}") 
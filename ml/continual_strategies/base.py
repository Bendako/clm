import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Union
from abc import ABC, abstractmethod


class ContinualStrategy(ABC):
    """Base interface for continual learning strategies.
    
    All continual learning strategies should inherit from this class
    and implement the required methods.
    """
    
    def __init__(self, model: nn.Module, device: torch.device = torch.device("cpu")):
        """
        Initialize the continual learning strategy.
        
        Args:
            model: The model being trained
            device: Device to use for computations
        """
        self.model = model
        self.device = device
        self.is_initialized = False
        
    @abstractmethod
    def before_training(self, task_id: int):
        """
        Called before training on a task begins.
        
        This method can be used to store task-specific information
        or prepare the strategy for the new task.
        
        Args:
            task_id: ID of the task about to be trained
        """
        pass
    
    @abstractmethod
    def compute_loss(self, loss: torch.Tensor, output: torch.Tensor, target: torch.Tensor, task_id: int) -> torch.Tensor:
        """
        Compute the regularized loss using the strategy.
        
        Args:
            loss: The standard task loss (e.g., cross-entropy)
            output: Output of the model
            target: Target values
            task_id: ID of the current task
            
        Returns:
            Modified loss with regularization
        """
        pass
    
    @abstractmethod
    def after_training(self, task_id: int):
        """
        Called after training on a task is complete.
        
        This method can be used to update task-specific information
        or clean up resources.
        
        Args:
            task_id: ID of the task that was trained
        """
        pass
    
    @abstractmethod
    def state_dict(self) -> Dict[str, Any]:
        """
        Get the state dictionary of the strategy.
        
        This method should return a dictionary containing all 
        information needed to restore the strategy's state.
        
        Returns:
            Dictionary with strategy state
        """
        pass
    
    @abstractmethod
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """
        Load the strategy state from a dictionary.
        
        Args:
            state_dict: Dictionary with strategy state
        """
        pass


class NoStrategy(ContinualStrategy):
    """A dummy strategy that does nothing.
    
    This can be used as a baseline or when no continual learning 
    strategy is needed.
    """
    
    def before_training(self, task_id: int):
        """No action needed before training."""
        pass
    
    def compute_loss(self, loss: torch.Tensor, output: torch.Tensor, target: torch.Tensor, task_id: int) -> torch.Tensor:
        """Return the original loss unchanged."""
        return loss
    
    def after_training(self, task_id: int):
        """No action needed after training."""
        pass
    
    def state_dict(self) -> Dict[str, Any]:
        """Return an empty state dictionary."""
        return {}
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """No state to load."""
        pass 
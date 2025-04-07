"""
PackNet Implementation for Continual Learning

This module implements PackNet, a parameter-efficient continual learning strategy
based on iterative pruning and weight freezing. It allows a single network to learn
multiple tasks without forgetting by "packing" task-specific parameters.

Reference:
Mallya, A., & Lazebnik, S. (2018). 
PackNet: Adding Multiple Tasks to a Single Network by Iterative Pruning.
IEEE Conference on Computer Vision and Pattern Recognition (CVPR).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Set, Optional, Tuple, Union, Any
import copy
import numpy as np
import logging
from pathlib import Path

from ml.continual_strategies.base import ContinualStrategy

logger = logging.getLogger(__name__)


class PackNet(ContinualStrategy):
    """
    PackNet continual learning strategy.
    
    This strategy uses iterative pruning and weight freezing to allow a single network
    to learn multiple tasks without catastrophic forgetting.
    """
    
    def __init__(
        self,
        model: nn.Module,
        prune_percentage: float = 0.75,
        prune_threshold: float = 0.001,
        use_magnitude_pruning: bool = True,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ):
        """
        Initialize the PackNet strategy.
        
        Args:
            model: The model to apply PackNet to
            prune_percentage: Percentage of weights to prune after each task (0.0-1.0)
            prune_threshold: Weights below this threshold are considered prunable
            use_magnitude_pruning: Whether to prune based on weight magnitude (True)
                                  or gradient magnitude (False)
            device: Device to use for computation
        """
        super().__init__(model=model, device=device)
        self.prune_percentage = prune_percentage
        self.prune_threshold = prune_threshold
        self.use_magnitude_pruning = use_magnitude_pruning
        
        # Keep track of task-specific masks and frozen weights
        self.task_masks = {}
        self.current_task_id = None
        self.frozen_weights = set()
        
        # Store gradient magnitudes for gradient-based pruning
        self.gradient_magnitudes = {}
        
        # Keep track of pruning history
        self.pruning_history = {}
        
        # Move model to device
        self.model.to(device)
    
    def before_training(self, task_id: int, **kwargs) -> None:
        """
        Prepare the model for training on a new task.
        
        Args:
            task_id: ID of the task to train on
        """
        self.current_task_id = task_id
        
        # Register hooks for gradient accumulation if using gradient pruning
        if not self.use_magnitude_pruning:
            self._register_gradient_hooks()
            self.gradient_magnitudes = {}
            
        # Create a new mask for this task or use existing one
        if task_id not in self.task_masks:
            # Initialize mask with 1s (all parameters active)
            self.task_masks[task_id] = {}
            
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    # New mask with all 1s (keep all weights)
                    mask = torch.ones_like(param.data, device=self.device)
                    
                    # Apply existing frozen weights mask
                    if name in self.frozen_weights:
                        for prev_task_id in sorted(self.task_masks.keys()):
                            if prev_task_id != task_id:
                                # Apply previous task's mask
                                prev_mask = self.task_masks[prev_task_id].get(name)
                                if prev_mask is not None:
                                    # Zero out parameters that are important for previous tasks
                                    mask = mask * (1 - prev_mask)
                    
                    self.task_masks[task_id][name] = mask
        
        # Apply the mask to the current parameters
        self._apply_mask(task_id)
    
    def _register_gradient_hooks(self) -> None:
        """Register hooks to accumulate gradient magnitudes during training."""
        
        def hook_factory(name):
            def hook(grad):
                if name not in self.gradient_magnitudes:
                    self.gradient_magnitudes[name] = torch.zeros_like(grad, device=self.device)
                # Accumulate absolute gradient values
                self.gradient_magnitudes[name] += torch.abs(grad)
                return grad
            return hook
        
        # Register hooks for each parameter
        self.hooks = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                hook = param.register_hook(hook_factory(name))
                self.hooks.append(hook)
    
    def _remove_gradient_hooks(self) -> None:
        """Remove the gradient accumulation hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def after_training(self, task_id: int, **kwargs) -> None:
        """
        Prune the network after training on a task and freeze important weights.
        
        Args:
            task_id: ID of the task just trained
        """
        # Remove gradient hooks if using gradient-based pruning
        if not self.use_magnitude_pruning:
            self._remove_gradient_hooks()
        
        # Compute masks for pruning
        self._compute_pruning_mask(task_id)
        
        # Apply the masks to freeze important weights
        self._apply_mask(task_id)
        
        # Add to frozen weights set
        for name in self.task_masks[task_id].keys():
            self.frozen_weights.add(name)
            
        # Record pruning statistics
        remaining = self._count_remaining_params()
        total = self._count_total_params()
        self.pruning_history[task_id] = {
            "remaining_params": remaining,
            "total_params": total,
            "sparsity": 1.0 - (remaining / total)
        }
        
        logger.info(f"Task {task_id} pruning complete. Remaining weights: {remaining}/{total} "
                   f"({100.0 * remaining / total:.2f}%)")
    
    def _compute_pruning_mask(self, task_id: int) -> None:
        """
        Compute pruning masks for the current task.
        
        Args:
            task_id: ID of the current task
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.task_masks[task_id]:
                # Get the current mask
                mask = self.task_masks[task_id][name]
                
                # Get values to determine importance
                if self.use_magnitude_pruning:
                    importance = torch.abs(param.data)
                else:
                    importance = self.gradient_magnitudes.get(name, torch.zeros_like(param.data))
                
                # Only consider weights that are currently active in this task's mask
                active_importance = importance * mask
                
                # Determine pruning threshold (keep top X% of weights)
                if torch.sum(mask) > 0:  # If there are any active weights
                    # Get the threshold value
                    k = int((1.0 - self.prune_percentage) * torch.sum(mask).item())
                    
                    if k > 0:
                        # Get the kth largest value
                        threshold = torch.kthvalue(active_importance.flatten(), k).values.item()
                        
                        # Create binary mask for weights to keep (above threshold and active)
                        keep_mask = (active_importance > threshold).float() * mask
                        
                        # Update task mask
                        self.task_masks[task_id][name] = keep_mask
                    else:
                        # No weights to keep - rare edge case
                        logger.warning(f"No weights to keep for parameter {name} in task {task_id}")
    
    def _apply_mask(self, task_id: int) -> None:
        """
        Apply masks to the model parameters for the given task.
        
        Args:
            task_id: ID of the task to apply masks for
        """
        # First zero out all previously frozen weights
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.frozen_weights:
                # Zero out the parameter
                param.data.mul_(0.0)
                
                # Apply all masks from previous tasks
                for prev_task_id in sorted(self.task_masks.keys()):
                    prev_mask = self.task_masks[prev_task_id].get(name)
                    if prev_mask is not None:
                        # Add back the important weights for this previous task
                        param.data.add_(param.data_backup * prev_mask)
        
        # Save a backup of the current parameters
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Store a backup of the original values
                if not hasattr(param, 'data_backup'):
                    param.data_backup = param.data.clone()
                else:
                    param.data_backup.copy_(param.data)
    
    def compute_loss(self, loss: torch.Tensor, output: torch.Tensor, target: torch.Tensor, task_id: int) -> torch.Tensor:
        """
        Compute loss with no modifications (masks are applied directly to parameters).
        
        Args:
            loss: Original loss value
            output: Model output
            target: Target values
            task_id: ID of the current task
            
        Returns:
            Loss value (unchanged for PackNet as weight constraints are applied via masks)
        """
        return loss
    
    def forward(self, x: torch.Tensor, task_id: int) -> torch.Tensor:
        """
        Forward pass using parameters for the specified task.
        
        Args:
            x: Input tensor
            task_id: ID of the task to use parameters for
            
        Returns:
            Model output
        """
        # Apply the correct mask for this task
        self._apply_task_specific_mask(task_id)
        
        # Forward pass through the model
        return self.model(x)
    
    def _apply_task_specific_mask(self, task_id: int) -> None:
        """
        Apply task-specific mask to use only parameters for the specified task.
        
        Args:
            task_id: ID of the task to apply mask for
        """
        if task_id not in self.task_masks:
            raise ValueError(f"No mask found for task {task_id}")
        
        # Restore from backup
        for name, param in self.model.named_parameters():
            if param.requires_grad and hasattr(param, 'data_backup'):
                # Start with zeros
                param.data.zero_()
                
                # Apply task-specific mask
                mask = self.task_masks[task_id].get(name)
                if mask is not None:
                    param.data.add_(param.data_backup * mask)
    
    def _count_remaining_params(self) -> int:
        """
        Count the number of non-zero parameters.
        
        Returns:
            Number of non-zero parameters
        """
        count = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                count += torch.sum(param.data != 0).item()
        return count
    
    def _count_total_params(self) -> int:
        """
        Count the total number of parameters.
        
        Returns:
            Total number of parameters
        """
        count = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                count += param.numel()
        return count
    
    def save_state(self, path: Union[str, Path]) -> None:
        """
        Save the PackNet state.
        
        Args:
            path: Path to save state to
        """
        path = Path(path)
        state = {
            'task_masks': self.task_masks,
            'frozen_weights': list(self.frozen_weights),
            'prune_percentage': self.prune_percentage,
            'prune_threshold': self.prune_threshold,
            'use_magnitude_pruning': self.use_magnitude_pruning,
            'pruning_history': self.pruning_history
        }
        torch.save(state, path)
        logger.info(f"Saved PackNet state to {path}")
    
    def load_state(self, path: Union[str, Path]) -> None:
        """
        Load the PackNet state.
        
        Args:
            path: Path to load state from
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"No state file found at {path}")
        
        state = torch.load(path, map_location=self.device)
        
        self.task_masks = state['task_masks']
        self.frozen_weights = set(state['frozen_weights'])
        self.prune_percentage = state['prune_percentage']
        self.prune_threshold = state['prune_threshold']
        self.use_magnitude_pruning = state['use_magnitude_pruning']
        self.pruning_history = state.get('pruning_history', {})
        
        logger.info(f"Loaded PackNet state from {path}")
    
    def state_dict(self) -> Dict[str, Any]:
        """
        Get the state dict for saving.
        
        Returns:
            State dictionary
        """
        return {
            'task_masks': self.task_masks,
            'frozen_weights': list(self.frozen_weights),
            'prune_percentage': self.prune_percentage,
            'prune_threshold': self.prune_threshold,
            'use_magnitude_pruning': self.use_magnitude_pruning,
            'pruning_history': self.pruning_history
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Load from a state dict.
        
        Args:
            state_dict: State dictionary to load from
        """
        self.task_masks = state_dict['task_masks']
        self.frozen_weights = set(state_dict['frozen_weights'])
        self.prune_percentage = state_dict['prune_percentage']
        self.prune_threshold = state_dict['prune_threshold']
        self.use_magnitude_pruning = state_dict['use_magnitude_pruning']
        self.pruning_history = state_dict.get('pruning_history', {}) 
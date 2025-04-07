"""
Experience Replay with Regularization (ER+) Implementation for Continual Learning

This module implements ER+, which combines replay memory with regularization to prevent
catastrophic forgetting. It balances current task performance with stability on previous tasks
through a simple, effective approach.

References:
- Buzzega, P., Boschini, M., Porrello, A., Abati, D., & Calderara, S. (2020).
  Dark Experience for General Continual Learning: a Strong, Simple Baseline.
  Advances in Neural Information Processing Systems.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import logging
from collections import defaultdict

from ml.continual_strategies.base import ContinualStrategy
from ml.replay_buffers.reservoir_sampling import TensorReservoirBuffer

logger = logging.getLogger(__name__)


class ERPlus(ContinualStrategy):
    """
    Experience Replay with Regularization (ER+) strategy for continual learning.
    
    This strategy combines experience replay with knowledge distillation regularization
    to preserve performance on previous tasks while learning new ones.
    """
    
    def __init__(
        self,
        model: nn.Module,
        memory_size: int = 500,
        batch_size: int = 32,
        reg_weight: float = 1.0,
        temperature: float = 2.0,
        task_balanced: bool = True,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ):
        """
        Initialize ER+ strategy.
        
        Args:
            model: Neural network model
            memory_size: Size of the replay buffer
            batch_size: Batch size for sampling from replay buffer
            reg_weight: Weight for the regularization/distillation loss
            temperature: Temperature for softening logits in distillation loss
            task_balanced: Whether to maintain balanced samples per task in buffer
            device: Device to use for computation
        """
        super().__init__(model=model, device=device)
        
        self.replay_buffer = TensorReservoirBuffer(
            capacity=memory_size,
            task_balanced=task_balanced,
            device=device
        )
        
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.reg_weight = reg_weight
        self.temperature = temperature
        
        # Dictionary to store model predictions for replay samples
        self.stored_predictions = {}
        
        # Current task info
        self.current_task_id = None
        
    def before_training(self, task_id: int) -> None:
        """
        Prepare the model for training on a new task.
        
        Args:
            task_id: ID of the task to train on
        """
        self.current_task_id = task_id
        
        # If this isn't the first task, store predictions for samples in replay buffer
        # using the current model (before training on the new task)
        if task_id > 0 and len(self.replay_buffer) > 0:
            self._store_predictions()
    
    def _store_predictions(self) -> None:
        """
        Store the current model's predictions on replay buffer samples.
        These will be used for regularization during training.
        """
        self.model.eval()
        self.stored_predictions = {}
        
        # Process each task's samples in replay buffer
        for task_id in self.replay_buffer.tasks:
            task_samples = self.replay_buffer.get_all_task_samples(task_id)
            
            if task_samples:
                inputs, targets = task_samples
                
                with torch.no_grad():
                    # Get model predictions for the samples
                    outputs = self.model(inputs)
                    
                    # Soften the outputs using temperature
                    if isinstance(outputs, tuple):
                        logits = outputs[0]
                    else:
                        logits = outputs
                    
                    soft_targets = F.softmax(logits / self.temperature, dim=-1)
                    
                    # Store softened predictions
                    self.stored_predictions[task_id] = (inputs, targets, soft_targets)
        
        self.model.train()
    
    def after_backward(self, task_id: int) -> None:
        """
        Apply any modifications to gradients after backward pass.
        Not needed for ER+, but included for completeness.
        
        Args:
            task_id: ID of the current task
        """
        pass
    
    def compute_loss(self, loss: torch.Tensor, output: torch.Tensor, target: torch.Tensor, task_id: int) -> torch.Tensor:
        """
        Compute regularized loss using replay buffer and stored predictions.
        
        Args:
            loss: Original loss on current task
            output: Model's output on current batch
            target: Target labels for current batch
            task_id: ID of the current task
            
        Returns:
            Regularized loss incorporating replay and distillation
        """
        # If this is the first task or if replay buffer is empty, just return the original loss
        if task_id == 0 or len(self.replay_buffer) == 0:
            return loss
        
        # Sample from the replay buffer
        replay_data = self.replay_buffer.sample(self.batch_size)
        
        if not replay_data:
            return loss
        
        replay_inputs, replay_targets = replay_data
        
        # Forward pass with replay data
        replay_outputs = self.model(replay_inputs)
        
        if isinstance(replay_outputs, tuple):
            replay_logits = replay_outputs[0]
        else:
            replay_logits = replay_outputs
        
        # Task loss on replay samples
        replay_loss = F.cross_entropy(replay_logits.view(-1, replay_logits.size(-1)), 
                                    replay_targets.view(-1))
        
        # Distillation loss on replay samples
        distill_loss = torch.tensor(0.0, device=self.device)
        
        for task_id, (_, _, soft_targets) in self.stored_predictions.items():
            # Skip if we don't have stored predictions for this task
            if soft_targets is None:
                continue
                
            # Get samples for this specific task
            task_samples = self.replay_buffer.get_all_task_samples(task_id)
            
            if not task_samples:
                continue
                
            inputs, _ = task_samples
            
            # Get current model predictions
            with torch.no_grad():
                outputs = self.model(inputs)
                
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs
                
                # Soften with temperature
                current_soft = F.softmax(logits / self.temperature, dim=-1)
                
                # Compute KL divergence loss
                task_distill_loss = F.kl_div(
                    F.log_softmax(current_soft, dim=-1),
                    soft_targets,
                    reduction='batchmean'
                ) * (self.temperature ** 2)
                
                distill_loss += task_distill_loss
        
        # Combine losses: original loss + replay loss + distillation regularization
        total_loss = loss + replay_loss + self.reg_weight * distill_loss
        
        return total_loss
    
    def after_training(self, task_id: int) -> None:
        """
        Update replay buffer after training on a task.
        
        Args:
            task_id: ID of the task just trained
        """
        self.current_task_id = None
    
    def update_replay_buffer(self, task_id: int, inputs: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Update the replay buffer with new samples.
        
        Args:
            task_id: ID of the current task
            inputs: Input samples to add to buffer
            targets: Target labels to add to buffer
        """
        self.replay_buffer.add(task_id, (inputs.detach(), targets.detach()))
    
    def state_dict(self) -> Dict[str, Any]:
        """
        Get state dictionary for saving.
        
        Returns:
            Dictionary with strategy state
        """
        return {
            'replay_buffer': self.replay_buffer.state_dict(),
            'stored_predictions': self.stored_predictions,
            'memory_size': self.memory_size,
            'batch_size': self.batch_size,
            'reg_weight': self.reg_weight,
            'temperature': self.temperature,
            'current_task_id': self.current_task_id
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Load strategy state from dictionary.
        
        Args:
            state_dict: Dictionary with strategy state
        """
        self.replay_buffer.load_state_dict(state_dict['replay_buffer'])
        self.stored_predictions = state_dict['stored_predictions']
        self.memory_size = state_dict['memory_size']
        self.batch_size = state_dict['batch_size']
        self.reg_weight = state_dict['reg_weight']
        self.temperature = state_dict['temperature']
        self.current_task_id = state_dict['current_task_id'] 
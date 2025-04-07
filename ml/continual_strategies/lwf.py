import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Union, Tuple
import copy

from ml.continual_strategies.base import ContinualStrategy


class LearningWithoutForgetting(ContinualStrategy):
    """Learning without Forgetting (LwF) continual learning strategy.
    
    LwF uses knowledge distillation to preserve knowledge of previous tasks
    while learning new ones. It stores outputs of the model on the new task
    data using the old model parameters, and uses these as soft targets.
    
    Reference:
        Li & Hoiem, "Learning without Forgetting"
        https://arxiv.org/abs/1606.09282
    """
    
    def __init__(
        self, 
        model: nn.Module, 
        alpha: float = 1.0,
        temperature: float = 2.0,
        device: torch.device = torch.device("cpu")
    ):
        """
        Initialize LwF strategy.
        
        Args:
            model: Model being trained
            alpha: Weight for the distillation loss (larger = more importance on previous tasks)
            temperature: Temperature for softening probability distributions (larger = softer)
            device: Device to use for computations
        """
        super().__init__(model, device)
        self.alpha = alpha
        self.temperature = temperature
        
        # Previous model (copy of parameters before training on new task)
        self.previous_model = None
        
        # Dictionary of previous model outputs for each task
        self.previous_model_outputs = {}
        
        # Track if we've saved the previous model
        self.has_previous_model = False
    
    def before_training(self, task_id: int):
        """Save a copy of the model before training on the new task."""
        # Only store previous model if this isn't the first task
        if task_id > 0:
            self.previous_model = copy.deepcopy(self.model)
            self.previous_model.eval()  # Set to evaluation mode
            self.has_previous_model = True
        else:
            self.has_previous_model = False
        
        self.is_initialized = True
    
    def compute_loss(self, loss: torch.Tensor, output: torch.Tensor, target: torch.Tensor, task_id: int) -> torch.Tensor:
        """
        Add LwF distillation loss to the standard task loss.
        
        For each batch, we compute the outputs of the previous model
        and use these as soft targets for a distillation loss.
        
        Args:
            loss: Standard task loss (e.g., cross-entropy)
            output: Output of the current model
            target: Target values
            task_id: ID of the current task
            
        Returns:
            Modified loss with distillation term
        """
        if task_id == 0 or not self.has_previous_model:
            # First task, no distillation needed
            return loss
        
        # Get input data from the output computation
        # This is a bit of a hack - would be better to pass inputs directly,
        # but adapting to the interface
        with torch.no_grad():
            if hasattr(self.model, 'get_last_input') and callable(self.model.get_last_input):
                # If model tracks inputs, use those
                inputs = self.model.get_last_input()
            else:
                # Otherwise we can't compute distillation loss
                return loss
            
            # Get outputs from previous model
            self.previous_model.eval()
            prev_outputs = self.previous_model(inputs)
            
            if isinstance(prev_outputs, tuple):
                prev_outputs = prev_outputs[0]  # Some models return multiple outputs
        
        # Compute knowledge distillation loss
        distillation_loss = self._compute_distillation_loss(output, prev_outputs)
        
        # Combined loss = standard loss + alpha * distillation loss
        combined_loss = loss + self.alpha * distillation_loss
        
        return combined_loss
    
    def _compute_distillation_loss(self, current_output: torch.Tensor, previous_output: torch.Tensor) -> torch.Tensor:
        """
        Compute the knowledge distillation loss between current and previous outputs.
        
        Args:
            current_output: Output from the current model
            previous_output: Output from the previous model (soft targets)
            
        Returns:
            Distillation loss
        """
        # Apply temperature scaling
        current_logits = current_output / self.temperature
        previous_logits = previous_output / self.temperature
        
        # Convert to probabilities (softmax with temperature)
        current_probs = F.softmax(current_logits, dim=-1)
        previous_probs = F.softmax(previous_logits, dim=-1)
        
        # Compute KL divergence loss
        # We use KL div directly between prob distributions, scale by temp²
        kl_loss = F.kl_div(
            F.log_softmax(current_logits, dim=-1),
            previous_probs,
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        return kl_loss
    
    def after_training(self, task_id: int):
        """Clean up resources after training."""
        # Clear the previous model to save memory
        self.previous_model = None
        self.has_previous_model = False
    
    def state_dict(self) -> Dict[str, Any]:
        """
        Get the state dictionary of the strategy.
        
        Returns:
            Dictionary with strategy state
        """
        return {
            'alpha': self.alpha,
            'temperature': self.temperature,
            'is_initialized': self.is_initialized,
            'has_previous_model': self.has_previous_model
            # Note: we don't save the previous_model in state_dict
            # as it would be redundant with the main model checkpoint
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """
        Load the strategy state from a dictionary.
        
        Args:
            state_dict: Dictionary with strategy state
        """
        self.alpha = state_dict.get('alpha', self.alpha)
        self.temperature = state_dict.get('temperature', self.temperature)
        self.is_initialized = state_dict.get('is_initialized', False)
        self.has_previous_model = state_dict.get('has_previous_model', False) 
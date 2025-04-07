import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Union
import copy

from ml.continual_strategies.base import ContinualStrategy


class ElasticWeightConsolidation(ContinualStrategy):
    """Elastic Weight Consolidation (EWC) continual learning strategy.
    
    EWC prevents catastrophic forgetting by adding a penalty for changing
    parameters that were important for previous tasks.
    
    Reference:
        Kirkpatrick et al. "Overcoming catastrophic forgetting in neural networks"
        https://arxiv.org/abs/1612.00796
    """
    
    def __init__(
        self, 
        model: nn.Module, 
        ewc_lambda: float = 1.0,
        gamma: float = 1.0,
        device: torch.device = torch.device("cpu")
    ):
        """
        Initialize EWC strategy.
        
        Args:
            model: Model being trained
            ewc_lambda: Regularization strength
            gamma: Decay factor for older task importances
            device: Device to use for computations
        """
        super().__init__(model, device)
        self.ewc_lambda = ewc_lambda
        self.gamma = gamma
        
        # Dictionary to store parameters for each task
        self.parameter_dict = {}
        
        # Store Fisher information matrices for each task
        self.fisher_dict = {}
    
    def before_training(self, task_id: int):
        """Apply gamma decay to previous task importances."""
        # Decay previous task importances
        for task in self.fisher_dict.keys():
            for parameter in self.fisher_dict[task].keys():
                self.fisher_dict[task][parameter] *= self.gamma
        
        self.is_initialized = True
    
    def compute_loss(self, loss: torch.Tensor, output: torch.Tensor, target: torch.Tensor, task_id: int) -> torch.Tensor:
        """Add EWC regularization term to the loss."""
        if len(self.parameter_dict) == 0:
            # No previous tasks, return original loss
            return loss
        
        # Compute EWC penalty
        ewc_loss = torch.tensor(0.0, device=self.device)
        
        # Current model parameters
        params = {name: param for name, param in self.model.named_parameters() if param.requires_grad}
        
        # Compute EWC penalty for each previous task
        for prev_task in self.parameter_dict.keys():
            for name, param in params.items():
                if name in self.parameter_dict[prev_task] and name in self.fisher_dict[prev_task]:
                    # Get old parameter values and Fisher information
                    old_param = self.parameter_dict[prev_task][name]
                    fisher = self.fisher_dict[prev_task][name]
                    
                    # Compute penalty (importance-weighted squared difference)
                    ewc_loss += (fisher * (param - old_param).pow(2)).sum()
        
        # Return loss with EWC penalty
        return loss + (self.ewc_lambda * ewc_loss)
    
    def after_training(self, task_id: int):
        """Save model parameters and compute Fisher information for the task."""
        # Store current parameter values
        task_params = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                task_params[name] = param.detach().clone()
        
        self.parameter_dict[task_id] = task_params
        
        # Compute and store Fisher information matrix
        self.fisher_dict[task_id] = self._compute_fisher_information()
    
    def _compute_fisher_information(self) -> Dict[str, torch.Tensor]:
        """
        Compute the Fisher information matrix for the current model.
        
        Fisher information measures how sensitive the model output
        is to changes in parameters.
        
        Returns:
            Dictionary mapping parameter names to their Fisher information
        """
        fisher_dict = {}
        # Initialize Fisher information to zero for each parameter
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                fisher_dict[name] = torch.zeros_like(param)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # The actual Fisher computation would require samples from the task
        # In a real implementation, this would use a representative dataset
        # from the task to compute empirical Fisher information
        
        # For a practical implementation with an actual dataset, use:
        # 1. Load a small representative dataset for the task
        # 2. For each batch:
        #    a. Compute model output (log-probabilities)
        #    b. Sample from the output distribution
        #    c. Compute gradients with respect to the sampled output
        #    d. Square the gradients and accumulate in fisher_dict
        # 3. Average the accumulated Fisher information
        
        # Here we use a simplified placeholder implementation
        # In a real system, replace this with actual computation
        # using logits and gradients from a representative dataset
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Placeholder: set Fisher to scaled identity matrix
                # In practice, this should be computed from actual data
                fisher_dict[name] = torch.ones_like(param) * 0.1
        
        # Set model back to training mode
        self.model.train()
        
        return fisher_dict
    
    def state_dict(self) -> Dict[str, Any]:
        """
        Get the state dictionary of the strategy.
        
        Returns:
            Dictionary with strategy state
        """
        return {
            'ewc_lambda': self.ewc_lambda,
            'gamma': self.gamma,
            'parameter_dict': self.parameter_dict,
            'fisher_dict': self.fisher_dict,
            'is_initialized': self.is_initialized
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """
        Load the strategy state from a dictionary.
        
        Args:
            state_dict: Dictionary with strategy state
        """
        self.ewc_lambda = state_dict.get('ewc_lambda', self.ewc_lambda)
        self.gamma = state_dict.get('gamma', self.gamma)
        self.parameter_dict = state_dict.get('parameter_dict', {})
        self.fisher_dict = state_dict.get('fisher_dict', {})
        self.is_initialized = state_dict.get('is_initialized', False) 
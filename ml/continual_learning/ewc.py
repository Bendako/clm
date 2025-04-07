"""
Elastic Weight Consolidation (EWC) Implementation

EWC is a regularization-based continual learning technique that constrains
important parameters to stay close to their old values when learning new tasks.
This helps prevent catastrophic forgetting in neural networks.

Reference:
Kirkpatrick, J., Pascanu, R., Rabinowitz, N., Veness, J., Desjardins, G., Rusu, A. A., ... & Hadsell, R. (2017).
Overcoming catastrophic forgetting in neural networks. Proceedings of the National Academy of Sciences.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

class EWC:
    """
    Elastic Weight Consolidation (EWC) implementation for preventing catastrophic forgetting.
    """
    
    def __init__(
        self, 
        model: nn.Module, 
        importance: float = 1000.0,
        fisher_sample_size: int = 200,
        fisher_approximation: str = "diagonal"  # "diagonal" or "kfac"
    ):
        """
        Initialize EWC.
        
        Args:
            model: The PyTorch model to apply EWC to
            importance: Hyperparameter controlling the importance of previous tasks
            fisher_sample_size: Number of samples to use for Fisher calculation
            fisher_approximation: Method to approximate the Fisher Information Matrix
        """
        self.model = model
        self.importance = importance
        self.fisher_sample_size = fisher_sample_size
        self.fisher_approximation = fisher_approximation
        
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self.saved_params = {}
        self.fisher_matrices = {}
        self.current_task = None
        
    def register_task(self, task_id: str, dataloader: torch.utils.data.DataLoader, loss_fn: callable):
        """
        Register a task by computing and storing the Fisher Information Matrix.
        
        Args:
            task_id: Identifier for the task
            dataloader: DataLoader containing samples from the task
            loss_fn: Loss function used for the task
        """
        logger.info(f"Registering task {task_id} with EWC")
        
        # Save current parameter values
        self.saved_params[task_id] = {n: p.clone().detach() for n, p in self.params.items()}
        
        # Compute Fisher Information Matrix
        fisher = {n: torch.zeros_like(p) for n, p in self.params.items()}
        
        self.model.eval()
        for i, batch in enumerate(dataloader):
            if i >= self.fisher_sample_size:
                break
                
            inputs, _ = batch if isinstance(batch, tuple) and len(batch) == 2 else (batch, None)
            
            self.model.zero_grad()
            outputs = self.model(inputs)
            
            # For classification, we use the log probabilities
            log_probs = F.log_softmax(outputs, dim=1)
            # For generation, we might use the loss directly
            
            # Sample from the model's predictions
            if log_probs.size(1) > 0:  # Classification case
                samples = torch.multinomial(torch.exp(log_probs), 1).squeeze()
                loss = F.nll_loss(log_probs, samples)
            else:  # If not classification, use provided loss
                loss = loss_fn(outputs, inputs)  # Example: reconstruction loss
                
            loss.backward()
            
            # Accumulate squared gradients for Fisher
            for n, p in self.params.items():
                if p.grad is not None:
                    fisher[n] += p.grad.detach() ** 2 / self.fisher_sample_size
        
        self.fisher_matrices[task_id] = fisher
        self.current_task = task_id
        
        logger.info(f"Task {task_id} registered successfully")
        
    def ewc_loss(self) -> torch.Tensor:
        """
        Compute the EWC penalty term to be added to the loss function.
        
        Returns:
            The EWC penalty term (quadratic distance from previous params weighted by Fisher)
        """
        if not self.saved_params:
            # No previous tasks registered
            return torch.tensor(0., device=next(self.model.parameters()).device)
            
        loss = torch.tensor(0., device=next(self.model.parameters()).device)
        
        for task_id, fisher in self.fisher_matrices.items():
            # Skip current task
            if task_id == self.current_task:
                continue
                
            for n, p in self.params.items():
                # Compute the squared difference between current and saved params
                # weighted by the Fisher Information
                loss += (fisher[n] * (p - self.saved_params[task_id][n]) ** 2).sum()
                
        return self.importance * loss / 2.0
    
    def apply_ewc_training(self, loss_fn: callable) -> callable:
        """
        Wraps a loss function to include the EWC penalty.
        
        Args:
            loss_fn: Original loss function
            
        Returns:
            Wrapped loss function with EWC penalty
        """
        def ewc_loss_fn(*args, **kwargs):
            # Compute original loss
            original_loss = loss_fn(*args, **kwargs)
            
            # Add EWC penalty
            ewc_penalty = self.ewc_loss()
            
            # Return combined loss
            return original_loss + ewc_penalty
            
        return ewc_loss_fn 
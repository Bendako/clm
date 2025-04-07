import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Union, Tuple
import numpy as np
import copy
import quadprog

from ml.continual_strategies.base import ContinualStrategy
from ml.replay_buffers.reservoir_sampling import TensorReservoirBuffer


class GradientEpisodicMemory(ContinualStrategy):
    """Gradient Episodic Memory (GEM) continual learning strategy.
    
    GEM uses episodic memory to store a subset of samples from
    previous tasks and constrains gradient updates to prevent
    increasing loss on these examples.
    
    Reference:
        Lopez-Paz & Ranzato, "Gradient Episodic Memory for Continual Learning"
        https://arxiv.org/abs/1706.08840
    """
    
    def __init__(
        self, 
        model: nn.Module, 
        memory_size: int = 200,
        samples_per_task: int = 50,
        margin: float = 0.5,
        device: torch.device = torch.device("cpu")
    ):
        """
        Initialize GEM strategy.
        
        Args:
            model: Model being trained
            memory_size: Total size of episodic memory
            samples_per_task: Number of samples to store per task
            margin: Margin parameter to avoid unnecessary constraint enforcement
            device: Device to use for computations
        """
        super().__init__(model, device)
        self.memory_size = memory_size
        self.samples_per_task = samples_per_task
        self.margin = margin
        
        # Memory buffers for each task
        self.memories = {}
        
        # Keep track of gradients of previous tasks
        self.grads = {}
        
        # Store which parameters are trainable for gradient manipulation
        self.params = [p for p in model.parameters() if p.requires_grad]
    
    def before_training(self, task_id: int):
        """Prepare memory buffer for the new task."""
        if task_id == 0:
            # First task, nothing to do
            pass
        
        self.is_initialized = True
    
    def compute_loss(self, loss: torch.Tensor, output: torch.Tensor, target: torch.Tensor, task_id: int) -> torch.Tensor:
        """
        Return original loss, but modify gradients during backward pass.
        
        GEM doesn't change the loss function but modifies gradients during backprop
        to ensure they don't increase loss on previous tasks.
        
        Args:
            loss: Standard task loss (e.g., cross-entropy)
            output: Output of the current model
            target: Target values
            task_id: ID of the current task
            
        Returns:
            Original loss (gradient modifications happen elsewhere)
        """
        # Just return the original loss, gradient constraints are applied in after_backward
        return loss
    
    def _project_gradient(self, gradient_src, gradient_ref_list):
        """
        Project the gradient so that it doesn't increase loss on previous tasks.
        
        Uses quadratic programming to find the nearest gradient that doesn't
        increase loss on previous tasks.
        
        Args:
            gradient_src: Current gradient
            gradient_ref_list: List of reference gradients from memory
            
        Returns:
            Projected gradient
        """
        # Flatten gradient vectors
        t = len(gradient_ref_list)
        gradient_src_flat = torch.cat([g.view(-1) for g in gradient_src])
        
        # If no reference gradients, return original
        if t == 0:
            return gradient_src
        
        # Create matrix of reference gradients
        gradient_ref_flat = torch.zeros((t, gradient_src_flat.size(0)), device=self.device)
        for i, g in enumerate(gradient_ref_list):
            gradient_ref_flat[i] = torch.cat([gi.view(-1) for gi in g])
        
        # Check if any constraints are violated
        # If current gradient has negative dot product with any reference,
        # it would increase loss on that task
        dotprods = torch.matmul(gradient_ref_flat, gradient_src_flat)
        if (dotprods < 0).sum() == 0:
            # No constraints violated, return original gradient
            return gradient_src
        
        # Use quadratic programming to find nearest gradient
        # that doesn't violate constraints
        gradient_ref_np = gradient_ref_flat.cpu().numpy()
        gradient_src_np = gradient_src_flat.cpu().numpy()
        
        # Solve QP problem
        # Minimize 0.5*x^T*P*x + q^T*x subject to Gx >= h
        
        # P matrix: identity (minimize squared distance)
        P = np.eye(gradient_src_np.shape[0])
        
        # q vector: negative source gradient (maximize dot product)
        q = -gradient_src_np
        
        # G matrix: reference gradients (ensure dot product >= 0)
        G = gradient_ref_np
        
        # h vector: zeros (ensure dot product >= 0)
        h = np.zeros(t) - self.margin  # Apply margin for stability
        
        try:
            # Solve QP problem
            qp_solution = quadprog.solve_qp(P, q, -G, h)[0]
            
            # Convert solution back to PyTorch tensor
            qp_solution = torch.from_numpy(qp_solution).to(self.device).float()
            
            # Reshape gradient
            start_idx = 0
            projected_gradient = []
            for g in gradient_src:
                param_size = g.numel()
                projected_gradient.append(qp_solution[start_idx:start_idx+param_size].view_as(g))
                start_idx += param_size
                
            return projected_gradient
            
        except Exception as e:
            print(f"QP solver failed: {e}")
            return gradient_src
    
    def after_backward(self, task_id: int):
        """
        Modify gradients to respect constraints from previous tasks.
        
        This method should be called after loss.backward() but before optimizer.step()
        """
        if task_id == 0:
            # First task, nothing to do
            return
        
        # Collect current gradients
        current_gradients = [p.grad.clone() for p in self.params]
        
        # Collect gradients from previous tasks
        prev_gradients = []
        for prev_task in range(task_id):
            if prev_task in self.memories and len(self.memories[prev_task]) > 0:
                # Compute gradients for memory samples
                task_grads = self._compute_memory_gradients(prev_task)
                prev_gradients.append(task_grads)
        
        # Project current gradients
        projected_gradients = self._project_gradient(current_gradients, prev_gradients)
        
        # Apply projected gradients
        for p, g in zip(self.params, projected_gradients):
            p.grad.copy_(g)
    
    def _compute_memory_gradients(self, task_id: int) -> List[torch.Tensor]:
        """
        Compute gradients on memory samples for a task.
        
        Args:
            task_id: ID of the task
            
        Returns:
            List of gradients for each parameter
        """
        if task_id not in self.memories or len(self.memories[task_id]) == 0:
            return []
        
        # Get memory samples
        memory = self.memories[task_id]
        
        # Compute loss on memory samples
        self.model.zero_grad()
        
        # Here we'd process the memory data and compute loss
        # This is a placeholder for the actual implementation
        # which would require inputs, targets, and loss function
        inputs, targets = memory['inputs'], memory['targets']
        
        # If inputs/targets are on CPU, move to device
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        
        # Forward pass
        outputs = self.model(inputs)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
            
        # Compute loss and gradients
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(outputs, targets)
        loss.backward()
        
        # Get gradients
        gradients = [p.grad.clone() for p in self.params]
        
        # Clean up
        self.model.zero_grad()
        
        return gradients
    
    def after_training(self, task_id: int):
        """Store samples from current task in memory."""
        # In a real implementation, you would:
        # 1. Select random samples from current task data
        # 2. Store them in the memory buffer
        
        # This is a placeholder, assuming samples are provided elsewhere
        # In practice, integrate with the actual dataset/dataloader
        
        # For demonstration purposes:
        if hasattr(self.model, 'get_task_samples') and callable(self.model.get_task_samples):
            inputs, targets = self.model.get_task_samples(task_id, self.samples_per_task)
            self.memories[task_id] = {
                'inputs': inputs.detach().clone(),
                'targets': targets.detach().clone()
            }
    
    def state_dict(self) -> Dict[str, Any]:
        """
        Get the state dictionary of the strategy.
        
        Returns:
            Dictionary with strategy state
        """
        return {
            'memory_size': self.memory_size,
            'samples_per_task': self.samples_per_task,
            'margin': self.margin,
            'memories': self.memories,
            'is_initialized': self.is_initialized
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """
        Load the strategy state from a dictionary.
        
        Args:
            state_dict: Dictionary with strategy state
        """
        self.memory_size = state_dict.get('memory_size', self.memory_size)
        self.samples_per_task = state_dict.get('samples_per_task', self.samples_per_task)
        self.margin = state_dict.get('margin', self.margin)
        self.memories = state_dict.get('memories', {})
        self.is_initialized = state_dict.get('is_initialized', False) 
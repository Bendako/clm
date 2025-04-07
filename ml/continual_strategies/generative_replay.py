import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
import numpy as np

from ml.continual_strategies.base import ContinualStrategy
from ml.replay_buffers.reservoir_sampling import ReservoirBuffer, TensorReservoirBuffer

logger = logging.getLogger(__name__)

class GenerativeReplay(ContinualStrategy):
    """
    Generative Replay strategy for continual learning.
    
    This strategy uses a generative model (e.g., VAE, GAN) to generate synthetic
    samples from previous tasks, which are then used during training on new tasks
    to preserve knowledge of past tasks.
    
    References:
        - Shin, H., et al. (2017). Continual learning with deep generative replay.
          Advances in Neural Information Processing Systems.
    """
    
    def __init__(
        self,
        model: nn.Module,
        generator: nn.Module,
        memory_size: int = 500,
        batch_size: int = 32,
        replay_weight: float = 1.0,
        generator_update_freq: int = 5,
        task_balanced: bool = True,
        device: torch.device = torch.device("cpu")
    ):
        """
        Initialize the Generative Replay strategy.
        
        Args:
            model: The main model being trained
            generator: A generative model (VAE/GAN) that can produce synthetic samples
            memory_size: Maximum number of samples to store in the replay buffer
            batch_size: Batch size for sampling from replay buffer
            replay_weight: Weight for the replay loss
            generator_update_freq: How often to update the generator (in iterations)
            task_balanced: Whether to balance samples across tasks
            device: Device to use for computation
        """
        super().__init__(model, device)
        self.generator = generator.to(device)
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.replay_weight = replay_weight
        self.generator_update_freq = generator_update_freq
        self.task_balanced = task_balanced
        
        # Initialize replay buffer for storing real samples to train generator
        self.replay_buffer = TensorReservoirBuffer(
            capacity=memory_size,
            task_balanced=task_balanced
        )
        
        # Generator optimizer
        self.generator_optimizer = torch.optim.Adam(self.generator.parameters(), lr=0.001)
        
        # Track current task
        self.current_task_id = None
        self.iteration = 0
        
        # Store tasks seen so far
        self.seen_tasks = set()
        
        logger.info(f"Initialized GenerativeReplay with memory_size={memory_size}, replay_weight={replay_weight}")
        
    def before_training(self, task_id: int) -> None:
        """
        Prepare for training on a new task.
        
        Args:
            task_id: ID of the task about to be trained
        """
        self.current_task_id = task_id
        self.seen_tasks.add(task_id)
        logger.info(f"GenerativeReplay: Starting training on task {task_id}")
        
    def _update_generator(self, inputs: torch.Tensor, targets: Optional[torch.Tensor] = None) -> float:
        """
        Update the generator using samples from the replay buffer.
        
        Args:
            inputs: Batch of input samples
            targets: Batch of target samples (optional)
            
        Returns:
            Generator loss value
        """
        self.generator_optimizer.zero_grad()
        
        # The exact loss depends on the generator architecture
        # For VAE:
        reconstructed, mu, logvar = self.generator(inputs)
        
        # Reconstruction loss
        recon_loss = F.mse_loss(reconstructed, inputs)
        
        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Total loss
        gen_loss = recon_loss + 0.1 * kl_loss
        
        gen_loss.backward()
        self.generator_optimizer.step()
        
        return gen_loss.item()
    
    def _generate_samples(self, task_id: int, num_samples: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Generate synthetic samples for a given task.
        
        Args:
            task_id: ID of the task to generate samples for
            num_samples: Number of samples to generate
            
        Returns:
            Tuple of (inputs, targets)
        """
        # Set generator to eval mode
        self.generator.eval()
        
        with torch.no_grad():
            # Generate latent vector
            z = torch.randn(num_samples, self.generator.latent_dim).to(self.device)
            
            # Generate samples
            generated_samples = self.generator.decode(z)
            
            # For classification tasks, we would also need to generate labels
            # This could be done with a separate model or using the main model
            self.model.eval()
            outputs = self.model(generated_samples)
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs
                
            # Get pseudo-labels using argmax
            pseudo_labels = torch.argmax(logits, dim=-1)
            
        # Reset models to training mode
        self.model.train()
        self.generator.train()
        
        return generated_samples, pseudo_labels
        
    def compute_loss(
        self, 
        loss: torch.Tensor, 
        output: torch.Tensor, 
        target: torch.Tensor, 
        task_id: int
    ) -> torch.Tensor:
        """
        Compute the regularized loss using generative replay.
        
        Args:
            loss: The standard task loss
            output: Output of the model
            target: Target values
            task_id: ID of the current task
            
        Returns:
            Modified loss with regularization
        """
        self.iteration += 1
        
        # Update the generator periodically
        if self.iteration % self.generator_update_freq == 0 and len(self.replay_buffer) > 0:
            # Sample from replay buffer to update generator
            replay_inputs, replay_targets = self.replay_buffer.sample(min(self.batch_size, len(self.replay_buffer)))
            replay_inputs = replay_inputs.to(self.device)
            
            gen_loss = self._update_generator(replay_inputs)
            logger.debug(f"Generator update: loss={gen_loss:.4f}")
        
        # If we have seen previous tasks, perform generative replay
        replay_loss = torch.tensor(0.0, device=self.device)
        if task_id > 0 and len(self.seen_tasks) > 1:
            # Generate samples from previous tasks
            total_replay_samples = 0
            
            # We'll collect losses for each previous task
            for prev_task_id in self.seen_tasks:
                if prev_task_id == task_id:
                    continue
                
                # Generate synthetic samples for this task
                generated_inputs, generated_targets = self._generate_samples(
                    prev_task_id, 
                    self.batch_size // max(1, len(self.seen_tasks) - 1)
                )
                
                if generated_inputs is None or generated_targets is None:
                    continue
                
                # Forward pass with generated samples
                generated_outputs = self.model(generated_inputs)
                
                if isinstance(generated_outputs, tuple):
                    generated_logits = generated_outputs[0]
                else:
                    generated_logits = generated_outputs
                
                # Compute loss on generated samples
                task_replay_loss = F.cross_entropy(
                    generated_logits.view(-1, generated_logits.size(-1)), 
                    generated_targets.view(-1)
                )
                
                replay_loss += task_replay_loss
                total_replay_samples += 1
            
            # Average the replay loss across tasks
            if total_replay_samples > 0:
                replay_loss = replay_loss / total_replay_samples
                
        # Combine losses: original loss + replay loss
        total_loss = loss + self.replay_weight * replay_loss
        
        return total_loss
    
    def after_backward(self, task_id: int) -> None:
        """
        Perform operations after backward pass (not used in this strategy).
        
        Args:
            task_id: ID of the current task
        """
        pass
    
    def after_training(self, task_id: int) -> None:
        """
        Update replay buffer after training on a task.
        
        Args:
            task_id: ID of the task just trained
        """
        self.current_task_id = None
        logger.info(f"GenerativeReplay: Completed training on task {task_id}") 
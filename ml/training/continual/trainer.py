"""
Continual Learning Trainer Module

This module implements a trainer for continual learning of language models,
integrating multiple anti-forgetting strategies like Elastic Weight Consolidation,
Learning without Forgetting, and Gradient Episodic Memory to prevent 
catastrophic forgetting.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import logging
import time
import mlflow
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import numpy as np
from tqdm import tqdm
import json
from pathlib import Path

# Import CLM components
from ml.continual_strategies import (
    ContinualStrategy, 
    NoStrategy,
    ElasticWeightConsolidation, 
    LearningWithoutForgetting,
    GradientEpisodicMemory
)
from ml.replay_buffers.reservoir_sampling import ReservoirBuffer, TensorReservoirBuffer

logger = logging.getLogger(__name__)

class ContinualTrainer:
    """
    Trainer for continual learning of language models.
    
    This trainer integrates multiple strategies to prevent catastrophic
    forgetting while training on a sequence of tasks.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        experiment_name: Optional[str] = None,
        checkpoint_dir: Optional[str] = None
    ):
        """
        Initialize the continual learning trainer.
        
        Args:
            model: The language model to train
            config: Configuration dictionary with training settings
            device: Device to run training on
            experiment_name: Name for MLflow experiment tracking
            checkpoint_dir: Directory to save model checkpoints
        """
        self.model = model
        self.config = config
        self.device = device
        self.experiment_name = experiment_name or "continual_learning"
        self.checkpoint_dir = checkpoint_dir or "./checkpoints"
        
        # Ensure checkpoint directory exists
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Move model to device
        self.model.to(self.device)
        
        # Initialize optimizer and scheduler
        self.init_optimizer()
        
        # Set up anti-forgetting strategies based on config
        self.setup_continual_strategies()
        
        # Track training history
        self.training_history = {
            "tasks": [],
            "metrics": {}
        }
        
        # Set up MLflow tracking
        self._setup_mlflow()
        
        logger.info(f"Initialized ContinualTrainer on {self.device}")
        
    def init_optimizer(self):
        """Initialize optimizer and learning rate scheduler based on config."""
        # Get optimizer parameters from config
        optim_config = self.config.get("optimizer", {"type": "adamw", "lr": 5e-5})
        if isinstance(optim_config, str):
            # Backward compatibility for string optimizer config
            optim_type = optim_config
            lr = self.config.get("learning_rate", 5e-5)
            weight_decay = self.config.get("weight_decay", 0.01)
        else:
            # Dictionary config
            optim_type = optim_config.get("type", "adamw")
            lr = optim_config.get("lr", 5e-5)
            weight_decay = optim_config.get("weight_decay", 0.01)
        
        # Set up optimizer based on type
        if optim_type.lower() == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optim_type.lower() == "adamw":
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optim_type.lower() == "sgd":
            momentum = self.config.get("momentum", 0.9)
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {optim_type}")
            
        # Set up scheduler if specified
        scheduler_config = self.config.get("scheduler", None)
        self.scheduler = None
        
        if scheduler_config:
            if isinstance(scheduler_config, str):
                # Backward compatibility for string scheduler config
                scheduler_type = scheduler_config
                scheduler_params = {}
            else:
                # Dictionary config
                scheduler_type = scheduler_config.get("type", "linear")
                scheduler_params = {k: v for k, v in scheduler_config.items() if k != "type"}
            
            # Create scheduler based on type
            if scheduler_type.lower() == "linear":
                total_steps = self.config.get("total_steps", 1000)
                warmup_steps = self.config.get("warmup_steps", 100)
                self.scheduler = transformers.get_linear_schedule_with_warmup(
                    self.optimizer,
                    num_warmup_steps=warmup_steps,
                    num_training_steps=total_steps,
                    **scheduler_params
                )
            elif scheduler_type.lower() == "cosine":
                total_steps = self.config.get("total_steps", 1000)
                warmup_steps = self.config.get("warmup_steps", 100)
                self.scheduler = transformers.get_cosine_schedule_with_warmup(
                    self.optimizer,
                    num_warmup_steps=warmup_steps,
                    num_training_steps=total_steps,
                    **scheduler_params
                )
            elif scheduler_type.lower() == "step":
                step_size = scheduler_params.get("step_size", 30)
                gamma = scheduler_params.get("gamma", 0.1)
                self.scheduler = optim.lr_scheduler.StepLR(
                    self.optimizer,
                    step_size=step_size,
                    gamma=gamma
                )
            else:
                logger.warning(f"Unsupported scheduler type: {scheduler_type}")
                
    def setup_continual_strategies(self):
        """Set up continual learning strategies based on configuration."""
        self.strategy = None
        self.replay_buffer = None
        
        # Get strategy configuration
        strategy_config = self.config.get("continual_strategy", {})
        
        # Setup EWC if enabled
        ewc_config = strategy_config.get("ewc", {"enabled": False})
        lwf_config = strategy_config.get("lwf", {"enabled": False})
        gem_config = strategy_config.get("gem", {"enabled": False})
        
        # Choose strategy (only one active at a time)
        if ewc_config.get("enabled", False):
            # Extract EWC parameters
            ewc_lambda = ewc_config.get("lambda", 1.0)
            gamma = ewc_config.get("gamma", 0.9)
            
            # Create EWC strategy
            logger.info(f"Enabling EWC with lambda={ewc_lambda}, gamma={gamma}")
            self.strategy = ElasticWeightConsolidation(
                model=self.model,
                ewc_lambda=ewc_lambda,
                gamma=gamma,
                device=self.device
            )
        elif lwf_config.get("enabled", False):
            # Extract LwF parameters
            alpha = lwf_config.get("alpha", 1.0)
            temperature = lwf_config.get("temperature", 2.0)
            
            # Create LwF strategy
            logger.info(f"Enabling LwF with alpha={alpha}, temperature={temperature}")
            self.strategy = LearningWithoutForgetting(
                model=self.model,
                alpha=alpha,
                temperature=temperature,
                device=self.device
            )
        elif gem_config.get("enabled", False):
            # Extract GEM parameters
            memory_size = gem_config.get("memory_size", 200)
            samples_per_task = gem_config.get("samples_per_task", 50)
            margin = gem_config.get("margin", 0.5)
            
            # Create GEM strategy
            logger.info(f"Enabling GEM with memory_size={memory_size}")
            self.strategy = GradientEpisodicMemory(
                model=self.model,
                memory_size=memory_size,
                samples_per_task=samples_per_task,
                margin=margin,
                device=self.device
            )
        else:
            # No strategy selected
            logger.info("No continual learning strategy enabled")
            self.strategy = NoStrategy(model=self.model, device=self.device)
            
        # Setup replay buffer if enabled
        replay_config = strategy_config.get("replay", {"enabled": False})
        if replay_config.get("enabled", False):
            # Extract replay buffer parameters
            buffer_size = replay_config.get("buffer_size", 1000)
            task_balanced = replay_config.get("task_balanced", True)
            
            # Create replay buffer
            logger.info(f"Enabling replay buffer with size={buffer_size}, task_balanced={task_balanced}")
            self.replay_buffer = TensorReservoirBuffer(
                capacity=buffer_size,
                task_balanced=task_balanced,
                device=self.device
            )
        
    def _setup_mlflow(self):
        """Set up MLflow experiment tracking."""
        try:
            # Set MLflow experiment
            mlflow.set_experiment(self.experiment_name)
        except Exception as e:
            logger.warning(f"Failed to set up MLflow: {e}")
        
    def train_task(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        task_name: str = None,
        task_id: int = None,
        loss_fn: Optional[Callable] = None,
        num_epochs: Optional[int] = None,
        callbacks: List[Callable] = None
    ) -> Dict[str, Any]:
        """
        Train the model on a specific task.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: Optional DataLoader for validation data
            task_name: Name of the task
            task_id: ID of the task (integer)
            loss_fn: Loss function (if None, use model's built-in loss)
            num_epochs: Number of epochs (overrides config if provided)
            callbacks: List of callback functions to call after each epoch
            
        Returns:
            Dictionary with training metrics
        """
        # Ensure we have a task ID and name
        if task_id is None and task_name is None:
            task_id = len(self.training_history["tasks"])
            task_name = f"task_{task_id}"
        elif task_id is None:
            task_id = len(self.training_history["tasks"])
        elif task_name is None:
            task_name = f"task_{task_id}"
            
        logger.info(f"Starting training for task: {task_name} (ID: {task_id})")
        
        # Call strategy before_training hook
        self.strategy.before_training(task_id)
        
        # Track this task
        if task_name not in self.training_history["tasks"]:
            self.training_history["tasks"].append(task_name)
            self.training_history["metrics"][task_name] = []
        
        # Set training parameters
        epochs = num_epochs if num_epochs is not None else self.config.get("epochs", 3)
        
        # Set default loss function if none provided
        if loss_fn is None:
            loss_fn = nn.CrossEntropyLoss()
        
        # Start MLflow run
        with mlflow.start_run(run_name=f"task_{task_name}"):
            # Log parameters
            mlflow.log_params({
                "task_name": task_name,
                "task_id": task_id,
                "epochs": epochs,
                "batch_size": train_loader.batch_size,
                "optimizer": self.config.get("optimizer", {}).get("type", "adamw"),
                "learning_rate": self.config.get("optimizer", {}).get("lr", 5e-5),
                "model_type": type(self.model).__name__
            })
            
            epoch_metrics = []
            
            # Training loop
            for epoch in range(epochs):
                # Train for one epoch
                metrics = self._train_epoch(
                    epoch, train_loader, task_id, task_name, loss_fn)
                
                # Validation if provided
                if val_loader:
                    val_metrics = self._validate(val_loader, task_id, task_name, loss_fn)
                    metrics.update(val_metrics)
                    
                # Update training history
                self.training_history["metrics"][task_name].append(metrics)
                epoch_metrics.append(metrics)
                
                # Save checkpoint
                self._save_checkpoint(task_name, epoch)
                
                # Log metrics to MLflow
                mlflow.log_metrics(metrics, step=epoch)
                
                # Run callbacks if provided
                if callbacks:
                    for callback in callbacks:
                        callback(self, task_name, epoch, metrics)
                        
                logger.info(f"Epoch {epoch+1}/{epochs} - "
                           f"train_loss: {metrics['train_loss']:.4f}" + 
                           (f" - val_loss: {metrics['val_loss']:.4f}" if val_loader else ""))
            
            # Call strategy after_training hook
            self.strategy.after_training(task_id)
            
            # Save final model for this task
            self._save_checkpoint(task_name, epochs, is_final=True)
        
        # Return the metrics from all epochs
        return {
            "task_name": task_name,
            "task_id": task_id,
            "epochs": epoch_metrics
        }
                
    def _train_epoch(
        self,
        epoch: int,
        dataloader: DataLoader,
        task_id: int,
        task_name: str,
        loss_fn: Callable
    ) -> Dict[str, float]:
        """
        Train for one epoch on the given dataloader.
        
        Args:
            epoch: Current epoch number
            dataloader: DataLoader for training data
            task_id: ID of the current task
            task_name: Name of the current task
            loss_fn: Loss function to use
            
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        # Progress bar
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(pbar):
            # Handle different batch formats
            if isinstance(batch, tuple) and len(batch) >= 2:
                inputs, targets = batch[0], batch[1]
                if isinstance(inputs, torch.Tensor):
                    inputs = inputs.to(self.device)
                if isinstance(targets, torch.Tensor):
                    targets = targets.to(self.device)
            elif isinstance(batch, dict):
                # Handle dictionary batch (like HuggingFace)
                inputs = batch
                for k in inputs:
                    if isinstance(inputs[k], torch.Tensor):
                        inputs[k] = inputs[k].to(self.device)
                # Default target is input_ids shifted right for language models
                targets = inputs.get("labels", None)
            else:
                # Assume the model knows how to handle the batch
                inputs = batch
                if isinstance(inputs, torch.Tensor):
                    inputs = inputs.to(self.device)
                targets = None
            
            # Clear gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(inputs)
            
            # Compute standard loss
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs
                
            if targets is not None:
                standard_loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
            else:
                # For models that return loss
                standard_loss = outputs.loss if hasattr(outputs, "loss") else outputs
                
            # Apply continual learning strategy
            loss = self.strategy.compute_loss(standard_loss, logits, targets, task_id)
            
            # Backward pass
            loss.backward()
            
            # Apply any gradient modifications needed (like for GEM)
            if hasattr(self.strategy, "after_backward"):
                self.strategy.after_backward(task_id)
            
            # Update parameters
            self.optimizer.step()
            
            # Update replay buffer if enabled
            if self.replay_buffer is not None:
                if isinstance(batch, tuple) and len(batch) >= 2:
                    self.replay_buffer.add(task_id, (inputs.detach(), targets.detach()))
                elif isinstance(batch, dict) and "input_ids" in batch and "labels" in batch:
                    self.replay_buffer.add(task_id, (batch["input_ids"].detach(), batch["labels"].detach()))
            
            # Update metrics
            total_loss += loss.item()
            
            # Calculate accuracy if possible
            if targets is not None and logits.size(-1) > 1:
                # For classification tasks
                if logits.dim() > 2:
                    # Sequence models like language models
                    pred = logits.argmax(dim=-1)
                    correct += (pred == targets).sum().item()
                else:
                    # Standard classification
                    pred = logits.argmax(dim=1)
                    correct += (pred == targets).sum().item()
                total += targets.numel()
            
            # Update scheduler if available
            if self.scheduler:
                self.scheduler.step()
            
            # Update progress bar
            pbar.set_postfix({"loss": loss.item()})
        
        # Calculate metrics
        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total if total > 0 else 0.0
        
        return {
            "train_loss": avg_loss,
            "train_accuracy": accuracy,
            "epoch": epoch
        }
    
    def _validate(
        self,
        dataloader: DataLoader,
        task_id: int,
        task_name: str,
        loss_fn: Callable
    ) -> Dict[str, float]:
        """
        Validate the model on the given dataloader.
        
        Args:
            dataloader: DataLoader for validation data
            task_id: ID of the current task
            task_name: Name of the current task
            loss_fn: Loss function to use
            
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in dataloader:
                # Handle different batch formats (same as in train_epoch)
                if isinstance(batch, tuple) and len(batch) >= 2:
                    inputs, targets = batch[0], batch[1]
                    if isinstance(inputs, torch.Tensor):
                        inputs = inputs.to(self.device)
                    if isinstance(targets, torch.Tensor):
                        targets = targets.to(self.device)
                elif isinstance(batch, dict):
                    # Handle dictionary batch (like HuggingFace)
                    inputs = batch
                    for k in inputs:
                        if isinstance(inputs[k], torch.Tensor):
                            inputs[k] = inputs[k].to(self.device)
                    # Default target is input_ids shifted right for language models
                    targets = inputs.get("labels", None)
                else:
                    # Assume the model knows how to handle the batch
                    inputs = batch
                    if isinstance(inputs, torch.Tensor):
                        inputs = inputs.to(self.device)
                    targets = None
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Compute loss
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs
                    
                if targets is not None:
                    loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
                else:
                    # For models that return loss
                    loss = outputs.loss if hasattr(outputs, "loss") else outputs
                
                # Update metrics
                total_loss += loss.item()
                
                # Calculate accuracy if possible
                if targets is not None and logits.size(-1) > 1:
                    # For classification tasks
                    if logits.dim() > 2:
                        # Sequence models like language models
                        pred = logits.argmax(dim=-1)
                        correct += (pred == targets).sum().item()
                    else:
                        # Standard classification
                        pred = logits.argmax(dim=1)
                        correct += (pred == targets).sum().item()
                    total += targets.numel()
        
        # Calculate metrics
        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total if total > 0 else 0.0
        
        return {
            "val_loss": avg_loss,
            "val_accuracy": accuracy
        }
    
    def replay_training(
        self,
        task_id: int,
        loss_fn: Callable,
        batch_size: int = 16
    ) -> Optional[Dict[str, float]]:
        """
        Train on samples from the replay buffer.
        
        Args:
            task_id: ID of the current task
            loss_fn: Loss function to use
            batch_size: Batch size for replay buffer samples
            
        Returns:
            Dictionary with replay metrics, if available
        """
        if self.replay_buffer is None or len(self.replay_buffer) == 0:
            return None
        
        self.model.train()
        
        # Sample from replay buffer
        samples = self.replay_buffer.sample(batch_size)
        if not samples:
            return None
        
        # Process samples (format depends on what was stored)
        if isinstance(samples, tuple) and len(samples) == 2:
            inputs, targets = samples
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
        else:
            return None
        
        # Forward pass
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        
        # Compute loss
        if isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs
            
        standard_loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        # Apply continual learning strategy (with task_id=-1 to indicate replay)
        loss = self.strategy.compute_loss(standard_loss, logits, targets, -1)
        
        # Backward pass and optimize
        loss.backward()
        self.optimizer.step()
        
        return {
            "replay_loss": loss.item()
        }
    
    def _save_checkpoint(self, task_name: str, epoch: int, is_final: bool = False):
        """
        Save model and optimizer state.
        
        Args:
            task_name: Name of the current task
            epoch: Current epoch number
            is_final: Whether this is the final checkpoint for the task
        """
        prefix = "final" if is_final else f"epoch_{epoch}"
        checkpoint_path = os.path.join(
            self.checkpoint_dir, f"{task_name}_{prefix}.pt")
        
        # Save model, optimizer, and strategy states
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "task_name": task_name,
            "epoch": epoch,
            "is_final": is_final,
            "strategy_state_dict": self.strategy.state_dict() if self.strategy else {}
        }
        
        # Save replay buffer if available
        if self.replay_buffer is not None:
            checkpoint["replay_buffer_state"] = self.replay_buffer.state_dict()
        
        # Save training history
        checkpoint["training_history"] = self.training_history
        
        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Log artifact to MLflow
        try:
            mlflow.log_artifact(checkpoint_path)
        except Exception as e:
            logger.warning(f"Failed to log artifact to MLflow: {e}")
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Load model and optimizer state from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Dictionary with checkpoint information
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint["model_state_dict"])
        
        # Load optimizer state if available
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        # Load strategy state if available
        if "strategy_state_dict" in checkpoint and self.strategy:
            self.strategy.load_state_dict(checkpoint["strategy_state_dict"])
        
        # Load replay buffer if available
        if "replay_buffer_state" in checkpoint and self.replay_buffer:
            self.replay_buffer.load_state_dict(checkpoint["replay_buffer_state"])
        
        # Load training history if available
        if "training_history" in checkpoint:
            self.training_history = checkpoint["training_history"]
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        
        return {
            "task_name": checkpoint.get("task_name", "unknown"),
            "epoch": checkpoint.get("epoch", 0),
            "is_final": checkpoint.get("is_final", False)
        }

    def evaluate_all_tasks(
        self,
        task_dataloaders: Dict[str, DataLoader],
        loss_fn: Optional[Callable] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate the model on all previously seen tasks.
        
        Args:
            task_dataloaders: Dict mapping task IDs to validation DataLoaders
            loss_fn: Loss function to use
            
        Returns:
            Dictionary with metrics for each task
        """
        results = {}
        
        # Evaluate each task
        for task_id, dataloader in task_dataloaders.items():
            logger.info(f"Evaluating task: {task_id}")
            metrics = self._validate(dataloader, task_id, task_id, loss_fn)
            results[task_id] = metrics
            
            logger.info(f"Task {task_id} - val_loss: {metrics['val_loss']:.4f}")
            
        return results
        
    def compute_forgetting(
        self,
        task_dataloaders: Dict[str, DataLoader],
        loss_fn: Optional[Callable] = None,
        reference_metrics: Optional[Dict[str, Dict[str, float]]] = None
    ) -> Dict[str, float]:
        """
        Compute forgetting metrics for all tasks.
        
        Args:
            task_dataloaders: Dict mapping task IDs to validation DataLoaders
            loss_fn: Loss function to use
            reference_metrics: Dict with metrics when tasks were first learned
            
        Returns:
            Dictionary with forgetting metrics
        """
        # First evaluate current performance
        current_metrics = self.evaluate_all_tasks(task_dataloaders, loss_fn)
        
        # If no reference metrics provided, return current metrics
        if reference_metrics is None:
            return {
                task_id: {"val_loss": metrics["val_loss"]}
                for task_id, metrics in current_metrics.items()
            }
            
        # Compute forgetting as difference in performance
        forgetting = {}
        
        for task_id, current in current_metrics.items():
            if task_id in reference_metrics:
                # Compute forgetting as increase in loss
                loss_diff = current["val_loss"] - reference_metrics[task_id]["val_loss"]
                forgetting[task_id] = loss_diff
                
        return forgetting 
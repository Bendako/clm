#!/usr/bin/env python
"""
Example script demonstrating how to use the ER+ (Experience Replay with Regularization)
strategy for continual learning.

This script trains a simple MLP model on a sequence of tasks (permuted MNIST),
using ER+ to mitigate catastrophic forgetting.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import yaml
import argparse
import logging
import os
from pathlib import Path
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

# Import CLM components
from ml.training.continual.trainer import ContinualTrainer
from ml.models.simple import SimpleMLP
from ml.utils.data import create_permuted_mnist_tasks

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train with ER+")
    parser.add_argument(
        "--config", type=str, default="configs/continual_learning/er_plus_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--num-tasks", type=int, default=5,
        help="Number of tasks to train on"
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default="checkpoints/er_plus",
        help="Directory to save checkpoints"
    )
    return parser.parse_args()

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def create_model(config):
    """Create model based on configuration."""
    model_config = config.get('model', {})
    model_type = model_config.get('type', 'mlp')
    
    if model_type == 'mlp':
        hidden_sizes = model_config.get('hidden_sizes', [256, 128])
        dropout = model_config.get('dropout', 0.2)
        
        model = SimpleMLP(
            input_size=28*28,
            hidden_sizes=hidden_sizes,
            output_size=10,
            dropout=dropout
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    return model

def plot_results(accuracies, forgetting, title="ER+ Performance"):
    """Plot accuracy and forgetting metrics."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracies
    for task_id, task_accs in enumerate(accuracies):
        ax1.plot(range(1, len(task_accs) + 1), task_accs, 
                 marker='o', label=f'Task {task_id+1}')
    
    ax1.set_xlabel('Tasks Encountered')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Accuracy Throughout Training')
    ax1.legend()
    ax1.grid(True)
    
    # Plot forgetting
    ax2.plot(range(2, len(forgetting) + 2), forgetting, marker='o')
    ax2.set_xlabel('Tasks Encountered')
    ax2.set_ylabel('Average Forgetting')
    ax2.set_title('Forgetting Throughout Training')
    ax2.grid(True)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    # Save figure
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/er_plus_performance.png')
    logger.info("Results plot saved to results/er_plus_performance.png")
    
    plt.show()

def main():
    """Main function for training with ER+."""
    args = parse_args()
    config = load_config(args.config)
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create model
    model = create_model(config)
    model.to(device)
    logger.info(f"Created model: {type(model).__name__}")
    
    # Initialize trainer
    trainer = ContinualTrainer(
        model=model,
        config=config,
        device=device,
        experiment_name=config.get('experiment', {}).get('name', 'er_plus_experiment'),
        checkpoint_dir=args.checkpoint_dir
    )
    logger.info("Initialized ContinualTrainer with ER+ strategy")
    
    # Create tasks (permuted MNIST)
    tasks, task_names = create_permuted_mnist_tasks(
        num_tasks=args.num_tasks,
        batch_size=config.get('batch_size', 64),
        train_split=0.8
    )
    logger.info(f"Created {len(tasks)} permuted MNIST tasks")
    
    # Training
    accuracies = [[] for _ in range(args.num_tasks)]
    forgetting = []
    
    # Train on each task
    for task_id, (task_name, (train_loader, val_loader)) in enumerate(zip(task_names, tasks)):
        logger.info(f"Training on {task_name} (Task {task_id+1}/{args.num_tasks})")
        
        # Train on current task
        trainer.train_task(
            train_loader=train_loader,
            val_loader=val_loader,
            task_name=task_name,
            task_id=task_id
        )
        
        # Evaluate on all previous tasks
        for prev_task_id, (prev_task_name, (_, prev_val_loader)) in enumerate(zip(task_names[:task_id+1], tasks[:task_id+1])):
            metrics = trainer._validate(
                dataloader=prev_val_loader,
                task_id=prev_task_id,
                task_name=prev_task_name,
                loss_fn=nn.CrossEntropyLoss()
            )
            
            accuracy = metrics.get('val_accuracy', metrics.get('accuracy', 0.0))
            accuracies[prev_task_id].append(accuracy)
            
            logger.info(f"Task {task_id+1} - Accuracy on {prev_task_name}: {accuracy:.4f}")
        
        # Calculate forgetting
        if task_id > 0:
            # Average forgetting of previous tasks
            forget = 0.0
            for prev_task_id in range(task_id):
                # Maximum previous accuracy - current accuracy
                max_prev_acc = max(accuracies[prev_task_id][:-1])
                curr_acc = accuracies[prev_task_id][-1]
                forget += max(0, max_prev_acc - curr_acc)
            
            forget /= task_id  # Average over previous tasks
            forgetting.append(forget)
            logger.info(f"Task {task_id+1} - Average Forgetting: {forget:.4f}")
    
    # Plot results
    plot_results(accuracies, forgetting)
    
    logger.info("Training complete!")
    return accuracies, forgetting

if __name__ == "__main__":
    main() 
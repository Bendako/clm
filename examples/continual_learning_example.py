#!/usr/bin/env python
"""
Example of using the CLM (Continual Learning for Models) framework
with a toy language model trained on multiple tasks sequentially.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# Import components from our CLM framework
from ml.training.continual import ContinualTrainer
from ml.evaluation import (
    LanguageModelEvaluator,
    ContinualLearningVisualizer,
    ContinualLearningMetrics
)

# Create a simple toy model for demonstration
class ToyLanguageModel(nn.Module):
    def __init__(self, vocab_size: int = 1000, embedding_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.lstm(embedded)
        return self.fc(output)

# Function to create synthetic data for multiple tasks
def create_synthetic_tasks(
    num_tasks: int = 3,
    samples_per_task: int = 1000,
    seq_length: int = 20,
    vocab_size: int = 1000
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Create synthetic language modeling tasks.
    
    Each task has a different pattern:
    - Task 1: Sequences where next word is (current_word + 1) % vocab_size
    - Task 2: Sequences where next word is (current_word + 2) % vocab_size
    - Task 3: Sequences where next word is (vocab_size - current_word) % vocab_size
    
    Returns:
        Dictionary mapping task names to (input_data, target_data) tuples
    """
    tasks = {}
    
    for task_id in range(num_tasks):
        task_name = f"task_{task_id+1}"
        
        # Create a pattern for this task
        inputs = torch.randint(0, vocab_size, (samples_per_task, seq_length))
        targets = torch.zeros_like(inputs)
        
        # Apply different patterns for different tasks
        if task_id == 0:
            # Task 1: next word = (current_word + 1) % vocab_size
            targets[:, :-1] = (inputs[:, 1:] + 1) % vocab_size
            targets[:, -1] = (inputs[:, 0] + 1) % vocab_size
        elif task_id == 1:
            # Task 2: next word = (current_word + 2) % vocab_size
            targets[:, :-1] = (inputs[:, 1:] + 2) % vocab_size
            targets[:, -1] = (inputs[:, 0] + 2) % vocab_size
        else:
            # Task 3: next word = (vocab_size - current_word) % vocab_size
            targets[:, :-1] = (vocab_size - inputs[:, 1:]) % vocab_size
            targets[:, -1] = (vocab_size - inputs[:, 0]) % vocab_size
        
        tasks[task_name] = (inputs, targets)
    
    return tasks

def create_dataloaders(
    tasks: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    batch_size: int = 32,
    train_ratio: float = 0.8
) -> Tuple[Dict[str, DataLoader], Dict[str, DataLoader]]:
    """
    Create train and validation DataLoaders for each task.
    
    Args:
        tasks: Dictionary mapping task names to (input_data, target_data) tuples
        batch_size: Batch size for DataLoaders
        train_ratio: Ratio of data to use for training
        
    Returns:
        Tuple of (train_loaders, val_loaders) dictionaries
    """
    train_loaders = {}
    val_loaders = {}
    
    for task_name, (inputs, targets) in tasks.items():
        # Split data into train and validation sets
        num_samples = inputs.shape[0]
        num_train = int(train_ratio * num_samples)
        
        # Create datasets
        train_dataset = TensorDataset(inputs[:num_train], targets[:num_train])
        val_dataset = TensorDataset(inputs[num_train:], targets[num_train:])
        
        # Create data loaders
        train_loaders[task_name] = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loaders[task_name] = DataLoader(val_dataset, batch_size=batch_size)
    
    return train_loaders, val_loaders

def main():
    # Parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = 1000
    embedding_dim = 64
    hidden_dim = 128
    batch_size = 32
    learning_rate = 1e-3
    num_epochs_per_task = 3
    results_dir = "results/continual_learning_example"
    
    # Create synthetic tasks
    print("Creating synthetic tasks...")
    tasks = create_synthetic_tasks(num_tasks=3, vocab_size=vocab_size)
    train_loaders, val_loaders = create_dataloaders(tasks, batch_size=batch_size)
    
    # Create output directory
    os.makedirs(results_dir, exist_ok=True)
    
    # Create model
    model = ToyLanguageModel(vocab_size=vocab_size, embedding_dim=embedding_dim, hidden_dim=hidden_dim)
    model.to(device)
    
    # Create loss function
    criterion = nn.CrossEntropyLoss()
    
    # Create continual learning trainer
    trainer_config = {
        "optimizer": {
            "type": "adam",
            "lr": learning_rate
        },
        "continual_strategy": {
            "ewc": {
                "enabled": True,
                "lambda": 1.0,
                "gamma": 0.9
            },
            "replay": {
                "enabled": True,
                "buffer_size": 500,
                "task_balanced": True
            }
        },
        "checkpoint_dir": os.path.join(results_dir, "checkpoints")
    }
    
    trainer = ContinualTrainer(
        model=model,
        config=trainer_config,
        device=device,
        experiment_name="toy_language_model"
    )
    
    # Create evaluator
    evaluator = LanguageModelEvaluator(device=device, log_to_mlflow=False)
    
    # Create visualizer
    visualizer = ContinualLearningVisualizer(save_dir=os.path.join(results_dir, "plots"))
    
    # Train on each task sequentially
    task_names = list(tasks.keys())
    
    for task_idx, task_name in enumerate(task_names):
        print(f"\n=== Training on {task_name} ===")
        
        # Train on current task
        trainer.train_task(
            train_loader=train_loaders[task_name],
            val_loader=val_loaders[task_name],
            task_name=task_name,
            task_id=task_idx,
            num_epochs=num_epochs_per_task
        )
        
        # Evaluate on all tasks seen so far
        print(f"\n=== Evaluating on all tasks after training on {task_name} ===")
        seen_tasks = task_names[:task_idx+1]
        seen_val_loaders = {tn: val_loaders[tn] for tn in seen_tasks}
        
        results = evaluator.evaluate_all_tasks(
            model=model,
            dataloaders=seen_val_loaders,
            current_task_idx=task_idx,
            criterion=criterion
        )
        
        for tn, metrics in results.items():
            print(f"Task: {tn}, Loss: {metrics['loss']:.4f}, Accuracy: {metrics['accuracy']:.4f}, Perplexity: {metrics['perplexity']:.2f}")
    
    # Create visualizations
    print("\n=== Creating visualizations ===")
    cl_metrics = evaluator.tracker.cl_metrics
    
    # Plot accuracy matrix
    visualizer.plot_accuracy_matrix(cl_metrics)
    
    # Plot forgetting measures
    visualizer.plot_forgetting(cl_metrics)
    
    # Plot task accuracies over time
    visualizer.plot_task_accuracies(cl_metrics)
    
    # Plot transfer measures
    visualizer.plot_transfer(cl_metrics)
    
    # Create summary dashboard
    visualizer.create_summary_dashboard(cl_metrics)
    
    print(f"\nResults saved to {results_dir}")

if __name__ == "__main__":
    main() 
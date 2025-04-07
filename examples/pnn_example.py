#!/usr/bin/env python
"""
Example demonstrating Progressive Neural Networks (PNN) for continual learning.

This example shows how PNN creates a separate network for each task while
establishing lateral connections to previously trained networks, completely
preventing catastrophic forgetting while enabling forward transfer of knowledge.
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
from ml.continual_strategies import ProgressiveNetworks, ProgressiveNeuralNetwork
from ml.continual_strategies.ewc import ElasticWeightConsolidation

# Create a simple base model for comparison
class BaseLanguageModel(nn.Module):
    def __init__(self, vocab_size: int = 1000, embedding_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.lstm(embedded)
        return self.fc(output)

# Function to create synthetic data for multiple tasks (same as in the original example)
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

def compare_strategies(tasks, train_loaders, val_loaders, results_dir, device, num_epochs=3):
    """Compare Progressive Neural Networks with EWC and naive fine-tuning."""
    vocab_size = 1000
    embedding_dim = 64
    hidden_dim = 128
    batch_size = 32
    learning_rate = 1e-3
    
    strategies = {
        "pnn": {
            "name": "Progressive Neural Networks",
            "config": {
                "optimizer": {"type": "adam", "lr": learning_rate},
                "continual_strategy": {
                    "pnn": {
                        "enabled": True,
                        "input_size": embedding_dim,  # After embedding
                        "hidden_sizes": [hidden_dim],
                        "output_size": vocab_size
                    },
                    "replay": {"enabled": False}
                },
                "checkpoint_dir": os.path.join(results_dir, "checkpoints/pnn")
            }
        },
        "ewc": {
            "name": "Elastic Weight Consolidation",
            "config": {
                "optimizer": {"type": "adam", "lr": learning_rate},
                "continual_strategy": {
                    "ewc": {
                        "enabled": True,
                        "lambda": 100.0,  # Higher lambda for stronger regularization
                        "gamma": 0.9
                    },
                    "replay": {"enabled": False}
                },
                "checkpoint_dir": os.path.join(results_dir, "checkpoints/ewc")
            }
        },
        "naive": {
            "name": "Naive Fine-tuning",
            "config": {
                "optimizer": {"type": "adam", "lr": learning_rate},
                "continual_strategy": {
                    "replay": {"enabled": False}
                },
                "checkpoint_dir": os.path.join(results_dir, "checkpoints/naive")
            }
        }
    }
    
    results = {}
    
    # Train each strategy
    for strategy_key, strategy_info in strategies.items():
        print(f"\n\n=== Training with {strategy_info['name']} ===")
        
        # Create model
        model = BaseLanguageModel(vocab_size=vocab_size, embedding_dim=embedding_dim, hidden_dim=hidden_dim)
        model.to(device)
        
        # Create trainer
        trainer = ContinualTrainer(
            model=model,
            config=strategy_info["config"],
            device=device,
            experiment_name=f"toy_language_model_{strategy_key}"
        )
        
        # Create evaluator and metrics
        evaluator = LanguageModelEvaluator(device=device, log_to_mlflow=False)
        metrics = ContinualLearningMetrics()
        
        # Create visualizer
        visualizer = ContinualLearningVisualizer(
            save_dir=os.path.join(results_dir, f"plots/{strategy_key}")
        )
        
        # Train on each task sequentially
        task_names = list(tasks.keys())
        
        for task_idx, task_name in enumerate(task_names):
            print(f"\n=== Training on {task_name} with {strategy_info['name']} ===")
            
            # Train on current task
            trainer.train_task(
                train_loader=train_loaders[task_name],
                val_loader=val_loaders[task_name],
                task_name=task_name,
                task_id=task_idx,
                num_epochs=num_epochs
            )
            
            # Evaluate on all tasks seen so far
            print(f"\n=== Evaluating on all tasks after training on {task_name} ===")
            seen_tasks = task_names[:task_idx+1]
            seen_val_loaders = {tn: val_loaders[tn] for tn in seen_tasks}
            
            task_results = evaluator.evaluate_all_tasks(
                model=trainer.model if not isinstance(trainer.strategy, ProgressiveNetworks) else trainer.strategy.model,
                dataloaders=seen_val_loaders,
                current_task_idx=task_idx,
                criterion=nn.CrossEntropyLoss()
            )
            
            # Update metrics
            metrics.update(task_idx, {tn: task_results[tn]["accuracy"] for tn in seen_tasks})
            
            # Visualize results
            visualizer.plot_task_accuracies(
                metrics.accuracy_matrix,
                task_names=seen_tasks,
                title=f"{strategy_info['name']} - Accuracy After Training on {task_name}"
            )
        
        # Save final metrics
        results[strategy_key] = metrics.get_summary()
        
        # Plot final accuracy for each strategy
        visualizer.plot_forgetting(
            metrics.accuracy_matrix, 
            task_names=task_names,
            title=f"{strategy_info['name']} - Forgetting Analysis"
        )
    
    # Compare strategies
    visualizer = ContinualLearningVisualizer(
        save_dir=os.path.join(results_dir, "plots/comparison")
    )
    
    # Extract final accuracies for each strategy
    final_accuracies = {
        name: [results[name]["task_accuracies"][task] for task in task_names]
        for name in strategies.keys()
    }
    
    # Plot comparison
    visualizer.plot_strategy_comparison(
        accuracies=final_accuracies,
        strategy_names=[strategies[k]["name"] for k in strategies.keys()],
        task_names=task_names,
        title="Strategy Comparison - Final Accuracies"
    )
    
    return results

def main():
    # Parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_dir = "results/pnn_example"
    
    # Create output directory
    os.makedirs(results_dir, exist_ok=True)
    
    # Create synthetic tasks
    print("Creating synthetic tasks...")
    tasks = create_synthetic_tasks(num_tasks=3)
    train_loaders, val_loaders = create_dataloaders(tasks, batch_size=32)
    
    # Compare strategies
    results = compare_strategies(
        tasks=tasks,
        train_loaders=train_loaders,
        val_loaders=val_loaders,
        results_dir=results_dir,
        device=device
    )
    
    # Print final results
    print("\n=== Final Results ===")
    for strategy_name, strategy_results in results.items():
        print(f"\n{strategy_name.upper()}:")
        print(f"Average Accuracy: {strategy_results['avg_accuracy']:.4f}")
        
        print("Task Accuracies:")
        for task_name, accuracy in strategy_results["task_accuracies"].items():
            print(f"  {task_name}: {accuracy:.4f}")
        
        print("Forgetting:")
        for task_name, forgetting in strategy_results.get("forgetting", {}).items():
            print(f"  {task_name}: {forgetting:.4f}")
    
if __name__ == "__main__":
    main() 
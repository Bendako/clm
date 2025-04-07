#!/usr/bin/env python
"""
Example demonstrating PackNet for continual learning.

This example shows how PackNet uses network pruning and weight freezing to allow
a single network to learn multiple tasks without catastrophic forgetting, while
being more parameter-efficient than Progressive Neural Networks.
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
from ml.continual_strategies import PackNet, ProgressiveNetworks, ProgressiveNeuralNetwork
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

def compare_strategies(tasks, train_loaders, val_loaders, results_dir, device, num_epochs=3):
    """Compare PackNet with PNN, EWC, and naive fine-tuning."""
    vocab_size = 1000
    embedding_dim = 64
    hidden_dim = 128
    batch_size = 32
    learning_rate = 1e-3
    
    strategies = {
        "packnet": {
            "name": "PackNet",
            "config": {
                "optimizer": {"type": "adam", "lr": learning_rate},
                "continual_strategy": {
                    "packnet": {
                        "enabled": True,
                        "prune_percentage": 0.75,  # 75% of weights pruned per task
                        "use_magnitude_pruning": True
                    },
                    "replay": {"enabled": False}
                },
                "checkpoint_dir": os.path.join(results_dir, "checkpoints/packnet")
            }
        },
        # Comment out other strategies for now
        # "pnn": {
        #     "name": "Progressive Neural Networks",
        #     "config": {
        #         "optimizer": {"type": "adam", "lr": learning_rate},
        #         "continual_strategy": {
        #             "pnn": {
        #                 "enabled": True,
        #                 "input_size": embedding_dim,  # After embedding
        #                 "hidden_sizes": [hidden_dim],
        #                 "output_size": vocab_size
        #             },
        #             "replay": {"enabled": False}
        #         },
        #         "checkpoint_dir": os.path.join(results_dir, "checkpoints/pnn")
        #     }
        # },
        # "ewc": {
        #     "name": "Elastic Weight Consolidation",
        #     "config": {
        #         "optimizer": {"type": "adam", "lr": learning_rate},
        #         "continual_strategy": {
        #             "ewc": {
        #                 "enabled": True,
        #                 "lambda": 100.0,  # Higher lambda for stronger regularization
        #                 "gamma": 0.9
        #             },
        #             "replay": {"enabled": False}
        #         },
        #         "checkpoint_dir": os.path.join(results_dir, "checkpoints/ewc")
        #     }
        # },
        # "naive": {
        #     "name": "Naive Fine-tuning",
        #     "config": {
        #         "optimizer": {"type": "adam", "lr": learning_rate},
        #         "continual_strategy": {
        #             "replay": {"enabled": False}
        #         },
        #         "checkpoint_dir": os.path.join(results_dir, "checkpoints/naive")
        #     }
        # }
    }
    
    results = {}
    parameter_stats = {}
    
    # Train each strategy
    for strategy_key, strategy_info in strategies.items():
        print(f"\n\n=== Training with {strategy_info['name']} ===")
        
        # Create model
        model = BaseLanguageModel(vocab_size=vocab_size, embedding_dim=embedding_dim, hidden_dim=hidden_dim)
        model.to(device)
        
        # Count initial parameters
        initial_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
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
        
        # Track parameter usage for each task
        param_usage = {"initial": initial_params}
        
        # Train on each task sequentially
        task_names = list(tasks.keys())
        
        for task_idx, task_name in enumerate(task_names):
            print(f"\n=== Training on {task_name} with {strategy_info['name']} ===")
            
            # Debug the data structure
            print(f"Examining data for {task_name}...")
            for i, batch in enumerate(train_loaders[task_name]):
                print(f"Batch type: {type(batch)}")
                if isinstance(batch, tuple):
                    print(f"Batch is a tuple of length {len(batch)}")
                    for j, item in enumerate(batch):
                        print(f"  Item {j} type: {type(item)}, ", end="")
                        if isinstance(item, torch.Tensor):
                            print(f"shape: {item.shape}, dtype: {item.dtype}")
                        else:
                            print(f"value: {item}")
                elif isinstance(batch, torch.Tensor):
                    print(f"Batch is a tensor of shape {batch.shape}, dtype: {batch.dtype}")
                else:
                    print(f"Batch is of type {type(batch)}")
                
                # Just print the first batch
                if i == 0:
                    break
            
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
            
            # For PackNet and PNN, modify evaluation for task-specific forward passes
            if strategy_key == "packnet":
                # Need to use task-specific forward with PackNet
                task_results = {}
                for tn in seen_tasks:
                    # Set the correct mask for this task
                    trainer.strategy._apply_task_specific_mask(seen_tasks.index(tn))
                    
                    # Evaluate
                    result = evaluator.evaluate_task(
                        model=trainer.model,
                        dataloader=seen_val_loaders[tn],
                        task_name=tn,
                        criterion=nn.CrossEntropyLoss()
                    )
                    task_results[tn] = result
            elif strategy_key == "pnn":
                # For PNN, need to use task-specific forward
                task_results = {}
                for tn in seen_tasks:
                    # Set the correct active column for this task
                    trainer.strategy.model.set_active_column(seen_tasks.index(tn))
                    
                    # Evaluate
                    result = evaluator.evaluate_task(
                        model=trainer.strategy.model,
                        dataloader=seen_val_loaders[tn],
                        task_name=tn,
                        criterion=nn.CrossEntropyLoss()
                    )
                    task_results[tn] = result
            else:
                # Regular evaluation for other strategies
                task_results = evaluator.evaluate_all_tasks(
                    model=trainer.model,
                    dataloaders=seen_val_loaders,
                    current_task_idx=task_idx,
                    criterion=nn.CrossEntropyLoss()
                )
            
            # Update metrics
            metrics.update(task_idx, {tn: task_results[tn]["accuracy"] for tn in seen_tasks})
            
            # Record parameter usage after this task
            if strategy_key == "pnn":
                # For PNN, count parameters in all columns
                task_params = sum(p.numel() for p in trainer.strategy.model.parameters() if p.requires_grad)
            elif strategy_key == "packnet":
                # For PackNet, count non-zero parameters
                non_zero_params = trainer.strategy._count_remaining_params()
                task_params = non_zero_params
            else:
                # For other strategies, parameter count stays constant
                task_params = initial_params
                
            param_usage[task_name] = task_params
            
            # Create temporary CL metrics object to use for plotting
            tmp_metrics = ContinualLearningMetrics()
            tmp_metrics.accuracy_matrix = metrics.accuracy_matrix
            tmp_metrics.task_names = seen_tasks
            
            # Visualize results
            visualizer.plot_task_accuracies(
                cl_metrics=tmp_metrics,
                title=f"{strategy_info['name']} - Accuracy After Training on {task_name}"
            )
        
        # Save final metrics
        results[strategy_key] = metrics.get_summary()
        parameter_stats[strategy_key] = param_usage
        
        # Plot final accuracy for each strategy
        visualizer.plot_forgetting(
            cl_metrics=tmp_metrics, 
            title=f"{strategy_info['name']} - Forgetting Analysis"
        )
    
    # Compare strategies
    visualizer = ContinualLearningVisualizer(
        save_dir=os.path.join(results_dir, "plots/comparison")
    )
    
    # Create a metrics object with all data
    final_metrics = ContinualLearningMetrics()
    final_metrics.task_names = task_names
    
    # Extract final accuracies for each strategy
    final_accuracies = {
        name: [results[name]["task_accuracies"][task] for task in task_names]
        for name in strategies.keys()
    }
    
    # Plot comparison
    for strategy_name, accuracies in final_accuracies.items():
        plt.figure(figsize=(10, 6))
        plt.bar(task_names, accuracies)
        plt.xlabel("Task")
        plt.ylabel("Accuracy")
        plt.title(f"{strategies[strategy_name]['name']} - Final Accuracies")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.ylim(0, 1.0)
        
        # Save plot
        os.makedirs(os.path.join(results_dir, "plots/comparison"), exist_ok=True)
        plt.savefig(os.path.join(results_dir, f"plots/comparison/{strategy_name}_final_accuracies.png"))
        plt.close()
    
    # Plot parameter usage
    plt.figure(figsize=(10, 6))
    
    for strategy_key, params in parameter_stats.items():
        plt.plot(
            ["Initial"] + task_names, 
            list(params.values()),
            marker='o',
            label=strategies[strategy_key]["name"]
        )
    
    plt.xlabel("Training Stage")
    plt.ylabel("Number of Parameters")
    plt.title("Parameter Usage Across Tasks")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save plot
    param_plot_path = os.path.join(results_dir, "plots/comparison/parameter_usage.png")
    os.makedirs(os.path.dirname(param_plot_path), exist_ok=True)
    plt.savefig(param_plot_path)
    plt.close()
    
    # Skip forgetting comparison since we're only using one strategy
    
    return results, parameter_stats

def main():
    # Parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_dir = "results/packnet_example"
    
    # Create output directory
    os.makedirs(results_dir, exist_ok=True)
    
    # Create synthetic tasks
    print("Creating synthetic tasks...")
    tasks = create_synthetic_tasks(num_tasks=3)
    train_loaders, val_loaders = create_dataloaders(tasks, batch_size=32)
    
    # Compare strategies
    results, parameter_stats = compare_strategies(
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
    
    print("\n=== Parameter Usage ===")
    for strategy_name, params in parameter_stats.items():
        print(f"\n{strategy_name.upper()}:")
        for stage, count in params.items():
            print(f"  {stage}: {count:,} parameters")
    
if __name__ == "__main__":
    main() 
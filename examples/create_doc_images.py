#!/usr/bin/env python
"""
Generate Example Images for Documentation

This script generates placeholder images for the benchmarking documentation.
These images illustrate typical outputs from the benchmarking framework.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def create_task_accuracy_matrix():
    """Create an example task accuracy matrix plot."""
    # Sample data (5 tasks)
    acc_matrix = np.array([
        [0.95, 0.90, 0.85, 0.82, 0.80],  # Task 1
        [0.00, 0.92, 0.88, 0.85, 0.82],  # Task 2
        [0.00, 0.00, 0.94, 0.90, 0.85],  # Task 3
        [0.00, 0.00, 0.00, 0.95, 0.91],  # Task 4
        [0.00, 0.00, 0.00, 0.00, 0.93]   # Task 5
    ])
    
    # Replace zeros with NaN for better visualization
    acc_matrix = np.where(acc_matrix == 0, np.nan, acc_matrix)
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    cmap = plt.cm.viridis
    cmap.set_bad('white', 1.0)
    im = plt.imshow(acc_matrix, cmap=cmap, vmin=0.7, vmax=1.0)
    
    # Add colorbar
    plt.colorbar(im, label='Accuracy')
    
    # Add labels
    plt.xlabel('After Learning Task')
    plt.ylabel('Task')
    plt.title('EWC - Task Accuracies (Permuted MNIST)')
    
    # Configure axis ticks
    plt.xticks(np.arange(5), np.arange(1, 6))
    plt.yticks(np.arange(5), np.arange(1, 6))
    
    # Add text annotations
    for i in range(5):
        for j in range(5):
            if j >= i:  # Only for tasks that have been trained
                plt.text(j, i, f"{acc_matrix[i, j]:.2f}", 
                         ha="center", va="center", 
                         color="white" if acc_matrix[i, j] < 0.85 else "black")
    
    plt.tight_layout()
    return plt

def create_forgetting_curve():
    """Create an example forgetting curve plot."""
    # Sample data for forgetting
    tasks = np.arange(2, 6)  # Tasks 2 to 5
    forgetting = np.array([0.05, 0.10, 0.15, 0.18])  # Forgetting after tasks 2, 3, 4, 5
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot forgetting
    plt.plot(tasks, forgetting, marker='o', linestyle='-', linewidth=2)
    
    # Add labels
    plt.xlabel('After Learning Task')
    plt.ylabel('Average Forgetting')
    plt.title('EWC - Forgetting (Permuted MNIST)')
    
    # Configure axis
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(tasks)
    plt.ylim(0, 0.2)
    
    # Add text annotations
    for i, val in enumerate(forgetting):
        plt.text(tasks[i], val + 0.01, f"{val:.3f}", ha='center')
    
    plt.tight_layout()
    return plt

def create_comparison_plot():
    """Create an example strategy comparison plot."""
    # Sample data for accuracy comparison
    strategies = ["EWC", "LwF", "GEM", "PackNet", "ER+", "GR"]
    accuracies = np.array([0.82, 0.79, 0.85, 0.78, 0.89, 0.87])
    
    # Sort by accuracy (descending)
    sorted_indices = np.argsort(accuracies)[::-1]
    strategies = [strategies[i] for i in sorted_indices]
    accuracies = accuracies[sorted_indices]
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Create bar chart
    bars = plt.bar(strategies, accuracies, width=0.6)
    
    # Add labels
    plt.xlabel('Strategy')
    plt.ylabel('Average Accuracy')
    plt.title('Comparison of Average Accuracy Across Strategies')
    
    # Configure axis
    plt.ylim(0.75, 0.95)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                 f"{height:.3f}", ha='center', va='bottom')
    
    plt.tight_layout()
    return plt

def main():
    """Generate all example images."""
    # Create output directory
    output_dir = Path("assets/images")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create task accuracy matrix
    plt_acc = create_task_accuracy_matrix()
    plt_acc.savefig(output_dir / "task_accuracy_matrix.png", dpi=150)
    plt_acc.close()
    print(f"Created {output_dir}/task_accuracy_matrix.png")
    
    # Create forgetting curve
    plt_forget = create_forgetting_curve()
    plt_forget.savefig(output_dir / "forgetting_curve.png", dpi=150)
    plt_forget.close()
    print(f"Created {output_dir}/forgetting_curve.png")
    
    # Create comparison plot
    plt_comp = create_comparison_plot()
    plt_comp.savefig(output_dir / "comparison_plot.png", dpi=150)
    plt_comp.close()
    print(f"Created {output_dir}/comparison_plot.png")
    
    print("All documentation images created successfully!")

if __name__ == "__main__":
    main() 
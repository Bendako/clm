#!/usr/bin/env python
"""
Generative Replay Example

This script demonstrates how to use the Generative Replay strategy for
continual learning using the MNIST dataset. It trains a model on a sequence
of tasks and evaluates its performance in terms of catastrophic forgetting.

The Generative Replay strategy uses a generative model (VAE) to produce
synthetic samples of previous tasks during training on new tasks.
"""

import os
import sys
import logging
import argparse
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add the parent directory to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.training.continual import ContinualTrainer
from ml.utils.data import create_permuted_mnist_tasks
from ml.models.simple import SimpleMLP, SimpleCNN
from ml.models.vae import VAE, ConvVAE

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train with Generative Replay')
    parser.add_argument(
        '--config', type=str, 
        default='configs/continual_learning/generative_replay_config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--num-tasks', type=int, default=5,
        help='Number of permuted MNIST tasks to create'
    )
    parser.add_argument(
        '--checkpoint-dir', type=str, default='checkpoints/generative_replay',
        help='Directory to save checkpoints'
    )
    return parser.parse_args()

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def create_model(config, input_dim, output_dim):
    """Create model based on config."""
    model_config = config.get('model', {})
    model_type = model_config.get('type', 'mlp')
    
    if model_type == 'mlp':
        hidden_sizes = model_config.get('hidden_sizes', [256, 128])
        dropout = model_config.get('dropout', 0.2)
        
        logger.info(f"Creating MLP with hidden sizes {hidden_sizes}")
        model = SimpleMLP(
            input_size=input_dim,
            hidden_sizes=hidden_sizes,
            output_size=output_dim,
            dropout=dropout
        )
    elif model_type == 'cnn':
        channels = model_config.get('channels', 1)
        hidden_dims = model_config.get('hidden_dims', [16, 32])
        
        logger.info(f"Creating CNN with channels={channels}, hidden_dims={hidden_dims}")
        model = SimpleCNN(
            num_channels=channels,
            hidden_dims=hidden_dims,
            output_dim=output_dim
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model

def visualize_generator_samples(generator, device, num_samples=10, img_size=28, epoch=0, save_dir='results/generative_replay'):
    """Visualize generated samples from the VAE."""
    os.makedirs(save_dir, exist_ok=True)
    
    generator.eval()
    with torch.no_grad():
        # Generate samples
        samples = generator.generate(num_samples, device)
        
        # If flattened, reshape to images
        if len(samples.shape) == 2:
            samples = samples.view(-1, 1, img_size, img_size)
            
        # Convert to numpy and plot
        samples = samples.cpu().numpy()
        
        # Create figure
        fig, axes = plt.subplots(1, num_samples, figsize=(2*num_samples, 2))
        for i, ax in enumerate(axes):
            if samples.shape[1] == 1:  # Grayscale
                ax.imshow(samples[i, 0], cmap='gray')
            else:  # RGB
                ax.imshow(np.transpose(samples[i], (1, 2, 0)))
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/generated_samples_epoch_{epoch}.png", dpi=150)
        plt.close()

def plot_results(accuracies, forgetting, save_dir='results/generative_replay'):
    """Plot accuracy and forgetting metrics."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot task accuracies
    plt.figure(figsize=(10, 6))
    for task_id, task_accs in enumerate(accuracies):
        plt.plot(task_accs, label=f'Task {task_id+1}')
    plt.xlabel('Tasks Learned')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.0)
    plt.title('Task Accuracies Throughout Training')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/task_accuracies.png", dpi=150)
    plt.close()
    
    # Plot forgetting
    plt.figure(figsize=(10, 6))
    plt.plot(forgetting, marker='o', linestyle='-')
    plt.xlabel('Tasks Learned')
    plt.ylabel('Average Forgetting')
    plt.title('Forgetting Throughout Training')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/forgetting.png", dpi=150)
    plt.close()

def main():
    """Main function."""
    args = parse_args()
    
    # Create save directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    results_dir = 'results/generative_replay'
    os.makedirs(results_dir, exist_ok=True)
    
    # Load configuration
    config = load_config(args.config)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create permuted MNIST tasks
    logger.info(f"Creating {args.num_tasks} permuted MNIST tasks...")
    task_data = create_permuted_mnist_tasks(
        num_tasks=args.num_tasks,
        batch_size=config['training']['batch_size'],
        data_dir='./data'
    )
    task_loaders, task_names = task_data
    
    # Create model
    input_dim = 784  # Flattened MNIST (28x28)
    output_dim = 10  # Number of MNIST classes (0-9)
    model = create_model(config, input_dim, output_dim)
    model.to(device)
    
    # Create continual trainer
    trainer_config = config.copy()
    trainer_config['checkpoint_dir'] = args.checkpoint_dir
    
    trainer = ContinualTrainer(
        model=model,
        config=trainer_config,
        device=device,
        experiment_name="generative_replay_experiment"
    )
    
    # Store metrics
    accuracies = [[] for _ in range(args.num_tasks)]
    forgetting = []
    
    # Train on each task sequentially
    for task_id, (train_loader, val_loader) in enumerate(task_loaders):
        logger.info(f"Training on task {task_id+1}/{args.num_tasks}: {task_names[task_id]}")
        
        # Train on current task
        trainer.train_task(
            train_loader=train_loader,
            val_loader=val_loader,
            task_id=task_id,
            task_name=task_names[task_id],
            num_epochs=config['training']['epochs']
        )
        
        # Visualize generated samples if using Generative Replay
        if hasattr(trainer, 'generator_model') and trainer.generator_model is not None:
            visualize_generator_samples(
                trainer.generator_model, 
                device,
                save_dir=f"{results_dir}/generator_samples",
                epoch=task_id
            )
        
        # Evaluate on all seen tasks
        logger.info(f"Evaluating on all seen tasks after learning task {task_id+1}...")
        
        task_forgetting = 0.0
        best_acc_per_task = [0.0] * args.num_tasks
        
        for eval_task_id in range(task_id + 1):
            _, eval_val_loader = task_loaders[eval_task_id]
            
            # Evaluate on this task
            eval_metrics = trainer.evaluate(
                eval_val_loader,
                task_id=eval_task_id,
                task_name=task_names[eval_task_id]
            )
            
            current_acc = eval_metrics['accuracy']
            accuracies[eval_task_id].append(current_acc)
            
            # Calculate forgetting for previously learned tasks
            if eval_task_id < task_id:
                # Forgetting is the difference between the best accuracy so far and current accuracy
                best_acc = max(accuracies[eval_task_id])
                forgetting_task = max(0, best_acc - current_acc)
                task_forgetting += forgetting_task
                
                logger.info(f"Task {eval_task_id+1} accuracy: {current_acc:.4f}, forgetting: {forgetting_task:.4f}")
            else:
                logger.info(f"Task {eval_task_id+1} accuracy: {current_acc:.4f}")
        
        # Calculate average forgetting over all previous tasks
        if task_id > 0:
            avg_forgetting = task_forgetting / task_id
            forgetting.append(avg_forgetting)
            logger.info(f"Average forgetting after task {task_id+1}: {avg_forgetting:.4f}")
        
    # Plot results
    logger.info("Plotting results...")
    plot_results(accuracies, forgetting, save_dir=results_dir)
    
    logger.info("Done!")

if __name__ == "__main__":
    main() 
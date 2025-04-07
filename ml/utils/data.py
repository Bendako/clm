"""
Utility functions for data handling, particularly for continual learning tasks.
"""

import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
from torchvision import datasets, transforms
from typing import List, Tuple, Dict, Optional, Union
import logging

logger = logging.getLogger(__name__)

def create_permuted_mnist_tasks(
    num_tasks: int = 5,
    batch_size: int = 64,
    train_split: float = 0.8,
    data_dir: str = "./data"
) -> Tuple[List[Tuple[DataLoader, DataLoader]], List[str]]:
    """
    Create a sequence of permuted MNIST tasks for continual learning.
    
    Each task is a different permutation of the pixels in the MNIST dataset.
    
    Args:
        num_tasks: Number of different permuted tasks to create
        batch_size: Batch size for dataloaders
        train_split: Fraction of data to use for training (vs. validation)
        data_dir: Directory to store datasets
        
    Returns:
        - A list of (train_loader, val_loader) pairs for each task
        - A list of task names
    """
    # Download and prepare MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    mnist_train = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    
    # Combine all MNIST data and flatten images
    train_images = mnist_train.data.float().view(mnist_train.data.shape[0], -1) / 255.0
    train_labels = mnist_train.targets
    test_images = mnist_test.data.float().view(mnist_test.data.shape[0], -1) / 255.0
    test_labels = mnist_test.targets
    
    all_images = torch.cat([train_images, test_images], dim=0)
    all_labels = torch.cat([train_labels, test_labels], dim=0)
    
    # Create permutations for each task (first is identity)
    permutations = [torch.arange(all_images.shape[1])]
    for _ in range(num_tasks - 1):
        permutations.append(torch.randperm(all_images.shape[1]))
    
    # Create dataloaders for each task
    tasks = []
    task_names = []
    
    for task_id, perm in enumerate(permutations):
        # Permute the pixels
        task_images = all_images[:, perm]
        
        # Create a TensorDataset
        dataset = TensorDataset(task_images, all_labels)
        
        # Split into train and validation
        train_size = int(len(dataset) * train_split)
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        tasks.append((train_loader, val_loader))
        task_names.append(f"PermutedMNIST-{task_id}")
        
        logger.info(f"Created task {task_id+1}/{num_tasks}: {task_names[-1]}")
    
    return tasks, task_names

def create_split_mnist_tasks(
    batch_size: int = 64,
    train_split: float = 0.8,
    data_dir: str = "./data"
) -> Tuple[List[Tuple[DataLoader, DataLoader]], List[str]]:
    """
    Create Split MNIST tasks for continual learning.
    
    Each task consists of a binary classification problem with two digits.
    
    Args:
        batch_size: Batch size for dataloaders
        train_split: Fraction of data to use for training (vs. validation)
        data_dir: Directory to store datasets
        
    Returns:
        - A list of (train_loader, val_loader) pairs for each task
        - A list of task names
    """
    # Download and prepare MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    mnist_train = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    
    # Combine all MNIST data and flatten images
    train_images = mnist_train.data.float().view(mnist_train.data.shape[0], -1) / 255.0
    train_labels = mnist_train.targets
    test_images = mnist_test.data.float().view(mnist_test.data.shape[0], -1) / 255.0
    test_labels = mnist_test.targets
    
    all_images = torch.cat([train_images, test_images], dim=0)
    all_labels = torch.cat([train_labels, test_labels], dim=0)
    
    # Define split MNIST tasks (pairs of digits)
    digit_pairs = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
    
    # Create dataloaders for each task
    tasks = []
    task_names = []
    
    for task_id, (digit1, digit2) in enumerate(digit_pairs):
        # Filter data for the two digits
        mask = (all_labels == digit1) | (all_labels == digit2)
        task_images = all_images[mask]
        task_labels = all_labels[mask]
        
        # Remap labels to 0 and 1 for binary classification
        task_labels = (task_labels == digit2).long()
        
        # Create a TensorDataset
        dataset = TensorDataset(task_images, task_labels)
        
        # Split into train and validation
        train_size = int(len(dataset) * train_split)
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        tasks.append((train_loader, val_loader))
        task_names.append(f"SplitMNIST-{digit1}vs{digit2}")
        
        logger.info(f"Created task {task_id+1}/{len(digit_pairs)}: {task_names[-1]}")
    
    return tasks, task_names 
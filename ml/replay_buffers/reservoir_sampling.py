"""
Reservoir Sampling Replay Buffer

This module implements a replay buffer for continual learning using reservoir sampling.
Reservoir sampling allows maintaining a fixed-size random sample from a stream of data
of unknown or unlimited size, ensuring each item has an equal probability of being
selected regardless of when it was observed.

Reference:
Vitter, J. S. (1985). Random sampling with a reservoir.
ACM Transactions on Mathematical Software, 11(1), 37-57.
"""

import numpy as np
import random
from typing import List, Dict, Any, Tuple, Optional, TypeVar, Generic, Union
from collections import defaultdict
import logging
import torch

T = TypeVar('T')  # Generic type for buffer items

logger = logging.getLogger(__name__)

class ReservoirBuffer(Generic[T]):
    """
    A replay buffer implementation using reservoir sampling algorithm.
    
    This buffer maintains a representative sample of data when its capacity is reached,
    and supports both uniform sampling and task-balanced sampling strategies.
    """
    
    def __init__(
        self, 
        capacity: int, 
        task_balanced: bool = True,
        seed: Optional[int] = None,
        item_transform: Optional[callable] = None,
        device: str = "cpu"
    ):
        """
        Initialize a reservoir sampling buffer.
        
        Args:
            capacity: Maximum number of items to store in the buffer
            task_balanced: If True, sampling will be balanced across tasks
            seed: Random seed for reproducibility
            item_transform: Optional function to transform items when sampling
            device: Device to store tensors on ("cpu" or "cuda")
        """
        self.capacity = capacity
        self.task_balanced = task_balanced
        self.item_transform = item_transform
        self.device = device
        
        # Set random seed if provided
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        # Buffer storage
        self.buffer: List[T] = []
        self.task_ids: List[str] = []  # Parallel list of task IDs
        self.task_to_indices: Dict[str, List[int]] = defaultdict(list)
        self.items_seen = 0
        
        logger.info(f"Initialized ReservoirBuffer with capacity {capacity}")
    
    def add(self, item: T, task_id: str) -> bool:
        """
        Add an item to the buffer using reservoir sampling.
        
        Args:
            item: The item to add to the buffer
            task_id: The ID of the task this item belongs to
            
        Returns:
            True if the item was added, False if it was rejected
        """
        self.items_seen += 1
        
        # If buffer is not full, add the item
        if len(self.buffer) < self.capacity:
            self.buffer.append(item)
            self.task_ids.append(task_id)
            self.task_to_indices[task_id].append(len(self.buffer) - 1)
            return True
        else:
            # Reservoir sampling: with probability capacity/items_seen, replace a random item
            probability = self.capacity / self.items_seen
            if random.random() < probability:
                # Select a random index to replace
                idx = random.randrange(self.capacity)
                old_task = self.task_ids[idx]
                
                # Update task indices
                self.task_to_indices[old_task].remove(idx)
                self.task_ids[idx] = task_id
                self.task_to_indices[task_id].append(idx)
                
                # Replace the item
                self.buffer[idx] = item
                return True
                
        return False
    
    def add_batch(self, items: List[T], task_id: str) -> int:
        """
        Add a batch of items to the buffer.
        
        Args:
            items: List of items to add
            task_id: Task ID for all items in the batch
            
        Returns:
            Number of items successfully added
        """
        added = 0
        for item in items:
            if self.add(item, task_id):
                added += 1
        return added
    
    def sample(self, batch_size: int) -> List[T]:
        """
        Sample a batch of items from the buffer.
        
        Args:
            batch_size: Number of items to sample
            
        Returns:
            List of sampled items
        """
        if not self.buffer:
            raise ValueError("Cannot sample from an empty buffer")
            
        # Perform sampling based on strategy
        if self.task_balanced and len(self.task_to_indices) > 1:
            return self._sample_task_balanced(batch_size)
        else:
            return self._sample_uniform(batch_size)
    
    def _sample_uniform(self, batch_size: int) -> List[T]:
        """Sample items uniformly from the buffer."""
        # Sample with replacement if batch_size > buffer size
        indices = random.choices(range(len(self.buffer)), k=batch_size)
        
        # Apply transformation if provided
        if self.item_transform:
            return [self.item_transform(self.buffer[i]) for i in indices]
        return [self.buffer[i] for i in indices]
    
    def _sample_task_balanced(self, batch_size: int) -> List[T]:
        """Sample an equal number of items from each task."""
        tasks = list(self.task_to_indices.keys())
        items_per_task = batch_size // len(tasks)
        remainder = batch_size % len(tasks)
        
        sampled_items = []
        
        # Sample items_per_task from each task
        for task in tasks:
            task_indices = self.task_to_indices[task]
            # Sample with replacement if we need more items than available
            sample_indices = random.choices(task_indices, k=items_per_task)
            sampled_items.extend([self.buffer[i] for i in sample_indices])
        
        # Sample remainder items from random tasks
        if remainder > 0:
            extra_tasks = random.choices(tasks, k=remainder)
            for task in extra_tasks:
                idx = random.choice(self.task_to_indices[task])
                sampled_items.append(self.buffer[idx])
        
        # Apply transformation if provided
        if self.item_transform:
            sampled_items = [self.item_transform(item) for item in sampled_items]
            
        # Shuffle to avoid task blocks
        random.shuffle(sampled_items)
        return sampled_items
    
    def sample_task(self, task_id: str, batch_size: int) -> List[T]:
        """
        Sample items only from a specific task.
        
        Args:
            task_id: The task ID to sample from
            batch_size: Number of items to sample
            
        Returns:
            List of sampled items from the specified task
        """
        if task_id not in self.task_to_indices or not self.task_to_indices[task_id]:
            raise ValueError(f"No items from task {task_id} in the buffer")
            
        # Sample with replacement if batch_size > number of task items
        indices = random.choices(self.task_to_indices[task_id], k=batch_size)
        
        # Apply transformation if provided
        if self.item_transform:
            return [self.item_transform(self.buffer[i]) for i in indices]
        return [self.buffer[i] for i in indices]
    
    def get_all_items(self, task_id: Optional[str] = None) -> List[T]:
        """
        Get all items in the buffer, optionally filtered by task.
        
        Args:
            task_id: If provided, only return items from this task
            
        Returns:
            List of all items (or all items from the specified task)
        """
        if task_id is None:
            return self.buffer.copy()
        
        if task_id not in self.task_to_indices:
            return []
            
        return [self.buffer[i] for i in self.task_to_indices[task_id]]
    
    def get_task_distribution(self) -> Dict[str, float]:
        """
        Get the distribution of tasks in the buffer.
        
        Returns:
            Dictionary mapping task IDs to their proportion in the buffer
        """
        if not self.buffer:
            return {}
            
        distribution = {}
        buffer_size = len(self.buffer)
        
        for task_id, indices in self.task_to_indices.items():
            distribution[task_id] = len(indices) / buffer_size
            
        return distribution
    
    def is_empty(self) -> bool:
        """Check if the buffer is empty."""
        return len(self.buffer) == 0
        
    def is_full(self) -> bool:
        """Check if the buffer is at capacity."""
        return len(self.buffer) >= self.capacity
    
    def clear(self) -> None:
        """Clear all items from the buffer."""
        self.buffer.clear()
        self.task_ids.clear()
        self.task_to_indices.clear()
        logger.info("Cleared buffer")
        
    def __len__(self) -> int:
        """Return the current size of the buffer."""
        return len(self.buffer)
    
    def stats(self) -> Dict[str, Any]:
        """
        Get statistics about the buffer.
        
        Returns:
            Dictionary with buffer statistics
        """
        return {
            "capacity": self.capacity,
            "size": len(self.buffer),
            "items_seen": self.items_seen,
            "num_tasks": len(self.task_to_indices),
            "tasks": list(self.task_to_indices.keys()),
            "task_distribution": self.get_task_distribution(),
            "is_full": self.is_full()
        }


class TensorReservoirBuffer(ReservoirBuffer[torch.Tensor]):
    """
    A specialized Reservoir Buffer for PyTorch tensors with additional functionality.
    """
    
    def __init__(
        self, 
        capacity: int, 
        task_balanced: bool = True,
        seed: Optional[int] = None,
        device: str = "cpu"
    ):
        """
        Initialize a tensor-specific reservoir buffer.
        
        Args:
            capacity: Maximum number of tensors to store in the buffer
            task_balanced: If True, sampling will be balanced across tasks
            seed: Random seed for reproducibility
            device: Device to store tensors on ("cpu" or "cuda")
        """
        super().__init__(capacity, task_balanced, seed, None, device)
        
    def add(self, item: torch.Tensor, task_id: str) -> bool:
        """Add a tensor to the buffer, ensuring it's on the right device."""
        # Move tensor to the correct device
        if item.device.type != self.device:
            item = item.to(self.device)
        return super().add(item, task_id)
    
    def sample_tensors(self, batch_size: int) -> torch.Tensor:
        """
        Sample a batch of tensors and stack them into a single tensor.
        
        Args:
            batch_size: Number of tensors to sample
            
        Returns:
            Stacked tensor of samples
        """
        samples = self.sample(batch_size)
        return torch.stack(samples)
    
    def sample_task_tensors(self, task_id: str, batch_size: int) -> torch.Tensor:
        """
        Sample a batch of tensors from a specific task and stack them.
        
        Args:
            task_id: The task ID to sample from
            batch_size: Number of tensors to sample
            
        Returns:
            Stacked tensor of samples from the specified task
        """
        samples = self.sample_task(task_id, batch_size)
        return torch.stack(samples) 
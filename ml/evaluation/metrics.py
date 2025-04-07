import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import torch
from collections import defaultdict


class ContinualLearningMetrics:
    """
    A class to calculate and track metrics specific to continual learning scenarios.
    
    This includes metrics such as:
    - Average accuracy
    - Backward transfer (how new learning affects performance on previous tasks)
    - Forward transfer (how previous learning affects performance on new tasks)
    - Forgetting measure
    """
    
    def __init__(self):
        """Initialize the metrics tracker."""
        # Store accuracy for each task after training on each task
        # Format: accuracy_matrix[i][j] = accuracy on task j after training on task i
        self.accuracy_matrix = []
        self.task_names = []
        
    def update(self, current_task_idx: int, all_tasks_results: Dict[str, float]):
        """
        Update metrics after evaluating on all tasks.
        
        Args:
            current_task_idx: Index of the task that was just trained
            all_tasks_results: Dictionary mapping task names to accuracy scores
        """
        # Extend the accuracy matrix if needed
        while len(self.accuracy_matrix) <= current_task_idx:
            self.accuracy_matrix.append([])
            
        # Update task names if new ones are present
        for task_name in all_tasks_results.keys():
            if task_name not in self.task_names:
                self.task_names.append(task_name)
                
        # Ensure all rows have same length (one entry per task)
        for i in range(len(self.accuracy_matrix)):
            while len(self.accuracy_matrix[i]) < len(self.task_names):
                self.accuracy_matrix[i].append(float('nan'))  # Fill with NaN for missing values
        
        # Add the new accuracies
        accuracies = []
        for task_name in self.task_names:
            if task_name in all_tasks_results:
                accuracies.append(all_tasks_results[task_name])
            else:
                accuracies.append(float('nan'))
                
        self.accuracy_matrix[current_task_idx] = accuracies
    
    def avg_accuracy(self, task_idx: Optional[int] = None) -> float:
        """
        Calculate average accuracy across all tasks.
        
        Args:
            task_idx: If provided, calculate avg accuracy after training on this task
                      If None, use the most recent task
        
        Returns:
            Average accuracy across all tasks
        """
        if not self.accuracy_matrix:
            return 0.0
            
        if task_idx is None:
            task_idx = len(self.accuracy_matrix) - 1
            
        accuracies = [acc for acc in self.accuracy_matrix[task_idx] if not np.isnan(acc)]
        if not accuracies:
            return 0.0
        return sum(accuracies) / len(accuracies)
    
    def forgetting(self, task_idx: int) -> float:
        """
        Calculate forgetting measure for a specific task.
        
        Forgetting is defined as the difference between the peak accuracy on this task
        and the current accuracy.
        
        Args:
            task_idx: Index of the task to calculate forgetting for
            
        Returns:
            Forgetting measure (higher means more forgetting)
        """
        if task_idx >= len(self.task_names) or not self.accuracy_matrix:
            return 0.0
            
        # Get accuracy values for this task across all trained models
        task_accuracies = [row[task_idx] for row in self.accuracy_matrix if len(row) > task_idx]
        task_accuracies = [acc for acc in task_accuracies if not np.isnan(acc)]
        
        if not task_accuracies:
            return 0.0
            
        # Peak accuracy for this task
        peak_accuracy = max(task_accuracies)
        
        # Current accuracy
        current_accuracy = self.accuracy_matrix[-1][task_idx]
        
        # Calculate forgetting
        return max(0, peak_accuracy - current_accuracy)
    
    def backward_transfer(self, task_idx: int) -> float:
        """
        Calculate backward transfer for a specific task.
        
        Backward transfer measures how training on subsequent tasks affects 
        performance on a previous task.
        
        Args:
            task_idx: Index of the task to calculate backward transfer for
            
        Returns:
            Backward transfer value (positive = good, negative = interference)
        """
        if task_idx >= len(self.task_names) or len(self.accuracy_matrix) <= task_idx:
            return 0.0
            
        # Initial accuracy right after learning the task
        initial_accuracy = self.accuracy_matrix[task_idx][task_idx]
        
        # Current accuracy after learning all subsequent tasks
        current_accuracy = self.accuracy_matrix[-1][task_idx]
        
        return current_accuracy - initial_accuracy
    
    def forward_transfer(self, task_idx: int) -> float:
        """
        Calculate forward transfer for a specific task.
        
        Forward transfer measures how well the model performs on a new task
        before specifically training on it (transfer learning effect).
        
        Args:
            task_idx: Index of the task to calculate forward transfer for
            
        Returns:
            Forward transfer value (higher = better transfer)
        """
        if task_idx == 0 or task_idx >= len(self.task_names) or len(self.accuracy_matrix) <= task_idx - 1:
            return 0.0
            
        # Performance before learning the task (zero-shot learning)
        if task_idx < len(self.accuracy_matrix[task_idx-1]):
            pre_learning_accuracy = self.accuracy_matrix[task_idx-1][task_idx]
        else:
            return 0.0
            
        # Random initialization performance (assumed to be zero, but could be replaced with a baseline)
        random_init_accuracy = 0.0
        
        return pre_learning_accuracy - random_init_accuracy
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all metrics.
        
        Returns:
            Dictionary containing all metrics
        """
        if not self.accuracy_matrix:
            return {"error": "No data collected yet"}
            
        final_accuracies = self.accuracy_matrix[-1]
        
        summary = {
            "avg_accuracy": self.avg_accuracy(),
            "task_accuracies": {name: acc for name, acc in zip(self.task_names, final_accuracies)},
            "forgetting": {name: self.forgetting(i) for i, name in enumerate(self.task_names) if i < len(self.accuracy_matrix) - 1},
            "backward_transfer": {name: self.backward_transfer(i) for i, name in enumerate(self.task_names) if i < len(self.accuracy_matrix) - 1},
            "forward_transfer": {name: self.forward_transfer(i) for i, name in enumerate(self.task_names) if i > 0}
        }
        
        return summary


class EvaluationTracker:
    """
    Track evaluation metrics during continual learning training.
    
    This class manages tracking metrics across different tasks, seeds, and hyperparameters.
    """
    
    def __init__(self, log_to_mlflow: bool = True):
        """
        Initialize the evaluation tracker.
        
        Args:
            log_to_mlflow: Whether to log metrics to MLflow
        """
        self.log_to_mlflow = log_to_mlflow
        self.task_metrics = defaultdict(list)  # Maps task name to list of metric values
        self.cl_metrics = ContinualLearningMetrics()
        self.task_confusion_matrices = {}  # For storing confusion matrices per task
        
    def log_task_metrics(self, task_name: str, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log metrics for a specific task.
        
        Args:
            task_name: Name of the task
            metrics: Dictionary of metric name to value
            step: Optional step number for logging
        """
        # Store locally
        for metric_name, value in metrics.items():
            self.task_metrics[f"{task_name}/{metric_name}"].append(value)
            
        # Log to MLflow
        if self.log_to_mlflow:
            try:
                import mlflow
                for metric_name, value in metrics.items():
                    mlflow.log_metric(f"{task_name}/{metric_name}", value, step=step)
            except ImportError:
                pass
    
    def update_confusion_matrix(self, task_name: str, 
                               predictions: Union[torch.Tensor, np.ndarray], 
                               targets: Union[torch.Tensor, np.ndarray],
                               num_classes: int):
        """
        Update confusion matrix for a specific task.
        
        Args:
            task_name: Name of the task
            predictions: Model predictions (class indices)
            targets: Ground truth labels
            num_classes: Number of classes
        """
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
            
        # Ensure predictions are class indices
        if predictions.ndim > 1 and predictions.shape[1] > 1:
            predictions = np.argmax(predictions, axis=1)
            
        # Initialize confusion matrix if needed
        if task_name not in self.task_confusion_matrices:
            self.task_confusion_matrices[task_name] = np.zeros((num_classes, num_classes), dtype=np.int64)
            
        # Update confusion matrix
        for pred, target in zip(predictions, targets):
            self.task_confusion_matrices[task_name][target, pred] += 1
    
    def update_cl_metrics(self, current_task_idx: int, all_tasks_accuracy: Dict[str, float]):
        """
        Update continual learning metrics.
        
        Args:
            current_task_idx: Index of the current task
            all_tasks_accuracy: Dictionary mapping task names to accuracy values
        """
        self.cl_metrics.update(current_task_idx, all_tasks_accuracy)
        
        # Log CL metrics
        if self.log_to_mlflow:
            try:
                import mlflow
                summary = self.cl_metrics.get_summary()
                mlflow.log_metric("avg_accuracy", summary["avg_accuracy"])
                
                # Log individual task accuracies
                for task_name, acc in summary["task_accuracies"].items():
                    mlflow.log_metric(f"final_accuracy/{task_name}", acc)
                    
                # Log forgetting
                for task_name, forgetting in summary.get("forgetting", {}).items():
                    mlflow.log_metric(f"forgetting/{task_name}", forgetting)
                    
                # Log backward transfer
                for task_name, bt in summary.get("backward_transfer", {}).items():
                    mlflow.log_metric(f"backward_transfer/{task_name}", bt)
            except ImportError:
                pass
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all evaluation metrics.
        
        Returns:
            Dictionary containing summary of all metrics
        """
        summary = {
            "task_metrics": {k: np.mean(v) for k, v in self.task_metrics.items()},
            "continual_learning_metrics": self.cl_metrics.get_summary(),
            "num_tasks_seen": len(self.cl_metrics.task_names)
        }
        return summary
    
    def log_final_metrics(self):
        """Log final summary metrics to MLflow."""
        if not self.log_to_mlflow:
            return
            
        try:
            import mlflow
            summary = self.get_summary()
            
            # Log aggregated metrics
            mlflow.log_metric("final_avg_accuracy", summary["continual_learning_metrics"]["avg_accuracy"])
            mlflow.log_metric("num_tasks", summary["num_tasks_seen"])
            
            # Calculate average forgetting across all tasks
            if "forgetting" in summary["continual_learning_metrics"]:
                forgetting_values = list(summary["continual_learning_metrics"]["forgetting"].values())
                if forgetting_values:
                    avg_forgetting = sum(forgetting_values) / len(forgetting_values)
                    mlflow.log_metric("avg_forgetting", avg_forgetting)
                    
            # Calculate average backward transfer
            if "backward_transfer" in summary["continual_learning_metrics"]:
                bt_values = list(summary["continual_learning_metrics"]["backward_transfer"].values())
                if bt_values:
                    avg_bt = sum(bt_values) / len(bt_values)
                    mlflow.log_metric("avg_backward_transfer", avg_bt)
        except ImportError:
            pass 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple, Union
import os

from ml.evaluation.metrics import ContinualLearningMetrics


class ContinualLearningVisualizer:
    """
    Visualizer for continual learning metrics.
    
    This class creates visual representations of key continual learning metrics
    to help understand and analyze model performance across tasks.
    """
    
    def __init__(self, save_dir: Optional[str] = None):
        """
        Initialize the visualizer.
        
        Args:
            save_dir: Directory to save generated plots (None = don't save)
        """
        self.save_dir = save_dir
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        # Set style for plots
        sns.set_style("whitegrid")
        plt.rcParams["figure.figsize"] = (12, 8)
    
    def plot_accuracy_matrix(self, 
                             cl_metrics: ContinualLearningMetrics, 
                             title: str = "Accuracy Matrix",
                             show: bool = True,
                             filename: Optional[str] = "accuracy_matrix.png") -> plt.Figure:
        """
        Plot the accuracy matrix for all tasks.
        
        Args:
            cl_metrics: ContinualLearningMetrics object
            title: Plot title
            show: Whether to display the plot
            filename: Filename for saving (None = don't save)
            
        Returns:
            Matplotlib figure
        """
        matrix = np.array(cl_metrics.accuracy_matrix)
        task_names = cl_metrics.task_names
        
        if len(matrix) == 0 or len(task_names) == 0:
            print("No data to visualize")
            return None
            
        fig, ax = plt.subplots()
        im = ax.imshow(matrix, cmap="YlGnBu")
        
        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("Accuracy", rotation=-90, va="bottom")
        
        # Set labels and ticks
        ax.set_xticks(np.arange(len(task_names)))
        ax.set_yticks(np.arange(len(matrix)))
        ax.set_xticklabels(task_names)
        ax.set_yticklabels([f"After Task {i+1}" for i in range(len(matrix))])
        
        # Rotate tick labels and set alignment
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add text annotations
        for i in range(len(matrix)):
            for j in range(len(task_names)):
                if j < len(matrix[i]) and not np.isnan(matrix[i][j]):
                    ax.text(j, i, f"{matrix[i][j]:.2f}", ha="center", va="center", 
                            color="black" if matrix[i][j] > 0.5 else "white")
        
        # Set title and layout
        ax.set_title(title)
        fig.tight_layout()
        
        # Save figure
        if self.save_dir and filename:
            plt.savefig(os.path.join(self.save_dir, filename), dpi=300, bbox_inches="tight")
            
        if show:
            plt.show()
        else:
            plt.close()
            
        return fig
    
    def plot_forgetting(self, 
                       cl_metrics: ContinualLearningMetrics,
                       title: str = "Forgetting Across Tasks",
                       show: bool = True,
                       filename: Optional[str] = "forgetting.png") -> plt.Figure:
        """
        Plot forgetting measures for each task.
        
        Args:
            cl_metrics: ContinualLearningMetrics object
            title: Plot title
            show: Whether to display the plot
            filename: Filename for saving (None = don't save)
            
        Returns:
            Matplotlib figure
        """
        task_names = cl_metrics.task_names[:-1]  # Exclude the last task (no forgetting)
        
        if len(task_names) == 0:
            print("Not enough tasks to visualize forgetting")
            return None
            
        forgetting_values = [cl_metrics.forgetting(i) for i in range(len(task_names))]
        
        fig, ax = plt.subplots()
        bars = ax.bar(task_names, forgetting_values, color=sns.color_palette("muted"))
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.2f}",
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha="center", va="bottom")
        
        # Set labels and title
        ax.set_xlabel("Task")
        ax.set_ylabel("Forgetting Measure")
        ax.set_title(title)
        
        # Add horizontal line at y=0
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        
        fig.tight_layout()
        
        # Save figure
        if self.save_dir and filename:
            plt.savefig(os.path.join(self.save_dir, filename), dpi=300, bbox_inches="tight")
            
        if show:
            plt.show()
        else:
            plt.close()
            
        return fig
    
    def plot_transfer(self, 
                     cl_metrics: ContinualLearningMetrics,
                     title: str = "Knowledge Transfer",
                     show: bool = True,
                     filename: Optional[str] = "transfer.png") -> plt.Figure:
        """
        Plot backward and forward transfer measures.
        
        Args:
            cl_metrics: ContinualLearningMetrics object
            title: Plot title
            show: Whether to display the plot
            filename: Filename for saving (None = don't save)
            
        Returns:
            Matplotlib figure
        """
        task_names = cl_metrics.task_names
        
        if len(task_names) <= 1:
            print("Not enough tasks to visualize transfer")
            return None
            
        # Backward transfer (excluding the last task)
        bt_tasks = task_names[:-1]
        bt_values = [cl_metrics.backward_transfer(i) for i in range(len(bt_tasks))]
        
        # Forward transfer (excluding the first task)
        ft_tasks = task_names[1:]
        ft_values = [cl_metrics.forward_transfer(i+1) for i in range(len(ft_tasks))]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot backward transfer
        bt_bars = ax1.bar(bt_tasks, bt_values, color=sns.color_palette("muted")[0])
        ax1.set_xlabel("Task")
        ax1.set_ylabel("Backward Transfer")
        ax1.set_title("Backward Transfer (impact of new learning on previous tasks)")
        ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        
        # Add value labels
        for bar in bt_bars:
            height = bar.get_height()
            ax1.annotate(f"{height:.2f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3 if height >= 0 else -15),
                        textcoords="offset points",
                        ha="center", va="bottom" if height >= 0 else "top")
        
        # Plot forward transfer
        ft_bars = ax2.bar(ft_tasks, ft_values, color=sns.color_palette("muted")[1])
        ax2.set_xlabel("Task")
        ax2.set_ylabel("Forward Transfer")
        ax2.set_title("Forward Transfer (zero-shot performance on new tasks)")
        ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        
        # Add value labels
        for bar in ft_bars:
            height = bar.get_height()
            ax2.annotate(f"{height:.2f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3 if height >= 0 else -15),
                        textcoords="offset points",
                        ha="center", va="bottom" if height >= 0 else "top")
        
        fig.suptitle(title, fontsize=16)
        fig.tight_layout()
        
        # Save figure
        if self.save_dir and filename:
            plt.savefig(os.path.join(self.save_dir, filename), dpi=300, bbox_inches="tight")
            
        if show:
            plt.show()
        else:
            plt.close()
            
        return fig
    
    def plot_task_accuracies(self, 
                           cl_metrics: ContinualLearningMetrics,
                           title: str = "Task Accuracies Over Time",
                           show: bool = True,
                           filename: Optional[str] = "task_accuracies.png") -> plt.Figure:
        """
        Plot the accuracy for each task over time (as more tasks are learned).
        
        Args:
            cl_metrics: ContinualLearningMetrics object
            title: Plot title
            show: Whether to display the plot
            filename: Filename for saving (None = don't save)
            
        Returns:
            Matplotlib figure
        """
        matrix = np.array(cl_metrics.accuracy_matrix)
        task_names = cl_metrics.task_names
        
        if len(matrix) == 0 or len(task_names) == 0:
            print("No data to visualize")
            return None
            
        # Convert to full matrix with NaN for missing values
        full_matrix = np.zeros((len(matrix), len(task_names))) * np.nan
        for i, row in enumerate(matrix):
            for j, val in enumerate(row):
                if j < len(row):
                    full_matrix[i, j] = val
        
        fig, ax = plt.subplots()
        
        # Plot line for each task
        x = np.arange(1, len(matrix) + 1)  # Task numbers
        for j, task_name in enumerate(task_names):
            task_accuracies = full_matrix[:, j]
            ax.plot(x, task_accuracies, marker='o', label=task_name)
            
        # Add markers for after-trained-on points
        for j, task_name in enumerate(task_names):
            if j < len(matrix):  # Only for tasks we've trained on
                if j < full_matrix.shape[1] and j < len(x):
                    ax.plot(x[j], full_matrix[j, j], 'ro', markersize=10, markeredgecolor='black')
        
        # Set labels and title
        ax.set_xlabel("Tasks Trained")
        ax.set_ylabel("Accuracy")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([f"After Task {i+1}" for i in range(len(matrix))])
        ax.set_ylim(0, 1.05)
        
        # Add legend
        ax.legend(loc="lower right")
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        fig.tight_layout()
        
        # Save figure
        if self.save_dir and filename:
            plt.savefig(os.path.join(self.save_dir, filename), dpi=300, bbox_inches="tight")
            
        if show:
            plt.show()
        else:
            plt.close()
            
        return fig
    
    def plot_confusion_matrix(self, 
                             matrix: np.ndarray,
                             class_names: List[str],
                             title: str = "Confusion Matrix",
                             show: bool = True,
                             filename: Optional[str] = "confusion_matrix.png") -> plt.Figure:
        """
        Plot a confusion matrix.
        
        Args:
            matrix: Confusion matrix as numpy array
            class_names: Names of the classes
            title: Plot title
            show: Whether to display the plot
            filename: Filename for saving (None = don't save)
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Normalize the matrix
        matrix_normalized = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
        matrix_normalized = np.nan_to_num(matrix_normalized)  # Replace NaNs with 0
        
        im = ax.imshow(matrix_normalized, interpolation='nearest', cmap=plt.cm.Blues)
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("Normalized Count", rotation=-90, va="bottom")
        
        # Set labels and title
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(title)
        
        # Add ticks and labels
        tick_marks = np.arange(len(class_names))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticklabels(class_names)
        
        # Add text annotations (only for matrices that aren't too large)
        if len(class_names) <= 30:
            fmt = '.2f' if matrix_normalized.max() < 0.1 else '.2f'
            thresh = matrix_normalized.max() / 2.
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    ax.text(j, i, format(matrix_normalized[i, j], fmt) if matrix_normalized[i, j] > 0 else '',
                            ha="center", va="center",
                            color="white" if matrix_normalized[i, j] > thresh else "black")
        
        fig.tight_layout()
        
        # Save figure
        if self.save_dir and filename:
            plt.savefig(os.path.join(self.save_dir, filename), dpi=300, bbox_inches="tight")
            
        if show:
            plt.show()
        else:
            plt.close()
            
        return fig
    
    def create_summary_dashboard(self, 
                               cl_metrics: ContinualLearningMetrics,
                               conf_matrices: Optional[Dict[str, np.ndarray]] = None,
                               class_names: Optional[Dict[str, List[str]]] = None,
                               title: str = "Continual Learning Summary",
                               show: bool = True,
                               filename: Optional[str] = "summary_dashboard.png") -> plt.Figure:
        """
        Create a comprehensive summary dashboard of all metrics.
        
        Args:
            cl_metrics: ContinualLearningMetrics object
            conf_matrices: Optional dict mapping task names to confusion matrices
            class_names: Optional dict mapping task names to class names for confusion matrices
            title: Dashboard title
            show: Whether to display the dashboard
            filename: Filename for saving (None = don't save)
            
        Returns:
            Matplotlib figure
        """
        if len(cl_metrics.task_names) == 0:
            print("No data to visualize")
            return None
            
        # Create subplot grid
        if conf_matrices and len(conf_matrices) > 0:
            # If we have confusion matrices, create a larger grid
            task_count = min(3, len(conf_matrices))
            fig = plt.figure(figsize=(20, 10 + 5 * task_count))
            gs = plt.GridSpec(3 + task_count, 2, figure=fig)
        else:
            # Basic metrics only
            fig = plt.figure(figsize=(20, 15))
            gs = plt.GridSpec(3, 2, figure=fig)
            
        # Add title
        fig.suptitle(title, fontsize=20, y=0.98)
        
        # Accuracy matrix
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_accuracy_matrix_on_axis(cl_metrics, ax1)
        
        # Task accuracies over time
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_task_accuracies_on_axis(cl_metrics, ax2)
        
        # Forgetting
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_forgetting_on_axis(cl_metrics, ax3)
        
        # Final task accuracies
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_final_accuracies_on_axis(cl_metrics, ax4)
        
        # Backward and forward transfer
        ax5 = fig.add_subplot(gs[2, :])
        self._plot_transfer_on_axis(cl_metrics, ax5)
        
        # Confusion matrices (if provided)
        if conf_matrices and class_names:
            for i, (task_name, conf_matrix) in enumerate(list(conf_matrices.items())[:task_count]):
                if task_name in class_names:
                    ax = fig.add_subplot(gs[3+i, :])
                    self._plot_confusion_matrix_on_axis(
                        conf_matrix, 
                        class_names[task_name], 
                        ax, 
                        title=f"Confusion Matrix: {task_name}"
                    )
        
        fig.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust for title
        
        # Save figure
        if self.save_dir and filename:
            plt.savefig(os.path.join(self.save_dir, filename), dpi=300, bbox_inches="tight")
            
        if show:
            plt.show()
        else:
            plt.close()
            
        return fig
    
    def _plot_accuracy_matrix_on_axis(self, cl_metrics: ContinualLearningMetrics, ax: plt.Axes):
        """Plot accuracy matrix on a specific axis."""
        matrix = np.array(cl_metrics.accuracy_matrix)
        task_names = cl_metrics.task_names
        
        if len(matrix) == 0 or len(task_names) == 0:
            ax.text(0.5, 0.5, "No data to visualize", ha="center", va="center")
            return
            
        im = ax.imshow(matrix, cmap="YlGnBu")
        
        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("Accuracy", rotation=-90, va="bottom")
        
        # Set labels and ticks
        ax.set_xticks(np.arange(len(task_names)))
        ax.set_yticks(np.arange(len(matrix)))
        ax.set_xticklabels(task_names)
        ax.set_yticklabels([f"After Task {i+1}" for i in range(len(matrix))])
        
        # Rotate tick labels and set alignment
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add text annotations (if not too many)
        if len(task_names) <= 10 and len(matrix) <= 10:
            for i in range(len(matrix)):
                for j in range(len(task_names)):
                    if j < len(matrix[i]) and not np.isnan(matrix[i][j]):
                        ax.text(j, i, f"{matrix[i][j]:.2f}", ha="center", va="center", 
                                color="black" if matrix[i][j] > 0.5 else "white")
        
        ax.set_title("Accuracy Matrix")
    
    def _plot_task_accuracies_on_axis(self, cl_metrics: ContinualLearningMetrics, ax: plt.Axes):
        """Plot task accuracies over time on a specific axis."""
        matrix = np.array(cl_metrics.accuracy_matrix)
        task_names = cl_metrics.task_names
        
        if len(matrix) == 0 or len(task_names) == 0:
            ax.text(0.5, 0.5, "No data to visualize", ha="center", va="center")
            return
            
        # Convert to full matrix with NaN for missing values
        full_matrix = np.zeros((len(matrix), len(task_names))) * np.nan
        for i, row in enumerate(matrix):
            for j, val in enumerate(row):
                if j < len(row):
                    full_matrix[i, j] = val
        
        # Plot line for each task
        x = np.arange(1, len(matrix) + 1)  # Task numbers
        for j, task_name in enumerate(task_names):
            task_accuracies = full_matrix[:, j]
            ax.plot(x, task_accuracies, marker='o', label=task_name)
            
        # Add markers for after-trained-on points
        for j, task_name in enumerate(task_names):
            if j < len(matrix):  # Only for tasks we've trained on
                if j < full_matrix.shape[1] and j < len(x):
                    ax.plot(x[j], full_matrix[j, j], 'ro', markersize=10, markeredgecolor='black')
        
        # Set labels and title
        ax.set_xlabel("Tasks Trained")
        ax.set_ylabel("Accuracy")
        ax.set_title("Task Accuracies Over Time")
        ax.set_xticks(x)
        ax.set_xticklabels([f"After Task {i+1}" for i in range(len(matrix))])
        ax.set_ylim(0, 1.05)
        
        # Add legend
        ax.legend(loc="lower right")
        
        # Add grid
        ax.grid(True, alpha=0.3)
    
    def _plot_forgetting_on_axis(self, cl_metrics: ContinualLearningMetrics, ax: plt.Axes):
        """Plot forgetting measures on a specific axis."""
        task_names = cl_metrics.task_names[:-1]  # Exclude the last task (no forgetting)
        
        if len(task_names) == 0:
            ax.text(0.5, 0.5, "Not enough tasks to visualize forgetting", ha="center", va="center")
            return
            
        forgetting_values = [cl_metrics.forgetting(i) for i in range(len(task_names))]
        
        bars = ax.bar(task_names, forgetting_values, color=sns.color_palette("muted"))
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.2f}",
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha="center", va="bottom")
        
        # Set labels and title
        ax.set_xlabel("Task")
        ax.set_ylabel("Forgetting Measure")
        ax.set_title("Forgetting Across Tasks")
        
        # Add horizontal line at y=0
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    
    def _plot_final_accuracies_on_axis(self, cl_metrics: ContinualLearningMetrics, ax: plt.Axes):
        """Plot final task accuracies on a specific axis."""
        if not cl_metrics.accuracy_matrix:
            ax.text(0.5, 0.5, "No data to visualize", ha="center", va="center")
            return
            
        final_accuracies = cl_metrics.accuracy_matrix[-1]
        task_names = cl_metrics.task_names
        
        bars = ax.bar(task_names, final_accuracies, color=sns.color_palette("deep"))
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.2f}",
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha="center", va="bottom")
        
        # Set labels and title
        ax.set_xlabel("Task")
        ax.set_ylabel("Accuracy")
        ax.set_title("Final Task Accuracies")
        
        # Set y-axis limits
        ax.set_ylim(0, 1.05)
    
    def _plot_transfer_on_axis(self, cl_metrics: ContinualLearningMetrics, ax: plt.Axes):
        """Plot backward and forward transfer on a specific axis."""
        task_names = cl_metrics.task_names
        
        if len(task_names) <= 1:
            ax.text(0.5, 0.5, "Not enough tasks to visualize transfer", ha="center", va="center")
            return
            
        # Backward transfer (excluding the last task)
        bt_tasks = task_names[:-1]
        bt_values = [cl_metrics.backward_transfer(i) for i in range(len(bt_tasks))]
        
        # Forward transfer (excluding the first task)
        ft_tasks = task_names[1:]
        ft_values = [cl_metrics.forward_transfer(i+1) for i in range(len(ft_tasks))]
        
        # Calculate positions for grouped bars
        x = np.arange(len(task_names))
        width = 0.35
        
        # Plot backward transfer (for all tasks except the last)
        bt_bars = ax.bar(x[:-1] - width/2, bt_values, width, label='Backward Transfer',
                        color=sns.color_palette("muted")[0])
        
        # Plot forward transfer (for all tasks except the first)
        ft_bars = ax.bar(x[1:] + width/2, ft_values, width, label='Forward Transfer',
                        color=sns.color_palette("muted")[1])
        
        # Add value labels
        for bars in [bt_bars, ft_bars]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f"{height:.2f}",
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3 if height >= 0 else -15),
                            textcoords="offset points",
                            ha="center", va="bottom" if height >= 0 else "top")
        
        # Set labels and title
        ax.set_xlabel("Task")
        ax.set_ylabel("Transfer Measure")
        ax.set_title("Knowledge Transfer Between Tasks")
        ax.set_xticks(x)
        ax.set_xticklabels(task_names)
        
        # Add legend and horizontal line at y=0
        ax.legend(loc="upper right")
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        
        # Add grid
        ax.grid(True, alpha=0.3)
    
    def _plot_confusion_matrix_on_axis(self, 
                                     matrix: np.ndarray, 
                                     class_names: List[str], 
                                     ax: plt.Axes,
                                     title: str = "Confusion Matrix"):
        """Plot confusion matrix on a specific axis."""
        # Normalize the matrix
        matrix_normalized = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
        matrix_normalized = np.nan_to_num(matrix_normalized)  # Replace NaNs with 0
        
        im = ax.imshow(matrix_normalized, interpolation='nearest', cmap=plt.cm.Blues)
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("Normalized Count", rotation=-90, va="bottom")
        
        # Set labels and title
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(title)
        
        # Add ticks and labels
        tick_marks = np.arange(len(class_names))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticklabels(class_names)
        
        # Add text annotations (only for matrices that aren't too large)
        if len(class_names) <= 20:
            fmt = '.2f' if matrix_normalized.max() < 0.1 else '.2f'
            thresh = matrix_normalized.max() / 2.
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    ax.text(j, i, format(matrix_normalized[i, j], fmt) if matrix_normalized[i, j] > 0 else '',
                            ha="center", va="center",
                            color="white" if matrix_normalized[i, j] > thresh else "black") 
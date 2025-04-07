import os
import torch
import numpy as np
import logging
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from datetime import datetime
import json
import pandas as pd
from dataclasses import dataclass
import yaml
from tqdm import tqdm

from ml.training.continual import ContinualTrainer
from ml.utils.data import create_permuted_mnist_tasks, create_split_mnist_tasks
from ml.models.simple import SimpleMLP, SimpleCNN
from ml.models.vae import VAE, ConvVAE

logger = logging.getLogger(__name__)

@dataclass
class BenchmarkResult:
    """Class for storing benchmark results."""
    strategy_name: str
    dataset_name: str
    num_tasks: int
    task_accuracies: np.ndarray  # Shape: [num_tasks, num_tasks]
    forgetting: np.ndarray  # Shape: [num_tasks-1]
    avg_accuracy: float
    avg_forgetting: float
    training_time: float
    memory_usage: int
    config: Dict

    def to_dict(self) -> Dict:
        """Convert result to dictionary for serialization."""
        return {
            "strategy_name": self.strategy_name,
            "dataset_name": self.dataset_name,
            "num_tasks": self.num_tasks,
            "task_accuracies": self.task_accuracies.tolist(),
            "forgetting": self.forgetting.tolist(),
            "avg_accuracy": self.avg_accuracy,
            "avg_forgetting": self.avg_forgetting,
            "training_time": self.training_time,
            "memory_usage": self.memory_usage,
            "config": self.config
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'BenchmarkResult':
        """Create object from dictionary."""
        return cls(
            strategy_name=data["strategy_name"],
            dataset_name=data["dataset_name"],
            num_tasks=data["num_tasks"],
            task_accuracies=np.array(data["task_accuracies"]),
            forgetting=np.array(data["forgetting"]),
            avg_accuracy=data["avg_accuracy"],
            avg_forgetting=data["avg_forgetting"],
            training_time=data["training_time"],
            memory_usage=data["memory_usage"],
            config=data["config"]
        )


class ContinualLearningBenchmark:
    """
    Benchmark different continual learning strategies.
    
    This class provides functionality to:
    1. Run benchmarks on different continual learning strategies
    2. Compare results across strategies
    3. Visualize performance metrics
    4. Save and load benchmark results
    """
    
    def __init__(
        self,
        results_dir: str = "results/benchmarks",
        configs_dir: str = "configs/continual_learning",
        device: torch.device = None
    ):
        """
        Initialize the benchmark framework.
        
        Args:
            results_dir: Directory to save benchmark results
            configs_dir: Directory containing strategy configurations
            device: Device to run benchmarks on
        """
        self.results_dir = Path(results_dir)
        self.configs_dir = Path(configs_dir)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results = {}  # Maps strategy_name to BenchmarkResult
        
        # Create results directory
        os.makedirs(self.results_dir, exist_ok=True)
        
        logger.info(f"Initialized ContinualLearningBenchmark on {self.device}")
    
    def run_benchmark(
        self,
        strategy_name: str,
        config_path: str,
        dataset: str = "permuted_mnist",
        num_tasks: int = 5,
        batch_size: int = 64,
        seed: int = 42,
        save_results: bool = True
    ) -> BenchmarkResult:
        """
        Run benchmark for a specific strategy.
        
        Args:
            strategy_name: Name of the strategy (e.g., "ewc", "lwf")
            config_path: Path to strategy configuration file
            dataset: Dataset to use ("permuted_mnist" or "split_mnist")
            num_tasks: Number of tasks to train on
            batch_size: Batch size for training
            seed: Random seed for reproducibility
            save_results: Whether to save results to disk
            
        Returns:
            BenchmarkResult object with performance metrics
        """
        # Set random seeds for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        
        # Load configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Create results directory for this run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.results_dir / f"{strategy_name}_{dataset}_{timestamp}"
        os.makedirs(run_dir, exist_ok=True)
        
        # Save configuration
        with open(run_dir / "config.yaml", 'w') as f:
            yaml.dump(config, f)
        
        # Create dataset
        if dataset == "permuted_mnist":
            task_data = create_permuted_mnist_tasks(
                num_tasks=num_tasks,
                batch_size=batch_size,
                data_dir="./data"
            )
            input_dim = 784  # Flattened MNIST
            output_dim = 10  # Digits 0-9
        elif dataset == "split_mnist":
            task_data = create_split_mnist_tasks(
                batch_size=batch_size,
                data_dir="./data"
            )
            input_dim = 784  # Flattened MNIST
            output_dim = 2   # Binary classification
            num_tasks = 5    # Fixed for split_mnist (5 pairs)
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
        
        task_loaders, task_names = task_data
        
        # Create model based on config
        model_config = config.get('model', {})
        model_type = model_config.get('type', 'mlp')
        
        if model_type == 'mlp':
            hidden_sizes = model_config.get('hidden_sizes', [256, 128])
            dropout = model_config.get('dropout', 0.2)
            
            model = SimpleMLP(
                input_size=input_dim,
                hidden_sizes=hidden_sizes,
                output_size=output_dim,
                dropout=dropout
            )
        elif model_type == 'cnn':
            channels = model_config.get('channels', 1)
            hidden_dims = model_config.get('hidden_dims', [16, 32])
            
            model = SimpleCNN(
                num_channels=channels,
                hidden_dims=hidden_dims,
                output_dim=output_dim
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model.to(self.device)
        
        # Create trainer
        trainer_config = config.copy()
        trainer_config['checkpoint_dir'] = str(run_dir / "checkpoints")
        
        trainer = ContinualTrainer(
            model=model,
            config=trainer_config,
            device=self.device,
            experiment_name=f"{strategy_name}_{dataset}_benchmark"
        )
        
        # Store metrics
        task_accuracies = np.zeros((num_tasks, num_tasks))  # [task_id, after_learning_task]
        forgetting = np.zeros(num_tasks - 1)
        best_acc_per_task = np.zeros(num_tasks)
        
        # Track training time
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        start_time.record()
        
        # Train on each task sequentially
        for task_id, (train_loader, val_loader) in enumerate(task_loaders):
            logger.info(f"Training on task {task_id+1}/{num_tasks}: {task_names[task_id]}")
            
            # Train on current task
            trainer.train_task(
                train_loader=train_loader,
                val_loader=val_loader,
                task_id=task_id,
                task_name=task_names[task_id],
                num_epochs=config['training']['epochs']
            )
            
            # Evaluate on all seen tasks
            logger.info(f"Evaluating on all seen tasks after learning task {task_id+1}...")
            
            task_forgetting = 0.0
            
            for eval_task_id in range(task_id + 1):
                _, eval_val_loader = task_loaders[eval_task_id]
                
                # Evaluate on this task
                eval_metrics = self._evaluate(
                    trainer, 
                    eval_val_loader, 
                    task_id=eval_task_id
                )
                
                current_acc = eval_metrics['accuracy']
                task_accuracies[eval_task_id, task_id] = current_acc
                
                # Update best accuracy for this task
                if task_id == eval_task_id or current_acc > best_acc_per_task[eval_task_id]:
                    best_acc_per_task[eval_task_id] = current_acc
                
                # Calculate forgetting for previously learned tasks
                if eval_task_id < task_id:
                    # Forgetting is the difference between the best accuracy so far
                    # and current accuracy
                    forgetting_task = max(0, best_acc_per_task[eval_task_id] - current_acc)
                    task_forgetting += forgetting_task
                    
                    logger.info(f"Task {eval_task_id+1} accuracy: {current_acc:.4f}, forgetting: {forgetting_task:.4f}")
                else:
                    logger.info(f"Task {eval_task_id+1} accuracy: {current_acc:.4f}")
            
            # Calculate average forgetting over all previous tasks
            if task_id > 0:
                avg_forgetting = task_forgetting / task_id
                forgetting[task_id - 1] = avg_forgetting
                logger.info(f"Average forgetting after task {task_id+1}: {avg_forgetting:.4f}")
        
        # Record end time
        end_time.record()
        torch.cuda.synchronize()
        training_time = start_time.elapsed_time(end_time) / 1000.0  # Convert to seconds
        
        # Estimate memory usage (rough approximation)
        memory_usage = 0
        for param in model.parameters():
            memory_usage += param.numel() * param.element_size()
        
        # Create result object
        avg_accuracy = np.mean(np.diag(task_accuracies))
        avg_forgetting = np.mean(forgetting) if len(forgetting) > 0 else 0.0
        
        result = BenchmarkResult(
            strategy_name=strategy_name,
            dataset_name=dataset,
            num_tasks=num_tasks,
            task_accuracies=task_accuracies,
            forgetting=forgetting,
            avg_accuracy=float(avg_accuracy),
            avg_forgetting=float(avg_forgetting),
            training_time=float(training_time),
            memory_usage=int(memory_usage),
            config=config
        )
        
        # Save result
        if save_results:
            result_path = run_dir / "result.json"
            with open(result_path, 'w') as f:
                json.dump(result.to_dict(), f, indent=2)
                
            # Create plots
            self._plot_task_accuracies(result, save_path=run_dir / "task_accuracies.png")
            self._plot_forgetting(result, save_path=run_dir / "forgetting.png")
        
        # Store result
        self.results[strategy_name] = result
        
        return result
    
    def _evaluate(
        self, 
        trainer: ContinualTrainer, 
        dataloader: torch.utils.data.DataLoader, 
        task_id: int
    ) -> Dict[str, float]:
        """
        Evaluate the model on a specific task.
        
        Args:
            trainer: ContinualTrainer instance
            dataloader: DataLoader for evaluation
            task_id: ID of the task to evaluate on
            
        Returns:
            Dictionary with metrics (accuracy, loss)
        """
        loss_fn = torch.nn.CrossEntropyLoss()
        return trainer._validate(dataloader, task_id, f"task_{task_id}", loss_fn)
    
    def run_all_strategies(
        self,
        dataset: str = "permuted_mnist",
        num_tasks: int = 5,
        strategies: Optional[List[str]] = None,
        batch_size: int = 64,
        seed: int = 42
    ) -> Dict[str, BenchmarkResult]:
        """
        Run benchmark for all available strategies.
        
        Args:
            dataset: Dataset to use ("permuted_mnist" or "split_mnist")
            num_tasks: Number of tasks to train on
            strategies: List of strategy names to benchmark (if None, run all)
            batch_size: Batch size for training
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary mapping strategy names to benchmark results
        """
        # Find available strategy configurations
        if strategies is None:
            config_files = list(self.configs_dir.glob("*.yaml"))
            strategies = [config.stem for config in config_files]
        
        results = {}
        
        for strategy in strategies:
            config_path = self.configs_dir / f"{strategy}_config.yaml"
            if not config_path.exists():
                logger.warning(f"Configuration file not found for strategy: {strategy}")
                continue
                
            logger.info(f"Running benchmark for strategy: {strategy}")
            try:
                result = self.run_benchmark(
                    strategy_name=strategy,
                    config_path=str(config_path),
                    dataset=dataset,
                    num_tasks=num_tasks,
                    batch_size=batch_size,
                    seed=seed
                )
                results[strategy] = result
            except Exception as e:
                logger.error(f"Error benchmarking strategy {strategy}: {e}")
        
        return results
    
    def compare_strategies(
        self,
        results: Optional[Dict[str, BenchmarkResult]] = None,
        metrics: List[str] = ["avg_accuracy", "avg_forgetting", "training_time", "memory_usage"],
        sort_by: str = "avg_accuracy"
    ) -> pd.DataFrame:
        """
        Compare results across strategies.
        
        Args:
            results: Dictionary mapping strategy names to benchmark results
            metrics: List of metrics to compare
            sort_by: Metric to sort results by
            
        Returns:
            Pandas DataFrame with strategy comparison
        """
        results = results or self.results
        
        data = []
        for strategy_name, result in results.items():
            row = {"strategy": strategy_name}
            for metric in metrics:
                if hasattr(result, metric):
                    row[metric] = getattr(result, metric)
            data.append(row)
        
        df = pd.DataFrame(data)
        if sort_by in df.columns:
            if sort_by == "avg_forgetting" or sort_by == "training_time" or sort_by == "memory_usage":
                df = df.sort_values(by=sort_by, ascending=True)
            else:
                df = df.sort_values(by=sort_by, ascending=False)
                
        return df
    
    def _plot_task_accuracies(
        self, 
        result: BenchmarkResult, 
        save_path: Optional[str] = None,
        show: bool = False
    ):
        """
        Plot task accuracies matrix.
        
        Args:
            result: BenchmarkResult to plot
            save_path: Path to save plot (if None, don't save)
            show: Whether to display the plot
        """
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Create heatmap
        im = plt.imshow(result.task_accuracies, cmap='viridis', vmin=0, vmax=1)
        
        # Add colorbar
        plt.colorbar(im, label='Accuracy')
        
        # Add labels
        plt.xlabel('After learning task')
        plt.ylabel('Task')
        plt.title(f'{result.strategy_name} - Task Accuracies ({result.dataset_name})')
        
        # Configure axis ticks
        plt.xticks(np.arange(result.num_tasks), np.arange(1, result.num_tasks + 1))
        plt.yticks(np.arange(result.num_tasks), np.arange(1, result.num_tasks + 1))
        
        # Add text annotations
        for i in range(result.num_tasks):
            for j in range(result.num_tasks):
                if j >= i:  # Only for tasks that have been trained
                    plt.text(j, i, f"{result.task_accuracies[i, j]:.2f}", 
                             ha="center", va="center", 
                             color="white" if result.task_accuracies[i, j] < 0.7 else "black")
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=150)
            
        if show:
            plt.show()
        else:
            plt.close()
    
    def _plot_forgetting(
        self, 
        result: BenchmarkResult, 
        save_path: Optional[str] = None,
        show: bool = False
    ):
        """
        Plot forgetting over tasks.
        
        Args:
            result: BenchmarkResult to plot
            save_path: Path to save plot (if None, don't save)
            show: Whether to display the plot
        """
        if len(result.forgetting) == 0:
            return
            
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Plot forgetting
        x = np.arange(2, result.num_tasks + 1)  # Tasks 2 to num_tasks
        plt.plot(x, result.forgetting, marker='o', linestyle='-', linewidth=2)
        
        # Add labels
        plt.xlabel('After learning task')
        plt.ylabel('Average Forgetting')
        plt.title(f'{result.strategy_name} - Forgetting ({result.dataset_name})')
        
        # Configure axis
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(x)
        
        # Add text annotations
        for i, val in enumerate(result.forgetting):
            plt.text(x[i], val + 0.01, f"{val:.3f}", ha='center')
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=150)
            
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_comparison(
        self,
        results: Optional[Dict[str, BenchmarkResult]] = None,
        metric: str = "avg_accuracy",
        save_path: Optional[str] = None,
        show: bool = False
    ):
        """
        Plot comparison of a specific metric across strategies.
        
        Args:
            results: Dictionary mapping strategy names to benchmark results
            metric: Metric to compare
            save_path: Path to save plot (if None, don't save)
            show: Whether to display the plot
        """
        results = results or self.results
        
        # Extract data
        strategies = []
        values = []
        
        for strategy_name, result in results.items():
            if hasattr(result, metric):
                strategies.append(strategy_name)
                values.append(getattr(result, metric))
        
        if not strategies:
            logger.warning(f"No data available for metric: {metric}")
            return
            
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Sort by metric value
        sort_asc = metric in ["avg_forgetting", "training_time", "memory_usage"]
        sorted_indices = np.argsort(values)
        if not sort_asc:
            sorted_indices = sorted_indices[::-1]
            
        strategies = [strategies[i] for i in sorted_indices]
        values = [values[i] for i in sorted_indices]
        
        # Create bar chart
        bars = plt.bar(strategies, values, width=0.6)
        
        # Add labels
        plt.xlabel('Strategy')
        ylabel = ' '.join(word.capitalize() for word in metric.split('_'))
        plt.ylabel(ylabel)
        plt.title(f'Comparison of {ylabel} Across Strategies')
        
        # Add values on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01 * max(values),
                     f"{height:.3f}", ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=150)
            
        if show:
            plt.show()
        else:
            plt.close()
    
    def load_results(self, results_dir: Optional[str] = None) -> Dict[str, BenchmarkResult]:
        """
        Load benchmark results from disk.
        
        Args:
            results_dir: Directory containing benchmark results
            
        Returns:
            Dictionary mapping strategy names to benchmark results
        """
        results_dir = Path(results_dir or self.results_dir)
        
        results = {}
        
        # Find all result directories
        for run_dir in results_dir.glob("**/result.json"):
            try:
                with open(run_dir, 'r') as f:
                    data = json.load(f)
                    
                result = BenchmarkResult.from_dict(data)
                results[result.strategy_name] = result
                
                logger.info(f"Loaded result for strategy: {result.strategy_name}")
            except Exception as e:
                logger.error(f"Error loading result from {run_dir}: {e}")
        
        self.results.update(results)
        return results
    
    def save_comparison(
        self,
        df: pd.DataFrame,
        save_path: Optional[str] = None
    ):
        """
        Save strategy comparison to disk.
        
        Args:
            df: DataFrame with strategy comparison
            save_path: Path to save comparison (if None, use default)
        """
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = self.results_dir / f"comparison_{timestamp}.csv"
            
        df.to_csv(save_path, index=False)
        logger.info(f"Saved comparison to {save_path}")


def run_benchmark_suite():
    """Run the complete benchmark suite on all strategies."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create benchmark
    benchmark = ContinualLearningBenchmark(
        results_dir="results/benchmarks",
        configs_dir="configs/continual_learning"
    )
    
    # List of strategies to benchmark
    strategies = [
        "ewc",
        "lwf",
        "gem",
        "packnet",
        "er_plus",
        "generative_replay"
    ]
    
    # Run benchmarks
    results = benchmark.run_all_strategies(
        dataset="permuted_mnist",
        num_tasks=5,
        strategies=strategies,
        batch_size=64,
        seed=42
    )
    
    # Compare strategies
    comparison = benchmark.compare_strategies(
        results=results,
        metrics=["avg_accuracy", "avg_forgetting", "training_time", "memory_usage"],
        sort_by="avg_accuracy"
    )
    
    # Save comparison
    benchmark.save_comparison(comparison)
    
    # Create summary plots
    benchmark.plot_comparison(
        results=results,
        metric="avg_accuracy",
        save_path="results/benchmarks/accuracy_comparison.png",
        show=False
    )
    
    benchmark.plot_comparison(
        results=results,
        metric="avg_forgetting",
        save_path="results/benchmarks/forgetting_comparison.png",
        show=False
    )
    
    return results, comparison


if __name__ == "__main__":
    results, comparison = run_benchmark_suite()
    print(comparison) 
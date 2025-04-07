#!/usr/bin/env python
"""
Continual Learning Benchmark Runner

This script runs benchmarks to compare different continual learning strategies
on standard datasets (permuted MNIST and split MNIST). It generates visualizations
and performance comparisons.
"""

import os
import sys
import logging
import argparse
import yaml
import torch
import pandas as pd
from pathlib import Path

# Add the parent directory to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.evaluation.benchmarks import ContinualLearningBenchmark, run_benchmark_suite

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run continual learning benchmarks')
    parser.add_argument(
        '--configs-dir', type=str, 
        default='configs/continual_learning',
        help='Directory containing strategy configurations'
    )
    parser.add_argument(
        '--results-dir', type=str, 
        default='results/benchmarks',
        help='Directory to save benchmark results'
    )
    parser.add_argument(
        '--dataset', type=str, 
        default='permuted_mnist',
        choices=['permuted_mnist', 'split_mnist'],
        help='Dataset to use for benchmarking'
    )
    parser.add_argument(
        '--num-tasks', type=int, 
        default=5,
        help='Number of tasks (only for permuted_mnist)'
    )
    parser.add_argument(
        '--batch-size', type=int, 
        default=64,
        help='Batch size for training'
    )
    parser.add_argument(
        '--seed', type=int, 
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--strategies', type=str, 
        nargs='+',
        default=None,
        help='List of strategies to benchmark (if not specified, benchmark all)'
    )
    parser.add_argument(
        '--no-plots', action='store_true',
        help='Disable generation of plots'
    )
    parser.add_argument(
        '--load-existing', action='store_true',
        help='Load existing results instead of running new benchmarks'
    )
    return parser.parse_args()

def print_summary(comparison_df):
    """Print a summary of benchmark results."""
    print("\n" + "="*80)
    print(" CONTINUAL LEARNING STRATEGIES BENCHMARK SUMMARY ")
    print("="*80)
    
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.float_format', '{:.4f}'.format)
    
    print(comparison_df)
    print("\n" + "="*80)
    
    # Print best strategy for each metric
    print("BEST STRATEGIES BY METRIC:")
    print("-"*80)
    
    metrics = comparison_df.columns.tolist()
    metrics.remove('strategy')
    
    for metric in metrics:
        if metric == 'avg_forgetting' or metric == 'training_time' or metric == 'memory_usage':
            best_strategy = comparison_df.loc[comparison_df[metric].idxmin()]
        else:
            best_strategy = comparison_df.loc[comparison_df[metric].idxmax()]
        
        print(f"Best for {metric}: {best_strategy['strategy']} ({best_strategy[metric]:.4f})")
    
    print("="*80 + "\n")

def main():
    """Main function."""
    args = parse_args()
    
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    if args.load_existing:
        # Load existing results
        logger.info("Loading existing benchmark results...")
        benchmark = ContinualLearningBenchmark(
            results_dir=args.results_dir,
            configs_dir=args.configs_dir
        )
        results = benchmark.load_results()
        
        if not results:
            logger.error("No existing results found. Please run benchmarks first.")
            return
    else:
        # Run benchmark suite
        logger.info("Running benchmark suite...")
        
        benchmark = ContinualLearningBenchmark(
            results_dir=args.results_dir,
            configs_dir=args.configs_dir
        )
        
        # Run benchmarks
        results = benchmark.run_all_strategies(
            dataset=args.dataset,
            num_tasks=args.num_tasks,
            strategies=args.strategies,
            batch_size=args.batch_size,
            seed=args.seed
        )
    
    # Compare strategies
    logger.info("Comparing strategies...")
    comparison = benchmark.compare_strategies(
        results=results,
        metrics=["avg_accuracy", "avg_forgetting", "training_time", "memory_usage"],
        sort_by="avg_accuracy"
    )
    
    # Save comparison
    comparison_path = os.path.join(args.results_dir, "comparison.csv")
    benchmark.save_comparison(comparison, comparison_path)
    logger.info(f"Saved comparison to {comparison_path}")
    
    # Print summary
    print_summary(comparison)
    
    # Create plots if not disabled
    if not args.no_plots:
        logger.info("Generating plots...")
        
        # Directory for plots
        plots_dir = os.path.join(args.results_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Create comparison plots for key metrics
        metrics = [
            "avg_accuracy", 
            "avg_forgetting", 
            "training_time", 
            "memory_usage"
        ]
        
        for metric in metrics:
            plot_path = os.path.join(plots_dir, f"{metric}_comparison.png")
            benchmark.plot_comparison(
                results=results,
                metric=metric,
                save_path=plot_path,
                show=False
            )
            logger.info(f"Created plot: {plot_path}")
        
        # For each strategy, create task accuracy and forgetting plots
        for strategy_name, result in results.items():
            strategy_dir = os.path.join(plots_dir, strategy_name)
            os.makedirs(strategy_dir, exist_ok=True)
            
            # Task accuracies
            benchmark._plot_task_accuracies(
                result,
                save_path=os.path.join(strategy_dir, "task_accuracies.png"),
                show=False
            )
            
            # Forgetting
            benchmark._plot_forgetting(
                result,
                save_path=os.path.join(strategy_dir, "forgetting.png"),
                show=False
            )
            
            logger.info(f"Created plots for {strategy_name}")
    
    logger.info("Benchmark completed successfully!")
    logger.info(f"Results saved to {args.results_dir}")

if __name__ == "__main__":
    main() 
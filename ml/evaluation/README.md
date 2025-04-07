# Evaluation Module

This module provides tools and utilities for evaluating and benchmarking continual learning strategies.

## Components

### Benchmarks (`benchmarks.py`)

The `benchmarks.py` module implements a comprehensive benchmarking framework for continual learning strategies. Key features include:

- Standardized evaluation on common datasets (Permuted MNIST, Split MNIST)
- Consistent measurement of key metrics (accuracy, forgetting, training time, memory usage)
- Visualization tools for performance comparison
- Support for serialization and loading of benchmark results

## Usage

For detailed usage instructions, see the [Benchmarking Documentation](../../docs/benchmarking.md).

### Quick Start

```python
from ml.evaluation.benchmarks import ContinualLearningBenchmark

# Create benchmark
benchmark = ContinualLearningBenchmark()

# Run benchmark for a specific strategy
result = benchmark.run_benchmark(
    strategy_name="ewc",
    config_path="configs/continual_learning/ewc_config.yaml",
    dataset="permuted_mnist",
    num_tasks=5
)

# Or run all strategies and compare
results = benchmark.run_all_strategies(dataset="permuted_mnist", num_tasks=5)
comparison = benchmark.compare_strategies(results)
```

## Command-Line Interface

The module can be used from the command line via:

```bash
python examples/run_benchmarks.py
```

## Future Extensions

Planned extensions to the evaluation module include:

1. Support for additional datasets (e.g., CIFAR-100, Mini-ImageNet)
2. Additional metrics (e.g., forward transfer, parameter efficiency)
3. Statistical significance testing
4. Model-based metrics (e.g., weight distance, representation similarity)
5. Support for continual learning in different domains (RL, NLP) 
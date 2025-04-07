# Continual Learning Strategies

This document provides an overview of the continual learning strategies implemented in this framework. Each strategy aims to mitigate catastrophic forgetting when training neural networks on a sequence of tasks.

## Available Strategies

1. [Elastic Weight Consolidation (EWC)](#elastic-weight-consolidation)
2. [Learning without Forgetting (LwF)](#learning-without-forgetting)
3. [Gradient Episodic Memory (GEM)](#gradient-episodic-memory)
4. [Progressive Neural Networks (PNN)](#progressive-neural-networks)
5. [PackNet](#packnet)
6. [Experience Replay with Regularization (ER+)](#experience-replay-with-regularization)
7. [Generative Replay (GR)](#generative-replay)

## Elastic Weight Consolidation

EWC prevents catastrophic forgetting by adding a regularization term to the loss function that penalizes changes to parameters that are important for previous tasks.

### How it works

1. After training on a task, compute the Fisher Information Matrix, which represents how sensitive the loss is to changes in each parameter.
2. When training on a new task, add a penalty to the loss function proportional to the squared distance from the original parameters, weighted by the Fisher Information Matrix.

### Configuration

```yaml
continual_strategy:
  ewc:
    enabled: true
    lambda: 5000.0  # Strength of the EWC penalty
    gamma: 1.0      # Decay factor for Fisher matrix across tasks
```

### Implementation

See [EWC Implementation](strategies/ewc.md) for details.

## Learning without Forgetting

LwF uses knowledge distillation to preserve performance on previous tasks without requiring access to their data.

### How it works

1. Before training on a new task, record the model's current outputs on the new task's data.
2. During training, include a distillation loss that penalizes changes to these recorded outputs.

### Configuration

```yaml
continual_strategy:
  lwf:
    enabled: true
    alpha: 1.0        # Weight for the distillation loss
    temperature: 2.0  # Temperature for softening logits
```

### Implementation

See [LwF Implementation](strategies/lwf.md) for details.

## Gradient Episodic Memory

GEM uses episodic memory to store a subset of samples from previous tasks and ensures that gradient updates on new tasks don't increase the loss on these stored samples.

### How it works

1. Maintain a memory buffer with samples from previous tasks.
2. Before updating the model parameters, check if the gradient from the current batch would increase the loss on the memory buffer.
3. If it would, project the gradient to a direction that doesn't increase the loss on previous tasks.

### Configuration

```yaml
continual_strategy:
  gem:
    enabled: true
    memory_size: 200     # Total capacity of memory buffer
    samples_per_task: 50  # Samples to store per task
    margin: 0.5          # Margin parameter for gradient projection
```

### Implementation

See [GEM Implementation](strategies/gem.md) for details.

## Progressive Neural Networks

PNN creates a new neural network for each task while preserving lateral connections to previously learned features, allowing knowledge transfer without interference.

### How it works

1. Start with a base network for the first task.
2. For each new task, create a new network but add lateral connections from all previous networks.
3. During training on a new task, only update the parameters of the new network and the lateral connections; previous networks remain frozen.

### Configuration

```yaml
continual_strategy:
  pnn:
    enabled: true
    input_size: 784
    hidden_sizes: [256, 128]
    output_size: 10
    lateral_connections: true  # Whether to use lateral connections
```

### Implementation

See [PNN Implementation](strategies/pnn.md) for details.

## PackNet

PackNet uses iterative pruning and weight freezing to allocate different subnetworks for different tasks within a single network architecture.

### How it works

1. Train the network on the first task.
2. Prune a percentage of the weights (e.g., 75%) that are least important.
3. Freeze the remaining weights to preserve performance on the first task.
4. For each new task, train using only the previously pruned (free) weights.
5. After training, prune and freeze again to allocate capacity for the next task.

### Configuration

```yaml
continual_strategy:
  packnet:
    enabled: true
    prune_percentage: 0.75  # Percentage of weights to prune after each task
    prune_threshold: 0.001  # Minimum magnitude for a weight to be kept
    use_magnitude_pruning: true  # Whether to use magnitude-based pruning
```

### Implementation

See [PackNet Implementation](strategies/packnet.md) for details.

## Experience Replay with Regularization

ER+ combines experience replay with knowledge distillation regularization to prevent catastrophic forgetting.

### How it works

1. Maintain a replay buffer of samples from previous tasks.
2. During training on a new task, mix batches from the current task with samples from the replay buffer.
3. Apply a knowledge distillation regularization term to maintain consistency with previous model outputs.

### Configuration

```yaml
continual_strategy:
  er_plus:
    enabled: true
    memory_size: 500     # Size of replay buffer
    batch_size: 32       # Batch size for sampling from buffer
    reg_weight: 1.0      # Weight for the regularization term
    temperature: 2.0     # Temperature for softening logits
    task_balanced: true  # Whether to balance tasks in the buffer
```

### Implementation

See [ER+ Implementation](strategies/er_plus.md) for details.

## Generative Replay

Generative Replay uses a generative model (e.g., VAE or GAN) to create synthetic examples of previous tasks, avoiding the need to store real data samples.

### How it works

1. Train a generative model alongside the main model to capture the data distribution of each task.
2. Before training on a new task, use the generative model to create synthetic examples of previous tasks.
3. During training on the new task, include these synthetic examples to maintain performance on previous tasks.

### Configuration

```yaml
continual_strategy:
  generative_replay:
    enabled: true
    memory_size: 500         # Size of replay buffer for training the generator
    batch_size: 32           # Batch size for sampling from replay buffer
    replay_weight: 1.0       # Weight for the replay loss
    generator_update_freq: 5 # How often to update the generator
    task_balanced: true      # Whether to balance samples across tasks
```

### Implementation

See [Generative Replay Implementation](strategies/generative_replay.md) for details.

## Comparison of Strategies

| Strategy | Memory Requirements | Computational Overhead | Task Boundaries | Privacy-Preserving |
|----------|---------------------|------------------------|-----------------|-------------------|
| EWC | Low (parameters only) | Low | Required | Yes |
| LwF | Low (outputs only) | Low | Required | Partially |
| GEM | Medium (sample storage) | Medium | Required | No |
| PNN | High (network per task) | High | Required | Yes |
| PackNet | Low (mask per task) | Medium | Required | Yes |
| ER+ | Medium (sample storage) | Medium | Required | No |
| GR | Medium (generator model) | High | Required | Yes |

## Best Practices for Strategy Selection

1. **Limited memory**: Consider EWC, LwF, or PackNet
2. **Maximum performance**: Consider PNN or ER+
3. **Privacy concerns**: Consider EWC, PackNet, or Generative Replay
4. **Computational constraints**: Avoid PNN and Generative Replay
5. **Balanced approach**: Consider ER+ or Generative Replay 
# Continual Learning Strategies

This document outlines the different continual learning strategies implemented in our CLM framework.

## Overview

Continual Learning strategies help mitigate catastrophic forgetting when a neural network learns multiple tasks sequentially. The following strategies are supported:

## Elastic Weight Consolidation (EWC)

### Overview
Elastic Weight Consolidation (EWC) prevents catastrophic forgetting by adding a regularization term to the loss function that penalizes changes to parameters that are important for previous tasks.

### How it Works
EWC estimates the importance of each parameter for previous tasks by calculating the Fisher Information Matrix (FIM). Parameters with high Fisher values are considered more important and are penalized more heavily if they change during training on a new task.

### Mathematical Formulation
The EWC loss is formulated as:

$L(\theta) = L_B(\theta) + \sum_{i} \frac{\lambda}{2} F_i (\theta_i - \theta_{A,i})^2$

Where:
- $L_B(\theta)$ is the loss for the current task
- $\lambda$ is a hyperparameter that controls the strength of the regularization
- $F_i$ is the Fisher Information for parameter $i$
- $\theta_{A,i}$ is the optimal parameter value for the previous task(s)

### Usage in CLM Framework
```python
config = {
    "continual_strategy": {
        "ewc": {
            "enabled": True,
            "lambda": 100.0,  # Regularization strength
            "gamma": 0.9  # For computing running average of Fisher matrices
        }
    }
}
```

## Learning without Forgetting (LwF)

### Overview
Learning without Forgetting (LwF) preserves performance on previous tasks by using knowledge distillation to maintain the outputs of the model on the previous tasks.

### How it Works
Before training on a new task, LwF records the outputs of the current model on the new task's data. After training, it includes a distillation loss that encourages the updated model to produce similar outputs for the new task's data as the original model.

### Mathematical Formulation
The LwF loss function is:

$L(\theta) = \lambda_B L_B(\theta) + \lambda_A L_{distill}(\theta)$

Where:
- $L_B(\theta)$ is the loss for the current task
- $L_{distill}(\theta)$ is the distillation loss that measures the difference between the current and previous model outputs
- $\lambda_A$ and $\lambda_B$ are hyperparameters that control the balance between the current task and knowledge distillation

### Usage in CLM Framework
```python
config = {
    "continual_strategy": {
        "lwf": {
            "enabled": True,
            "alpha": 1.0,  # Weight for current task loss
            "temperature": 2.0  # Temperature for softening the distillation targets
        }
    }
}
```

## Gradient Episodic Memory (GEM)

### Overview
Gradient Episodic Memory (GEM) maintains a small episodic memory of examples from previous tasks and uses them to constrain the gradients of the current task to not increase the loss on past tasks.

### How it Works
GEM stores a subset of examples from each task. When training on a new task, it computes gradients both for the current task and for samples from the episodic memory. It then projects the current gradient to ensure it doesn't increase the loss on previous tasks.

### Mathematical Formulation
GEM solves the following optimization problem:

$\min_{g'} \frac{1}{2}||g - g'||^2 \quad \text{s.t.} \quad \langle g', g_i \rangle \geq 0 \quad \forall i \in \{1, \ldots, t-1\}$

Where:
- $g$ is the gradient of the current task
- $g'$ is the projected gradient that will be used for the update
- $g_i$ are the gradients computed on the episodic memory for task $i$

### Usage in CLM Framework
```python
config = {
    "continual_strategy": {
        "gem": {
            "enabled": True,
            "memory_size": 200,  # Total memory size across all tasks
            "sample_size": 10  # Number of samples to use for gradient constraint
        }
    }
}
```

## Progressive Neural Networks (PNN)

### Overview
Progressive Neural Networks (PNN) completely prevent forgetting by creating a new neural network for each task, while allowing knowledge transfer from previous tasks via lateral connections.

### How it Works
For each new task, PNN creates a new "column" (a neural network) that receives inputs from both the current task and from the hidden layers of previously trained columns. This ensures there is no interference with previous task parameters while allowing new tasks to leverage previously learned features.

### Mathematical Formulation
In a PNN with L layers and t task columns, the activation $h^{(t,l)}$ in layer $l$ for task $t$ is computed as:

$h^{(t,l)} = f\left(W^{(t,l)}h^{(t,l-1)} + \sum_{k=1}^{t-1} U^{(k \rightarrow t,l)}h^{(k,l-1)}\right)$

Where:
- $W^{(t,l)}$ are the weights within the column for task $t$ at layer $l$
- $U^{(k \rightarrow t,l)}$ are the lateral connection weights from task column $k$ to task column $t$ at layer $l$
- $f$ is a non-linear activation function

### Advantages and Limitations
**Advantages:**
- Completely prevents forgetting since parameters for previous tasks are never modified
- Enables effective knowledge transfer through lateral connections
- No need for replay memory or regularization terms

**Limitations:**
- Model size grows linearly with the number of tasks
- Requires knowing task identity at inference time
- May be computationally expensive for large models or many tasks

### Usage in CLM Framework
```python
config = {
    "continual_strategy": {
        "pnn": {
            "enabled": True,
            "input_size": 784,  # Size of the input
            "hidden_sizes": [1024, 512],  # Size of hidden layers
            "output_size": 10,  # Size of the output
            "lateral_connections": True  # Whether to use lateral connections
        }
    }
}
```

## PackNet

### Overview
PackNet is a parameter-efficient continual learning approach that uses network pruning to free up parameters after training on a task, allowing a single network to learn multiple tasks sequentially without forgetting.

### How it Works
After training on each task, PackNet prunes a portion of the least important weights (based on magnitude or gradient information), freezes the remaining non-zero weights which are important for the current task, and then uses the freed-up capacity (pruned weights) for learning new tasks. 

This creates task-specific binary masks that protect important weights for each previously learned task while allowing new tasks to utilize the pruned portions of the network.

### Mathematical Formulation
For a network with weights $\theta$, PackNet defines a binary mask $M_t$ for each task $t$:

$M_t(i) = \begin{cases} 
1, & \text{if } \theta_i \text{ is important for task } t \\ 
0, & \text{otherwise} 
\end{cases}$

During training, the effective weights $\theta_{\text{eff}}$ for task $t$ are:

$\theta_{\text{eff}} = \theta \odot (1 - \sum_{i=1}^{t-1} M_i)$

Where $\odot$ represents element-wise multiplication. For inference on task $t$, the network uses:

$\theta_{\text{task}_t} = \theta \odot M_t$

### Advantages and Limitations
**Advantages:**
- More parameter-efficient than Progressive Neural Networks
- Effectively prevents forgetting with a fixed model size
- No need for additional regularization terms or replay memory

**Limitations:**
- Requires knowing task identity at inference time
- Performance may degrade if pruning too aggressively
- Training becomes more constrained for later tasks as more parameters are frozen

### Usage in CLM Framework
```python
config = {
    "continual_strategy": {
        "packnet": {
            "enabled": True,
            "prune_percentage": 0.75,  # Percentage of weights to prune after each task
            "prune_threshold": 0.001,  # Minimum absolute value for weights to be kept
            "use_magnitude_pruning": True  # Use magnitude-based pruning (true) or gradient-based (false)
        }
    }
}
```

## Strategy Comparison

| Strategy | Forgetting Prevention | Parameter Growth | Memory Requirements | Task Identity Required |
|----------|----------------------|-----------------|---------------------|------------------------|
| Naive Fine-tuning | None | None | None | No |
| EWC | Moderate | None | Low (Fisher Matrix) | No |
| LwF | Moderate | None | Low (Previous Outputs) | No |
| GEM | Strong | None | High (Replay Buffer) | Yes |
| PNN | Complete | Linear | None | Yes |
| PackNet | Strong | None | Low (Binary Masks) | Yes | 
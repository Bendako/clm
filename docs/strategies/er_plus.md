# Experience Replay with Regularization (ER+)

## Overview

ER+ (Experience Replay with Regularization) is a simple yet effective continual learning strategy that combines two key approaches:

1. **Experience Replay**: Stores a subset of data from previous tasks in a replay buffer and replays them during training on new tasks.
2. **Knowledge Distillation Regularization**: Adds a regularization term that penalizes changes to the model's predictions on previous task data.

This combination helps the model preserve knowledge from previous tasks while learning new ones, effectively mitigating catastrophic forgetting.

## How It Works

ER+ follows these key steps:

1. **Before Training on a New Task:**
   - Store the current model's predictions (logits) for samples in the replay buffer.
   - These stored predictions serve as "soft targets" for regularization.

2. **During Training on a New Task:**
   - Compute the standard task loss on current task data.
   - Sample a batch from the replay buffer.
   - Compute a replay loss on the sampled data (standard cross-entropy).
   - Compute a distillation loss that penalizes changes to the model's predictions on previous task data.
   - Combine these losses: `total_loss = task_loss + replay_loss + reg_weight * distillation_loss`.

3. **After Training on a Task:**
   - Update the replay buffer with samples from the completed task.

## Parameters

- **memory_size**: Size of the replay buffer (default: 500).
- **batch_size**: Batch size for sampling from replay buffer (default: 32).
- **reg_weight**: Weight for the regularization/distillation loss (default: 1.0).
- **temperature**: Temperature for softening logits in distillation loss (default: 2.0).
- **task_balanced**: Whether to maintain balanced samples per task in buffer (default: True).

## Usage

### Configuration

```yaml
# Continual learning strategy configuration
continual_strategy:
  er_plus:
    enabled: true
    memory_size: 500
    batch_size: 32
    reg_weight: 1.0
    temperature: 2.0
    task_balanced: true
```

### Code Example

```python
from ml.continual_strategies import ERPlus
from ml.training.continual.trainer import ContinualTrainer

# Create a model
model = create_model()

# Initialize trainer with ER+ strategy
trainer = ContinualTrainer(
    model=model,
    config=config,  # Config with ER+ settings
    device=device
)

# Train on sequential tasks
for task_id, (task_name, (train_loader, val_loader)) in enumerate(tasks):
    trainer.train_task(
        train_loader=train_loader,
        val_loader=val_loader,
        task_name=task_name,
        task_id=task_id
    )
```

## Advantages

- **Simplicity**: ER+ is conceptually simpler than many other continual learning methods.
- **Effectiveness**: Despite its simplicity, it performs competitively compared to more complex approaches.
- **Efficiency**: Memory requirements scale linearly with the number of tasks.
- **Stability**: The regularization term provides stability for previous task performance.

## References

- Buzzega, P., Boschini, M., Porrello, A., Abati, D., & Calderara, S. (2020). Dark Experience for General Continual Learning: a Strong, Simple Baseline. Advances in Neural Information Processing Systems.

## Related Strategies

- **EWC (Elastic Weight Consolidation)**: Constrains important parameters for previous tasks.
- **LwF (Learning without Forgetting)**: Uses knowledge distillation from a copy of the model before training on a new task.
- **GEM (Gradient Episodic Memory)**: Uses episodic memory to constrain gradients to avoid interfering with previous tasks. 
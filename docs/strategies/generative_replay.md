# Generative Replay Strategy

## Overview

Generative Replay (GR) is a continual learning strategy that uses a generative model to mitigate catastrophic forgetting. Instead of storing raw samples from previous tasks, GR trains a generative model (typically a Variational Autoencoder or GAN) to generate synthetic samples that represent data distributions of previous tasks. These synthetic samples are then used during training on new tasks to maintain performance on older ones.

This approach is also known as "pseudo-rehearsal" or "deep generative replay" in the literature. It offers a particularly effective solution when privacy constraints prevent storing raw data from previous tasks.

## How It Works

The Generative Replay strategy consists of the following steps:

### Before Training on Each New Task
1. The generative model (e.g., VAE) is used to create synthetic samples representing previous tasks
2. The strategy initializes a reservoir buffer to collect samples from the current task for training the generator

### During Training on Each New Task
1. For each batch from the current task:
   - The main model is trained on the current task data
   - The loss is computed as a combination of:
     - Standard task loss on current data
     - Loss on synthetic samples from previous tasks
   - Periodically, the generator is updated using samples from the replay buffer
   - New samples are added to the replay buffer for future generator updates

### After Training on Each Task
1. The replay buffer contains a representative set of the current task's data
2. The generative model has been updated to capture the current task's distribution

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `memory_size` | 500 | Maximum number of samples to store in the replay buffer for training the generator |
| `batch_size` | 32 | Batch size for sampling from the replay buffer |
| `replay_weight` | 1.0 | Weight for the replay loss (how much to emphasize previous task performance) |
| `generator_update_freq` | 5 | How often to update the generator model (in iterations) |
| `task_balanced` | true | Whether to balance samples across tasks in the replay buffer |

## Configuration Example

```yaml
# Enable Generative Replay
continual_strategy:
  generative_replay:
    enabled: true
    memory_size: 500
    batch_size: 32
    replay_weight: 1.0
    generator_update_freq: 5
    task_balanced: true
  
  # Generator model configuration
  generator:
    type: "vae"  # Options: vae, conv_vae
    hidden_dims: [256, 128]
    latent_dim: 32
    dropout: 0.1
```

## Code Example

```python
from ml.training.continual import ContinualTrainer
from ml.models.simple import SimpleMLP
from ml.models.vae import VAE

# Create main model
model = SimpleMLP(input_size=784, hidden_sizes=[256, 128], output_size=10)

# Create VAE generator (will be handled by the trainer)
# The trainer will automatically create the generator based on configuration

# Create trainer with Generative Replay
trainer = ContinualTrainer(
    model=model,
    config={
        "optimizer": {"type": "adam", "lr": 0.001},
        "continual_strategy": {
            "generative_replay": {
                "enabled": True,
                "memory_size": 500,
                "replay_weight": 1.0,
                "generator_update_freq": 5
            }
        },
        "generator": {
            "type": "vae",
            "hidden_dims": [256, 128],
            "latent_dim": 32
        }
    }
)

# Train on multiple tasks sequentially
for task_id, (train_loader, val_loader) in enumerate(task_data):
    trainer.train_task(
        train_loader=train_loader,
        val_loader=val_loader,
        task_id=task_id,
        task_name=f"task_{task_id}",
        num_epochs=10
    )
    
    # Optionally visualize generated samples
    if hasattr(trainer, 'generator_model'):
        samples = trainer.generator_model.generate(num_samples=10, device=device)
        # Visualize samples
```

## Advantages

- **Privacy-preserving**: No need to store raw data from previous tasks, only a generative model
- **Scalability**: Memory requirements do not increase with the number of tasks
- **Flexibility**: Works well with different model architectures and task types
- **Performance**: Often achieves better performance than simple replay methods, especially as the number of tasks increases

## Limitations

- **Computational overhead**: Requires training and maintaining a generative model
- **Quality dependent**: Performance heavily depends on the quality of the generative model
- **Initial tasks**: May struggle with accurately generating samples for the very first tasks as training progresses

## References

1. Shin, H., Lee, J. K., Kim, J., & Kim, J. (2017). Continual learning with deep generative replay. Advances in Neural Information Processing Systems, 30.
2. van de Ven, G. M., & Tolias, A. S. (2018). Generative replay with feedback connections as a general strategy for continual learning. arXiv preprint arXiv:1809.10635.

## Related Strategies

- **EWC (Elastic Weight Consolidation)**: Constrains important weights, while GR uses synthetic data
- **LwF (Learning without Forgetting)**: Uses knowledge distillation without samples, while GR generates synthetic samples
- **ER+ (Experience Replay with Regularization)**: Combines replay with regularization, but requires storing real samples
- **DGR (Deep Generative Replay)**: A specific implementation of generative replay using GANs 
# Model Checkpoints

This directory stores trained model weights.

## Pre-trained Models

Pre-trained checkpoints will be added after training is complete.

### Planned Checkpoints

- `baseline_random.pt` - Random baseline (for comparison)
- `baseline_greedy_density.pt` - Greedy density baseline
- `ppo_1m_steps.pt` - PPO agent trained for 1M steps
- `ppo_10m_steps.pt` - PPO agent trained for 10M steps
- `best_model.pt` - Best performing model

## Loading Checkpoints

```python
# Using PyTorch
import torch

checkpoint = torch.load('checkpoints/ppo_1m_steps.pt')
policy.load_state_dict(checkpoint['policy_state_dict'])
```

## Checkpoint Format

Saved checkpoints include:
- `policy_state_dict`: Policy network weights
- `optimizer_state_dict`: Optimizer state (for resuming training)
- `config`: Training configuration
- `step`: Global training step
- `metrics`: Performance metrics at save time

## Training Your Own

To train a model and save checkpoints:

```bash
python -m src.training.train --config configs/default.yaml --steps 1000000
```

Checkpoints are saved automatically every 1000 updates (configurable in config).

"""
Training Script
===============
Main training entrypoint using PufferLib PPO.
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.training.config import Config
from src.training.puffer_config import make_env


def train(config: Config, seed: int = None):
    """
    Train an agent using PufferLib PPO.

    Args:
        config: Training configuration
        seed: Random seed
    """
    print("=" * 60)
    print("Toronto Transit RL Training")
    print("=" * 60)

    # Create checkpoint directory
    os.makedirs(config.checkpoints.save_dir, exist_ok=True)

    try:
        import pufferlib
        import pufferlib.vector
        import torch
        import torch.nn as nn

        print("\nInitializing environment...")

        # Create environment factory
        env_creator = make_env(
            grid_size=tuple(config.env.grid_size),
            n_zones=config.env.n_zones,
            budget=config.env.budget,
            max_steps=config.env.max_steps,
            equity_weight=config.env.equity_weight,
            seed=seed,
            normalize_obs=config.env.normalize_obs,
        )

        # Create single env to get observation/action spaces
        test_env = env_creator()
        obs_space = test_env.observation_space
        action_space = test_env.action_space
        print(f"Observation space: {obs_space}")
        print(f"Action space: {action_space}")

        # Create vectorized environment
        print(f"\nCreating {config.training.num_envs} parallel environments...")
        vec_env = pufferlib.vector.make(
            env_creator,
            num_envs=config.training.num_envs,
        )

        # Simple MLP policy for discrete actions
        class Policy(nn.Module):
            def __init__(self, obs_dim, action_dim, hidden_dim=256):
                super().__init__()
                self.network = nn.Sequential(
                    nn.Linear(obs_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                )
                self.actor = nn.Linear(hidden_dim, action_dim)
                self.critic = nn.Linear(hidden_dim, 1)

            def forward(self, obs, action_mask=None):
                features = self.network(obs)
                logits = self.actor(features)

                # Apply action mask if provided
                if action_mask is not None:
                    logits = logits.masked_fill(~action_mask, float('-inf'))

                value = self.critic(features)
                return logits, value

        obs_dim = obs_space.shape[0]
        action_dim = action_space.n

        print(f"\nInitializing policy (obs_dim={obs_dim}, action_dim={action_dim})...")
        policy = Policy(obs_dim, action_dim)

        # Note: Full PufferLib integration requires their PPO trainer
        # This is a simplified placeholder showing the structure
        print("\nTraining configuration:")
        print(f"  Total steps: {config.training.total_steps}")
        print(f"  Learning rate: {config.training.learning_rate}")
        print(f"  Batch size: {config.training.batch_size}")
        print(f"  Gamma: {config.training.gamma}")
        print(f"  Clip coefficient: {config.training.clip_coef}")

        print("\nNOTE: Full PufferLib PPO training requires additional setup.")
        print("This script demonstrates the structure. For actual training:")
        print("  1. Install PufferLib: pip install pufferlib")
        print("  2. Use PufferLib's PPO trainer with the policy and vec_env")
        print("  3. Integrate action masking into the training loop")

        # Placeholder for actual training loop
        # In a full implementation, you would use:
        # trainer = pufferlib.PPO(policy, vec_env, **training_config)
        # trainer.train(total_steps=config.training.total_steps)

        print("\nSetup complete! Environment and policy ready for training.")

    except ImportError as e:
        print(f"\nError: {e}")
        print("\nPufferLib or PyTorch not installed.")
        print("Install with: pip install pufferlib torch")
        print("\nFalling back to simple random rollout test...")

        # Fallback: simple test rollout
        env = env_creator()
        obs, info = env.reset(seed=seed)

        print("\nRunning test rollout...")
        total_reward = 0
        for step in range(10):
            mask = info.get("action_mask", np.ones(env.action_space.n, dtype=bool))
            valid_actions = np.where(mask)[0]
            action = np.random.choice(valid_actions)

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            print(f"  Step {step + 1}: action={action}, reward={reward:.3f}")

            if terminated or truncated:
                break

        print(f"\nTest rollout complete. Total reward: {total_reward:.3f}")


def main():
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Train Toronto Transit RL agent")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Total training steps (overrides config)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (overrides config)",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=None,
        help="Number of parallel environments (overrides config)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    # Load config
    config = Config.from_yaml(args.config)

    # Override with CLI args
    if args.steps is not None:
        config.training.total_steps = args.steps
    if args.lr is not None:
        config.training.learning_rate = args.lr
    if args.num_envs is not None:
        config.training.num_envs = args.num_envs

    # Train
    train(config, seed=args.seed)


if __name__ == "__main__":
    main()

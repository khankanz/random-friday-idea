"""
PufferLib Configuration and Wrappers
=====================================
Environment wrappers and factory functions for PufferLib integration.
"""

import numpy as np
import gymnasium as gym
from typing import Optional, Callable
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.env.toronto_env import TorontoTransitEnv


class ActionMaskWrapper(gym.Wrapper):
    """
    Wrapper to handle action masking for PufferLib.

    Adds action mask to observation space and modifies invalid actions
    to have -inf logits during policy selection.
    """

    def __init__(self, env):
        super().__init__(env)
        # Keep original observation space, mask will be in info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info["action_mask"] = self.env.get_action_mask()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["action_mask"] = self.env.get_action_mask()
        return obs, reward, terminated, truncated, info


class ObservationNormalizationWrapper(gym.ObservationWrapper):
    """
    Optional observation normalization wrapper.

    Keeps running statistics and normalizes observations.
    """

    def __init__(self, env, epsilon=1e-8):
        super().__init__(env)
        self.epsilon = epsilon
        self.obs_mean = np.zeros(env.observation_space.shape, dtype=np.float32)
        self.obs_var = np.ones(env.observation_space.shape, dtype=np.float32)
        self.count = 0

    def observation(self, obs):
        """Normalize observation."""
        return (obs - self.obs_mean) / (np.sqrt(self.obs_var) + self.epsilon)

    def update_stats(self, obs):
        """Update running mean and variance."""
        self.count += 1
        delta = obs - self.obs_mean
        self.obs_mean += delta / self.count
        delta2 = obs - self.obs_mean
        self.obs_var += delta * delta2


def make_env(
    env_id: str = "toronto-transit-v0",
    grid_size: tuple = (40, 30),
    n_zones: int = 100,
    budget: float = 5000.0,
    max_steps: int = 20,
    equity_weight: float = 0.3,
    seed: Optional[int] = None,
    normalize_obs: bool = False,
) -> Callable:
    """
    Environment factory for PufferLib.

    Returns a function that creates wrapped environments.

    Args:
        env_id: Environment identifier
        grid_size: Grid dimensions
        n_zones: Number of demand zones
        budget: Initial budget
        max_steps: Maximum episode length
        equity_weight: Weight for equity score in reward
        seed: Random seed
        normalize_obs: Whether to normalize observations

    Returns:
        Function that creates a wrapped environment
    """

    def _make():
        env = TorontoTransitEnv(
            grid_size=grid_size,
            n_zones=n_zones,
            budget=budget,
            max_steps=max_steps,
            equity_weight=equity_weight,
            seed=seed,
        )

        # Add action mask wrapper
        env = ActionMaskWrapper(env)

        # Optionally add normalization
        if normalize_obs:
            env = ObservationNormalizationWrapper(env)

        return env

    return _make


def make_pufferlib_env(num_envs: int = 4, **env_kwargs):
    """
    Create vectorized PufferLib environment.

    Args:
        num_envs: Number of parallel environments
        **env_kwargs: Arguments passed to make_env

    Returns:
        Vectorized environment compatible with PufferLib
    """
    try:
        import pufferlib
        import pufferlib.vector

        env_creator = make_env(**env_kwargs)

        # Create vectorized environment
        vec_env = pufferlib.vector.make(
            env_creator,
            num_envs=num_envs,
            env_kwargs={},
        )

        return vec_env

    except ImportError:
        print("PufferLib not installed. Install with: pip install pufferlib")
        raise


def apply_action_mask_to_logits(logits, action_mask):
    """
    Apply action mask to logits by setting invalid actions to -inf.

    Args:
        logits: Action logits from policy (batch_size, num_actions)
        action_mask: Boolean mask of valid actions (batch_size, num_actions)

    Returns:
        Masked logits
    """
    # Convert mask to same device as logits
    if hasattr(logits, 'device'):  # PyTorch tensor
        import torch
        mask_tensor = torch.tensor(action_mask, device=logits.device, dtype=torch.bool)
        masked_logits = logits.clone()
        masked_logits[~mask_tensor] = float('-inf')
        return masked_logits
    else:  # NumPy array
        masked_logits = logits.copy()
        masked_logits[~action_mask] = float('-inf')
        return masked_logits


if __name__ == "__main__":
    print("Testing PufferLib config...")

    # Test basic environment creation
    env_fn = make_env(grid_size=(20, 15), n_zones=20, seed=42)
    env = env_fn()

    print(f"Environment created: {env}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # Test reset and step
    obs, info = env.reset()
    print(f"Initial obs shape: {obs.shape}")
    print(f"Action mask in info: {'action_mask' in info}")

    # Test step
    mask = info["action_mask"]
    valid_actions = np.where(mask)[0]
    action = valid_actions[0]

    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step successful, reward: {reward:.3f}")

    print("\nPufferLib config tests passed!")

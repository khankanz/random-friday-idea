"""
Evaluation Script
=================
Evaluate trained models and compare to baselines.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
from typing import Dict, List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.training.puffer_config import make_env
from src.training.config import Config


class RandomBaseline:
    """Random action baseline."""

    def predict(self, obs, action_mask):
        """Select random valid action."""
        valid_actions = np.where(action_mask)[0]
        if len(valid_actions) == 0:
            return 0  # Default action
        return np.random.choice(valid_actions)


class GreedyDensityBaseline:
    """Greedy baseline that selects highest density node without transit."""

    def __init__(self, env):
        self.env = env

    def predict(self, obs, action_mask):
        """Select highest density valid node."""
        # Extract node densities from observation
        # obs format: [node_features (n_nodes * 3), budget, ttt, steps]
        n_nodes = self.env.n_nodes
        node_features = obs[:n_nodes * 3].reshape(n_nodes, 3)
        densities = node_features[:, 0]  # First feature is density

        # Mask out invalid actions
        valid_densities = densities.copy()
        valid_densities[~action_mask[:n_nodes]] = -1

        # Select highest density
        return np.argmax(valid_densities)


def evaluate_policy(
    env,
    policy,
    num_episodes: int = 10,
    deterministic: bool = True,
    seed: int = None,
) -> Dict:
    """
    Evaluate a policy for multiple episodes.

    Args:
        env: Environment
        policy: Policy with predict(obs, action_mask) method
        num_episodes: Number of episodes to run
        deterministic: Whether to use deterministic actions
        seed: Random seed

    Returns:
        Dictionary of evaluation metrics
    """
    episode_rewards = []
    episode_lengths = []
    episode_ttt_improvements = []
    episode_equity_scores = []
    episode_budgets_used = []

    for ep in range(num_episodes):
        obs, info = env.reset(seed=seed + ep if seed else None)
        baseline_ttt = info["baseline_ttt"]

        episode_reward = 0
        episode_length = 0
        done = False

        while not done:
            # Get action from policy
            action_mask = env.get_action_mask()
            action = policy.predict(obs, action_mask)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            done = terminated or truncated

        # Collect metrics
        final_ttt = env.current_ttt
        ttt_improvement = (baseline_ttt - final_ttt) / baseline_ttt
        equity_score = env._compute_equity_score()
        budget_used = env.initial_budget - env.budget

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_ttt_improvements.append(ttt_improvement)
        episode_equity_scores.append(equity_score)
        episode_budgets_used.append(budget_used)

    # Compute statistics
    results = {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_ttt_improvement": np.mean(episode_ttt_improvements),
        "std_ttt_improvement": np.std(episode_ttt_improvements),
        "mean_equity_score": np.mean(episode_equity_scores),
        "std_equity_score": np.std(episode_equity_scores),
        "mean_budget_used": np.mean(episode_budgets_used),
        "mean_episode_length": np.mean(episode_lengths),
    }

    return results


def main():
    """Main evaluation entrypoint."""
    parser = argparse.ArgumentParser(description="Evaluate Toronto Transit RL agent")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to trained checkpoint (if None, uses baselines)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="random",
        choices=["random", "greedy-density"],
        help="Baseline policy to use if no checkpoint provided",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Toronto Transit RL Evaluation")
    print("=" * 60)

    # Load config
    config = Config.from_yaml(args.config)

    # Create environment
    print("\nCreating environment...")
    env_creator = make_env(
        grid_size=tuple(config.env.grid_size),
        n_zones=config.env.n_zones,
        budget=config.env.budget,
        max_steps=config.env.max_steps,
        equity_weight=config.env.equity_weight,
        seed=args.seed,
    )
    env = env_creator()

    # Load policy
    if args.checkpoint:
        print(f"\nLoading checkpoint: {args.checkpoint}")
        print("NOTE: Checkpoint loading requires trained model.")
        print("Using random baseline instead for demonstration.")
        policy = RandomBaseline()
    else:
        print(f"\nUsing {args.baseline} baseline policy")
        if args.baseline == "random":
            policy = RandomBaseline()
        elif args.baseline == "greedy-density":
            policy = GreedyDensityBaseline(env)

    # Evaluate
    print(f"\nRunning evaluation for {args.num_episodes} episodes...")
    results = evaluate_policy(
        env,
        policy,
        num_episodes=args.num_episodes,
        seed=args.seed,
    )

    # Print results
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"Mean reward: {results['mean_reward']:.3f} ± {results['std_reward']:.3f}")
    print(f"Mean TTT improvement: {results['mean_ttt_improvement']:.2%} ± {results['std_ttt_improvement']:.2%}")
    print(f"Mean equity score: {results['mean_equity_score']:.3f} ± {results['std_equity_score']:.3f}")
    print(f"Mean budget used: ${results['mean_budget_used']:.0f}M")
    print(f"Mean episode length: {results['mean_episode_length']:.1f} steps")
    print("=" * 60)


if __name__ == "__main__":
    main()

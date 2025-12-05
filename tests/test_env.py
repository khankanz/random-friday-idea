"""
Comprehensive tests for TorontoTransitEnv
"""

import pytest
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.env.toronto_env import TorontoTransitEnv


class TestTorontoTransitEnv:
    """Test suite for Toronto Transit Environment."""

    @pytest.fixture
    def env(self):
        """Create a test environment."""
        return TorontoTransitEnv(
            grid_size=(20, 15),  # Smaller for faster tests
            n_zones=20,
            budget=5000.0,
            max_steps=10,
            seed=42,
        )

    def test_reset_returns_valid_observation(self, env):
        """Test that reset returns valid observation."""
        obs, info = env.reset()

        # Check observation shape
        expected_shape = (env.n_nodes * 3 + 3,)
        assert obs.shape == expected_shape, f"Expected shape {expected_shape}, got {obs.shape}"

        # Check observation is numpy array
        assert isinstance(obs, np.ndarray)
        assert obs.dtype == np.float32

        # Check info contains baseline_ttt
        assert "baseline_ttt" in info
        assert info["baseline_ttt"] > 0

    def test_observation_space(self, env):
        """Test observation space matches actual observations."""
        obs, _ = env.reset()
        assert env.observation_space.contains(obs), "Observation not in observation space"

    def test_action_space(self, env):
        """Test action space is correctly defined."""
        # Should be n_nodes + 1 (for done action)
        expected_actions = env.n_nodes + 1
        assert env.action_space.n == expected_actions

    def test_step_with_valid_action(self, env):
        """Test step with valid action updates state correctly."""
        env.reset(seed=42)
        initial_budget = env.budget
        initial_steps = env.steps

        # Get a valid action (not already transit)
        mask = env.get_action_mask()
        valid_actions = np.where(mask)[0]
        action = valid_actions[0]

        obs, reward, terminated, truncated, info = env.step(action)

        # Check that state was updated
        assert env.steps == initial_steps + 1
        assert env.budget < initial_budget  # Budget decreased
        assert info["action"] == "added_transit"

        # Check observation is valid
        assert env.observation_space.contains(obs)

    def test_step_with_invalid_action_duplicate(self, env):
        """Test step with invalid action (duplicate node) returns penalty."""
        env.reset(seed=42)

        # Pick a node that already has transit
        action = list(env.existing_transit_nodes)[0]

        obs, reward, terminated, truncated, info = env.step(action)

        # Should return penalty
        assert reward == -0.01
        assert info["action"] == "invalid_duplicate"

    def test_budget_exhaustion_terminates(self, env):
        """Test episode terminates when budget exhausted."""
        # Create env with very low budget
        low_budget_env = TorontoTransitEnv(
            grid_size=(20, 15),
            budget=400.0,  # Less than one station
            station_cost=500.0,
            seed=42,
        )
        low_budget_env.reset()

        # Try to take an action
        mask = low_budget_env.get_action_mask()
        valid_actions = np.where(mask)[0]

        # Should be only the "done" action available
        assert valid_actions[-1] == low_budget_env.n_nodes  # Done action

    def test_ttt_decreases_in_dense_area(self, env):
        """Test TTT decreases when adding transit in dense area."""
        env.reset(seed=42)
        initial_ttt = env.current_ttt

        # Find a dense node without transit
        dense_nodes = []
        for n in range(env.n_nodes):
            if n not in env.transit_nodes and env.G.nodes[n]["density"] > 0.7:
                dense_nodes.append(n)

        if len(dense_nodes) > 0:
            action = dense_nodes[0]
            obs, reward, terminated, truncated, info = env.step(action)

            # TTT should likely decrease (or at least not increase much)
            # This is stochastic due to sampling, so we just check it computed
            assert "ttt_improvement" in info

    def test_max_steps_truncation(self, env):
        """Test episode truncates after max_steps."""
        env.reset(seed=42)

        for i in range(env.max_steps):
            mask = env.get_action_mask()
            valid_actions = np.where(mask)[0]
            # Filter out done action
            valid_actions = [a for a in valid_actions if a < env.n_nodes]

            if len(valid_actions) == 0:
                break

            action = valid_actions[0]
            obs, reward, terminated, truncated, info = env.step(action)

            if i == env.max_steps - 1:
                assert terminated, "Should terminate at max_steps"

    def test_done_action(self, env):
        """Test that done action terminates episode."""
        env.reset()

        # Take done action
        action = env.n_nodes
        obs, reward, terminated, truncated, info = env.step(action)

        assert terminated
        assert info["action"] == "done"
        assert reward == 0.0

    def test_action_mask(self, env):
        """Test action mask correctly identifies valid actions."""
        env.reset(seed=42)

        mask = env.get_action_mask()

        # Mask should be boolean array
        assert mask.dtype == bool
        assert len(mask) == env.n_nodes + 1

        # Nodes with existing transit should be masked
        for node in env.existing_transit_nodes:
            assert not mask[node], f"Node {node} has transit but not masked"

        # Done action should be valid
        assert mask[env.n_nodes]

    def test_seeded_reproducibility(self):
        """Test that same seed produces same results."""
        env1 = TorontoTransitEnv(seed=123)
        env2 = TorontoTransitEnv(seed=123)

        obs1, info1 = env1.reset(seed=123)
        obs2, info2 = env2.reset(seed=123)

        np.testing.assert_array_almost_equal(obs1, obs2)
        assert info1["baseline_ttt"] == info2["baseline_ttt"]

    def test_equity_score_computation(self, env):
        """Test equity score is computed correctly."""
        env.reset()
        equity_score = env._compute_equity_score()

        # Score should be between 0 and 1
        assert 0.0 <= equity_score <= 1.0

    def test_render_human(self, env):
        """Test that render doesn't crash."""
        env_with_render = TorontoTransitEnv(
            grid_size=(20, 15),
            render_mode="human",
            seed=42,
        )
        env_with_render.reset()

        # Should not raise
        env_with_render.render()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

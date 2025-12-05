"""
Toronto Transit RL Environment
==============================
A Gymnasium environment for training agents to design transit networks.

State: Grid representation of Toronto with density, existing transit, budget
Action: Select a candidate location to add transit infrastructure
Reward: Improvement in weighted average travel time, with equity bonus
"""

import gymnasium as gym
from gymnasium import spaces
import networkx as nx
import numpy as np
from typing import Optional, Tuple, Dict, Any


class TorontoTransitEnv(gym.Env):
    """
    RL Environment for Toronto transit network design.
    
    The agent proposes transit extensions over a series of steps,
    trying to minimize travel time while respecting a budget.
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 1}
    
    def __init__(
        self,
        grid_size: Tuple[int, int] = (40, 30),
        n_zones: int = 100,
        budget: float = 5000.0,  # Million $
        station_cost: float = 500.0,  # Million $ per station
        line_cost_per_km: float = 200.0,  # Million $ per km
        max_steps: int = 20,
        equity_weight: float = 0.3,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()
        
        self.grid_x, self.grid_y = grid_size
        self.n_nodes = self.grid_x * self.grid_y
        self.n_zones = n_zones
        self.initial_budget = budget
        self.station_cost = station_cost
        self.line_cost_per_km = line_cost_per_km
        self.max_steps = max_steps
        self.equity_weight = equity_weight
        self.render_mode = render_mode
        
        # Constants
        self.ROAD_WEIGHT = 10.0
        self.TRANSIT_WEIGHT = 2.0
        self.GRID_SPACING_KM = 0.5  # Each grid cell is ~500m
        
        # Action space: choose a node to add as a transit hub
        # (will connect to nearest existing transit)
        self.action_space = spaces.Discrete(self.n_nodes + 1)  # +1 for "done" action
        
        # Observation space
        self.observation_space = spaces.Dict({
            # Per-node features: density, has_transit, is_zone_centroid
            "node_features": spaces.Box(
                low=0, high=1, shape=(self.n_nodes, 3), dtype=np.float32
            ),
            # Global state
            "budget_remaining": spaces.Box(low=0, high=budget, shape=(1,), dtype=np.float32),
            "current_ttt": spaces.Box(low=0, high=1000, shape=(1,), dtype=np.float32),
            "steps_taken": spaces.Box(low=0, high=max_steps, shape=(1,), dtype=np.int32),
        })
        
        # Initialize
        self._rng = np.random.default_rng(seed)
        self._build_base_network()
        self.reset(seed=seed)
    
    def _build_base_network(self):
        """Build the base Toronto road network with existing TTC."""
        self.G_base = nx.Graph()
        
        # Add nodes with coordinates and density
        for i in range(self.grid_x):
            for j in range(self.grid_y):
                node_id = i * self.grid_y + j
                lon = -79.65 + (i / self.grid_x) * 0.55
                lat = 43.58 + (j / self.grid_y) * 0.27
                
                # Density peaks near downtown (i=25, j=15)
                dist_to_downtown = np.sqrt((i - 25)**2 + (j - 15)**2)
                density = max(0.1, 1.0 - dist_to_downtown / 30)
                
                # Secondary peak near North York (i=25, j=22)
                dist_to_ny = np.sqrt((i - 25)**2 + (j - 22)**2)
                density = max(density, 0.7 - dist_to_ny / 20)
                
                # Scarborough (i=35, j=12) - moderate density
                dist_to_scarb = np.sqrt((i - 35)**2 + (j - 12)**2)
                density = max(density, 0.5 - dist_to_scarb / 25)
                
                self.G_base.add_node(node_id, x=lon, y=lat, density=density)
        
        # Add road edges
        for i in range(self.grid_x):
            for j in range(self.grid_y):
                node_id = i * self.grid_y + j
                if i < self.grid_x - 1:
                    self.G_base.add_edge(
                        node_id, (i + 1) * self.grid_y + j,
                        weight=self.ROAD_WEIGHT, edge_type="road"
                    )
                if j < self.grid_y - 1:
                    self.G_base.add_edge(
                        node_id, i * self.grid_y + (j + 1),
                        weight=self.ROAD_WEIGHT, edge_type="road"
                    )
        
        # Add existing TTC Lines
        self.existing_transit_nodes = set()
        
        # Line 1 (Yonge-University-Spadina): U-shape
        # Simplified as vertical line at x=25
        line1 = [25 * self.grid_y + j for j in range(5, 26)]
        self._add_transit_line(line1, "subway_line1")
        
        # Line 2 (Bloor-Danforth): horizontal at y=15
        line2 = [i * self.grid_y + 15 for i in range(8, 36)]
        self._add_transit_line(line2, "subway_line2")
        
        # Select zone centroids (stratified by density)
        all_nodes = list(range(self.n_nodes))
        densities = [self.G_base.nodes[n]["density"] for n in all_nodes]
        # Sample proportional to density
        probs = np.array(densities) / sum(densities)
        self.zone_nodes = self._rng.choice(
            all_nodes, size=self.n_zones, replace=False, p=probs
        )
        self.zone_densities = np.array([
            self.G_base.nodes[n]["density"] for n in self.zone_nodes
        ])
        
        # Identify low-income zones (outer areas = lower density here as proxy)
        self.low_income_zones = self.zone_densities < np.percentile(self.zone_densities, 40)
    
    def _add_transit_line(self, nodes: list, edge_type: str):
        """Add a transit line connecting a sequence of nodes."""
        for k in range(len(nodes) - 1):
            self.G_base.add_edge(
                nodes[k], nodes[k + 1],
                weight=self.TRANSIT_WEIGHT, edge_type=edge_type
            )
        self.existing_transit_nodes.update(nodes)
    
    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        
        # Copy base network
        self.G = self.G_base.copy()
        self.transit_nodes = self.existing_transit_nodes.copy()
        self.new_transit_nodes = set()
        
        # Reset state
        self.budget = self.initial_budget
        self.steps = 0
        self.baseline_ttt = self._compute_ttt()
        self.current_ttt = self.baseline_ttt
        
        return self._get_obs(), {"baseline_ttt": self.baseline_ttt}
    
    def _get_obs(self) -> Dict[str, np.ndarray]:
        """Construct observation dictionary."""
        # Node features: [density, has_transit, is_zone]
        node_features = np.zeros((self.n_nodes, 3), dtype=np.float32)
        for n in range(self.n_nodes):
            node_features[n, 0] = self.G.nodes[n]["density"]
            node_features[n, 1] = 1.0 if n in self.transit_nodes else 0.0
        for z in self.zone_nodes:
            node_features[z, 2] = 1.0
        
        return {
            "node_features": node_features,
            "budget_remaining": np.array([self.budget], dtype=np.float32),
            "current_ttt": np.array([self.current_ttt], dtype=np.float32),
            "steps_taken": np.array([self.steps], dtype=np.int32),
        }
    
    def _compute_ttt(self) -> float:
        """Compute demand-weighted average travel time."""
        # Sample zone pairs for efficiency
        n_sample = min(30, len(self.zone_nodes))
        sample_idx = self._rng.choice(len(self.zone_nodes), size=n_sample, replace=False)
        sample_zones = self.zone_nodes[sample_idx]
        sample_densities = self.zone_densities[sample_idx]
        
        total_demand = 0.0
        weighted_ttt = 0.0
        
        for i, src in enumerate(sample_zones):
            lengths = nx.single_source_dijkstra_path_length(self.G, src, weight="weight")
            for j, dst in enumerate(self.zone_nodes):
                if src != dst:
                    demand = sample_densities[i] * self.zone_densities[j]
                    ttt = lengths.get(dst, 1000.0)
                    weighted_ttt += demand * ttt
                    total_demand += demand
        
        return weighted_ttt / total_demand if total_demand > 0 else 1000.0
    
    def _compute_equity_score(self) -> float:
        """Compute transit accessibility for low-income zones."""
        # Fraction of low-income zones with transit access
        low_income_zone_nodes = self.zone_nodes[self.low_income_zones]
        n_with_transit = sum(
            1 for z in low_income_zone_nodes
            if any(
                self.G.has_edge(z, t) or z == t
                for t in self.transit_nodes
            ) or z in self.transit_nodes
        )
        return n_with_transit / len(low_income_zone_nodes)
    
    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Execute one step: add transit at the chosen location.
        
        Args:
            action: Node ID to add transit, or n_nodes for "done"
        
        Returns:
            observation, reward, terminated, truncated, info
        """
        self.steps += 1
        info = {}
        
        # Check for "done" action
        if action == self.n_nodes:
            terminated = True
            reward = 0.0
            info["action"] = "done"
            return self._get_obs(), reward, terminated, False, info
        
        # Validate action
        if action in self.transit_nodes:
            # Already has transit - small penalty, no change
            reward = -0.01
            info["action"] = "invalid_duplicate"
            terminated = self.budget < self.station_cost or self.steps >= self.max_steps
            return self._get_obs(), reward, terminated, False, info
        
        # Find nearest transit node to connect to
        min_dist = float("inf")
        nearest_transit = None
        action_x = action // self.grid_y
        action_y = action % self.grid_y
        
        for t in self.transit_nodes:
            t_x = t // self.grid_y
            t_y = t % self.grid_y
            dist = abs(action_x - t_x) + abs(action_y - t_y)  # Manhattan
            if dist < min_dist:
                min_dist = dist
                nearest_transit = t
        
        # Compute cost
        distance_km = min_dist * self.GRID_SPACING_KM
        cost = self.station_cost + distance_km * self.line_cost_per_km
        
        if cost > self.budget:
            reward = -0.01
            info["action"] = "over_budget"
            terminated = True
            return self._get_obs(), reward, terminated, False, info
        
        # Add the transit connection
        self._connect_transit(action, nearest_transit)
        self.transit_nodes.add(action)
        self.new_transit_nodes.add(action)
        self.budget -= cost
        
        # Compute new TTT and reward
        old_ttt = self.current_ttt
        self.current_ttt = self._compute_ttt()
        
        # Reward: TTT improvement + equity bonus
        ttt_improvement = (old_ttt - self.current_ttt) / self.baseline_ttt
        equity_score = self._compute_equity_score()
        
        reward = ttt_improvement + self.equity_weight * equity_score - 0.001 * (cost / 1000)
        
        # Termination
        terminated = self.budget < self.station_cost or self.steps >= self.max_steps
        truncated = False
        
        info.update({
            "action": "added_transit",
            "node": action,
            "cost": cost,
            "ttt_improvement": ttt_improvement,
            "equity_score": equity_score,
            "budget_remaining": self.budget,
        })
        
        return self._get_obs(), reward, terminated, truncated, info
    
    def _connect_transit(self, new_node: int, existing_node: int):
        """Add transit edges connecting new node to existing network."""
        # Add direct edge
        self.G.add_edge(
            new_node, existing_node,
            weight=self.TRANSIT_WEIGHT, edge_type="new_transit"
        )
        
        # Also connect along the path (simplified: just direct connection)
        # In a fuller version, you'd add intermediate stations
    
    def render(self):
        """Render the current state (placeholder for Folium/matplotlib)."""
        if self.render_mode == "human":
            print(f"\n=== Step {self.steps} ===")
            print(f"Budget: ${self.budget:.0f}M remaining")
            print(f"TTT: {self.current_ttt:.2f} (baseline: {self.baseline_ttt:.2f})")
            print(f"Improvement: {(self.baseline_ttt - self.current_ttt) / self.baseline_ttt * 100:.1f}%")
            print(f"New transit nodes: {len(self.new_transit_nodes)}")
            print(f"Equity score: {self._compute_equity_score():.2f}")
    
    def get_valid_actions(self) -> np.ndarray:
        """Return mask of valid actions (nodes without transit + done)."""
        mask = np.ones(self.n_nodes + 1, dtype=bool)
        for n in self.transit_nodes:
            mask[n] = False
        return mask


# Quick test
if __name__ == "__main__":
    print("Testing TorontoTransitEnv...")
    
    env = TorontoTransitEnv(render_mode="human", seed=42)
    obs, info = env.reset()
    
    print(f"\nInitial observation shapes:")
    for k, v in obs.items():
        print(f"  {k}: {v.shape}")
    print(f"Baseline TTT: {info['baseline_ttt']:.2f}")
    
    # Run a few random actions
    print("\n--- Random rollout ---")
    total_reward = 0
    for i in range(5):
        valid = env.get_valid_actions()
        valid_actions = np.where(valid)[0]
        action = np.random.choice(valid_actions)
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        env.render()
        
        if terminated:
            break
    
    print(f"\nTotal reward: {total_reward:.3f}")
    print(f"Final TTT improvement: {(env.baseline_ttt - env.current_ttt) / env.baseline_ttt * 100:.1f}%")

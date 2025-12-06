"""
Training Configuration
======================
Configuration dataclasses for training hyperparameters.
"""

from dataclasses import dataclass, field
from typing import Optional, List
import yaml


@dataclass
class EnvConfig:
    """Environment configuration."""
    grid_size: List[int] = field(default_factory=lambda: [40, 30])
    n_zones: int = 100
    budget: float = 5000.0
    station_cost: float = 500.0
    line_cost_per_km: float = 200.0
    max_steps: int = 20
    equity_weight: float = 0.3
    normalize_obs: bool = False


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    total_steps: int = 1000000
    num_envs: int = 8
    learning_rate: float = 3e-4
    batch_size: int = 256
    num_epochs: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5


@dataclass
class LoggingConfig:
    """Logging configuration."""
    log_interval: int = 10
    save_interval: int = 1000
    wandb_project: str = "toronto-transit-rl"
    wandb_entity: Optional[str] = None
    use_wandb: bool = False
    use_tensorboard: bool = True


@dataclass
class CheckpointConfig:
    """Checkpoint configuration."""
    save_dir: str = "checkpoints"
    load_path: Optional[str] = None


@dataclass
class Config:
    """Full configuration."""
    env: EnvConfig = field(default_factory=EnvConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    checkpoints: CheckpointConfig = field(default_factory=CheckpointConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load configuration from YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        return cls(
            env=EnvConfig(**data.get("env", {})),
            training=TrainingConfig(**data.get("training", {})),
            logging=LoggingConfig(**data.get("logging", {})),
            checkpoints=CheckpointConfig(**data.get("checkpoints", {})),
        )

    def to_yaml(self, path: str):
        """Save configuration to YAML file."""
        data = {
            "env": vars(self.env),
            "training": vars(self.training),
            "logging": vars(self.logging),
            "checkpoints": vars(self.checkpoints),
        }
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)


if __name__ == "__main__":
    # Test loading default config
    config = Config.from_yaml("configs/default.yaml")
    print("Loaded config:")
    print(f"  Env: {config.env}")
    print(f"  Training: {config.training}")
    print(f"  Logging: {config.logging}")

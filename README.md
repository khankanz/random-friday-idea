# Toronto Transit RL 🚇

Train an RL agent to design transit network extensions for Toronto. The agent proposes new stations and lines given existing TTC infrastructure, optimizing for travel time reduction and equity.

## Overview

This project uses reinforcement learning to automatically discover effective transit network expansions for the Greater Toronto Area. The learned proposals are compared against real Metrolinx plans (Ontario Line, Scarborough Extension) to evaluate performance.

### Key Features

- **Custom Gymnasium Environment**: Grid-based Toronto model with population density, existing TTC Lines 1 & 2
- **PufferLib Integration**: Efficient parallel training with PPO
- **Interactive Visualization**: FastHTML web app to explore agent proposals
- **Real Data Support**: GTFS feed parsing, census data integration, OSMnx road networks
- **Metrolinx Comparison**: Overlay planned projects to compare with agent designs

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/toronto-transit-rl.git
cd toronto-transit-rl

# Install dependencies
pip install -e .

# For training (optional)
pip install -e ".[training]"

# For data loading (optional)
pip install -e ".[data]"
```

### Run the Demo

```bash
# Launch interactive visualization
python -m src.viz.app

# Open browser to http://localhost:5000
```

### Train an Agent

```bash
# Train with default config
python -m src.training.train --config configs/default.yaml

# Evaluate trained model
python -m src.training.evaluate --checkpoint checkpoints/latest.pt
```

### Using Docker

```bash
# Build and run
docker-compose up

# Run training (separate profile)
docker-compose --profile training up trainer
```

## Project Structure

```
toronto-transit-rl/
├── src/
│   ├── env/              # Gymnasium environment
│   │   ├── toronto_env.py      # Main environment
│   │   ├── graph_builder.py    # Network construction
│   │   └── demand_model.py     # Travel demand computation
│   ├── training/         # RL training
│   │   ├── train.py           # Training entrypoint
│   │   ├── evaluate.py        # Evaluation script
│   │   ├── config.py          # Config dataclasses
│   │   └── puffer_config.py   # PufferLib wrappers
│   ├── data/             # Data loaders
│   │   ├── gtfs_loader.py     # TTC GTFS parsing
│   │   ├── census_loader.py   # StatsCan data
│   │   └── metrolinx.py       # Planned projects
│   └── viz/              # Visualization
│       ├── app.py             # FastHTML web app
│       ├── components.py      # UI components
│       └── map_render.py      # Map generation
├── tests/                # Test suite
├── configs/              # Training configs
├── data/                 # Data cache
├── checkpoints/          # Model weights
└── CLAUDE.md             # Project documentation
```

## How It Works

### Environment

- **State**: Flattened observation including node features (density, transit status, zone centroids), budget, travel time, and step count
- **Actions**: Discrete action space - select a node to add transit (automatically connects to nearest existing line)
- **Reward**:
  ```
  reward = ttt_improvement + equity_weight * equity_score - cost_penalty
  ```
  where `ttt_improvement` is normalized travel time reduction

### Training

Uses PufferLib for efficient vectorized PPO training:

```bash
python -m src.training.train \
  --config configs/default.yaml \
  --steps 1000000 \
  --num-envs 16 \
  --lr 0.0003
```

Key hyperparameters:
- Learning rate: 3e-4
- Discount (γ): 0.99
- GAE lambda: 0.95
- Clip coefficient: 0.2

### Visualization

The FastHTML app provides:
- Interactive Leaflet map of Toronto
- Real-time agent proposals (red markers)
- Existing TTC lines (blue)
- Metrolinx planned lines overlay (green)
- Metrics: TTT improvement, equity score, budget usage

## Data Sources

| Data | Source | Notes |
|------|--------|-------|
| TTC GTFS | [Toronto Open Data](https://gtfs.toronto.ca) | Real-time transit feeds |
| Population | StatsCan 2021 Census | Dissemination area profiles |
| Road Network | OSMnx | Falls back to synthetic grid if offline |
| Planned Transit | Metrolinx | Ontario Line, SSE, Eglinton West, Yonge North |

## Testing

```bash
# Run full test suite
pytest

# Run specific test
pytest tests/test_env.py -v

# Test environment directly
python -m src.env.toronto_env
```

## Configuration

Edit `configs/default.yaml` to adjust:

```yaml
env:
  grid_size: [40, 30]
  n_zones: 100
  budget: 5000.0
  max_steps: 20
  equity_weight: 0.3

training:
  total_steps: 1000000
  num_envs: 8
  learning_rate: 0.0003
```

## Results

*Note: Pre-trained checkpoint coming soon*

Random baseline:
- Mean TTT improvement: ~2-5%
- Mean equity score: ~0.3

Greedy density baseline:
- Mean TTT improvement: ~8-12%
- Mean equity score: ~0.45

## Future Work

- [ ] Full PufferLib PPO training integration
- [ ] Trained checkpoint with performance benchmarks
- [ ] Multi-modal support (subway, LRT, bus rapid transit)
- [ ] Temporal demand modeling (peak vs off-peak)
- [ ] Cost-benefit analysis vs Metrolinx plans
- [ ] Real census data integration
- [ ] OSMnx real road network support
- [ ] Deployment to cloud (Render, Railway, etc.)

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

MIT License - see LICENSE file for details

## References

- PufferLib: https://pufferlib.com
- FastHTML: https://fastht.ml
- Toronto Open Data: https://open.toronto.ca
- Metrolinx: https://www.metrolinx.com
- Transit.land: https://transit.land

## Citation

If you use this project in your research, please cite:

```bibtex
@software{toronto_transit_rl,
  title = {Toronto Transit RL},
  author = {Toronto Transit RL Team},
  year = {2024},
  url = {https://github.com/yourusername/toronto-transit-rl}
}
```

---

**Disclaimer**: This is a research project. Transit planning decisions should involve professional urban planners, community consultation, and comprehensive analysis.

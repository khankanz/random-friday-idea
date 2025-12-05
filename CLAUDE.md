# Toronto Transit RL Project

## Overview
Train an RL agent to design transit network extensions for Toronto. The agent proposes new stations/lines given existing TTC infrastructure, optimizing for travel time reduction and equity. Compare learned proposals against real Metrolinx plans (Ontario Line, Scarborough Extension).

## Tech Stack
- **RL Framework**: PufferLib (not stable-baselines3)
- **Environment**: Custom Gymnasium env (see `toronto_transit_env.py`)
- **Visualization**: FastHTML for interactive web demo
- **Graph Operations**: NetworkX
- **Geo Data**: OSMnx (when network available), synthetic grid otherwise
- **Data Sources**: Toronto Open Data, StatsCan census, TTC GTFS

## Project Structure
```
toronto-transit-rl/
├── CLAUDE.md
├── pyproject.toml
├── src/
│   ├── env/
│   │   ├── __init__.py
│   │   ├── toronto_env.py      # Gymnasium environment
│   │   ├── graph_builder.py    # Road/transit network construction
│   │   └── demand_model.py     # Gravity model for travel demand
│   ├── training/
│   │   ├── __init__.py
│   │   ├── puffer_config.py    # PufferLib training config
│   │   └── train.py            # Training entrypoint
│   ├── data/
│   │   ├── __init__.py
│   │   ├── gtfs_loader.py      # TTC GTFS parsing
│   │   ├── census_loader.py    # StatsCan density data
│   │   └── metrolinx.py        # Planned expansions for comparison
│   └── viz/
│       ├── __init__.py
│       ├── app.py              # FastHTML web app
│       ├── components.py       # UI components
│       └── map_render.py       # Folium/Leaflet map generation
├── data/
│   ├── gtfs/                   # TTC GTFS files
│   └── census/                 # StatsCan DA profiles
├── checkpoints/                # Trained model weights
└── tests/
    ├── test_env.py
    └── test_demand.py
```

## Current State
- [x] Basic Gymnasium environment working (`toronto_transit_env.py`)
- [x] Synthetic Toronto grid with density model
- [x] Existing TTC Lines 1 & 2 as baseline
- [x] TTT (travel time) computation via Dijkstra
- [x] Equity scoring (low-income zone access)
- [ ] PufferLib integration
- [ ] Real data loading (GTFS, census)
- [ ] FastHTML visualization
- [ ] Comparison against Metrolinx plans

## Key Design Decisions

### Environment
- **State**: Node features (density, has_transit, is_zone) + budget + current TTT
- **Actions**: Discrete - select node to add transit (connects to nearest existing)
- **Reward**: `ttt_improvement + equity_weight * equity_score - cost_penalty`
- **Episode**: 10-20 steps (budget cycle), terminates on budget exhaustion or "done" action

### Reward Shaping
```python
reward = (old_ttt - new_ttt) / baseline_ttt  # Normalized improvement
       + equity_weight * equity_score         # Access for underserved areas  
       - 0.001 * (cost / 1000)               # Cost penalty
```

### Costs (from Metrolinx estimates)
- Subway station: $500M
- Line extension: $200M/km
- Total budget per episode: $5-10B (realistic decade cycle)

## Commands

```bash
# Install dependencies
pip install pufferlib gymnasium networkx geopandas python-fasthtml folium

# Run environment test
python -m src.env.toronto_env

# Train agent
python -m src.training.train --steps 100000 --env toronto-transit-v0

# Launch viz demo
python -m src.viz.app
```

## Data Sources

| Data | Source | URL/Notes |
|------|--------|-----------|
| TTC GTFS | Transit.land | `f-dpz8-ttc` feed, download ZIP |
| Road Network | OSMnx | `ox.graph_from_place('Toronto, Canada')` |
| Population | StatsCan | 2021 Census DA profiles |
| Employment | Toronto Open Data | Employment Survey |
| Zoning | Toronto Open Data | Zoning By-law dataset |
| Planned Transit | Metrolinx | Ontario Line, SSE, Eglinton West |

## PufferLib Integration Notes

PufferLib expects:
1. Gymnasium-compatible env with `reset()`, `step()`, `observation_space`, `action_space`
2. Vectorized envs for speed - use `pufferlib.vector.make()`
3. Clean observation/action spaces (no nested dicts if possible, or flatten)

Current env uses Dict observation space - may need to flatten for PufferLib:
```python
# Option 1: Flatten in env
obs = np.concatenate([node_features.flatten(), [budget], [ttt], [steps]])

# Option 2: Use PufferLib's observation wrapper
```

## FastHTML Viz Plan

Minimal interactive demo:
1. **Map view**: Leaflet map of Toronto with existing TTC + agent proposals
2. **Controls**: Budget slider, "step" button, "reset" button
3. **Metrics**: TTT improvement %, equity score, cost spent
4. **Comparison toggle**: Show Metrolinx planned lines overlay

```python
# FastHTML skeleton
from fasthtml.common import *

app, rt = fast_app()

@rt("/")
def get():
    return Titled("Toronto Transit RL",
        Div(id="map"),  # Leaflet container
        Div(
            Button("Step", hx_post="/step"),
            Button("Reset", hx_post="/reset"),
            Div(id="metrics")
        )
    )
```

## Known Issues / TODOs
- [ ] GTFS join logic needs routes → trips → stop_times → stops chain
- [ ] OSMnx requires network access (blocked in some envs)
- [ ] Equity score is simplistic (uses density as income proxy)
- [ ] No multi-modal yet (subway only, no buses/LRT distinction)
- [ ] Action masking not integrated with PufferLib policy

## References
- PufferLib docs: https://pufferlib.com
- FastHTML docs: https://fastht.ml
- Toronto Open Data: https://open.toronto.ca
- Transit.land API: https://transit.land

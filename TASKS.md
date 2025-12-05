# Toronto Transit RL - Task Breakdown

## Epic 1: Environment Foundation
**Goal**: Robust, tested Gymnasium environment ready for PufferLib

### TASK-001: Refactor env to flat observation space
**Priority**: High | **Estimate**: 2h
**Description**: PufferLib works best with flat Box observations. Refactor the Dict space to a single flattened array.
**Acceptance Criteria**:
- [ ] `observation_space` is `spaces.Box` not `spaces.Dict`
- [ ] Obs shape is `(n_nodes * 3 + 3,)` = node features + budget + ttt + steps
- [ ] Existing test still passes
**Files**: `src/env/toronto_env.py`

### TASK-002: Add action masking support
**Priority**: High | **Estimate**: 2h
**Description**: Invalid actions (nodes with existing transit, over-budget) should be maskable. PufferLib supports action masks via `action_mask` in info dict or as part of observation.
**Acceptance Criteria**:
- [ ] `get_valid_actions()` returns boolean mask
- [ ] Mask included in observation or info dict
- [ ] Invalid actions return small negative reward (already done)
**Files**: `src/env/toronto_env.py`

### TASK-003: Extract graph builder to separate module
**Priority**: Medium | **Estimate**: 1h
**Description**: Move `_build_base_network()` logic to `graph_builder.py` for cleaner separation and easier testing.
**Acceptance Criteria**:
- [ ] `GraphBuilder` class with `build_synthetic()` and `build_from_osm()` methods
- [ ] Env imports and uses GraphBuilder
- [ ] Unit test for graph construction
**Files**: `src/env/graph_builder.py`, `src/env/toronto_env.py`

### TASK-004: Extract demand model to separate module
**Priority**: Medium | **Estimate**: 1h  
**Description**: Move TTT computation and gravity model to `demand_model.py`.
**Acceptance Criteria**:
- [ ] `DemandModel` class with `compute_ttt()` method
- [ ] Configurable gravity parameters
- [ ] Caching for repeated zone-pair queries
**Files**: `src/env/demand_model.py`, `src/env/toronto_env.py`

### TASK-005: Add comprehensive env tests
**Priority**: Medium | **Estimate**: 2h
**Description**: pytest suite for environment correctness.
**Acceptance Criteria**:
- [ ] Test reset returns valid observation
- [ ] Test step with valid action updates state correctly
- [ ] Test step with invalid action (duplicate node) returns penalty
- [ ] Test budget exhaustion terminates episode
- [ ] Test TTT decreases when adding transit in dense area
**Files**: `tests/test_env.py`

---

## Epic 2: PufferLib Training
**Goal**: Train an agent that beats random baseline

### TASK-006: PufferLib env wrapper
**Priority**: High | **Estimate**: 3h
**Description**: Create PufferLib-compatible wrapper. May need to handle vectorization and observation normalization.
**Acceptance Criteria**:
- [ ] `make_env()` function returns PufferLib-wrapped env
- [ ] Works with `pufferlib.vector.make()` for parallel envs
- [ ] Observation normalization if needed
**Files**: `src/training/puffer_config.py`
**Reference**: https://pufferlib.com/docs/environments

### TASK-007: Basic PPO training script
**Priority**: High | **Estimate**: 3h
**Description**: Training entrypoint using PufferLib's PPO.
**Acceptance Criteria**:
- [ ] CLI with `--steps`, `--lr`, `--num-envs` args
- [ ] Logs to wandb or tensorboard
- [ ] Saves checkpoints every N steps
- [ ] Prints episode reward stats
**Files**: `src/training/train.py`

### TASK-008: Hyperparameter config
**Priority**: Medium | **Estimate**: 1h
**Description**: YAML or dataclass config for training hyperparameters.
**Acceptance Criteria**:
- [ ] Config includes: lr, gamma, num_envs, batch_size, epochs, clip_coef
- [ ] Env params also configurable (budget, equity_weight, max_steps)
**Files**: `src/training/config.py` or `configs/default.yaml`

### TASK-009: Evaluation script
**Priority**: Medium | **Estimate**: 2h
**Description**: Load trained checkpoint, run N episodes, report metrics.
**Acceptance Criteria**:
- [ ] Loads model from checkpoint path
- [ ] Runs deterministic rollouts
- [ ] Reports: mean reward, mean TTT improvement, mean equity score
- [ ] Compares to random baseline
**Files**: `src/training/evaluate.py`

### TASK-010: Action mask integration with policy
**Priority**: High | **Estimate**: 2h
**Description**: Ensure PufferLib policy respects action masks during training and inference.
**Acceptance Criteria**:
- [ ] Masked actions have -inf logits
- [ ] No invalid actions taken during rollouts
**Files**: `src/training/puffer_config.py`

---

## Epic 3: Data Integration
**Goal**: Replace synthetic data with real Toronto data

### TASK-011: GTFS loader for TTC
**Priority**: Medium | **Estimate**: 4h
**Description**: Download and parse TTC GTFS feed. Handle the routes → trips → stop_times → stops join correctly.
**Acceptance Criteria**:
- [ ] Downloads from transit.land API
- [ ] Parses stops with lat/lon
- [ ] Extracts route geometries (shapes.txt)
- [ ] Returns GeoDataFrame of stops and lines
**Files**: `src/data/gtfs_loader.py`
**Note**: Use `gtfs_kit` library, specifically `feed.get_stop_times_by_route()`

### TASK-012: Census density loader
**Priority**: Medium | **Estimate**: 3h
**Description**: Load StatsCan 2021 census dissemination area data for population/income.
**Acceptance Criteria**:
- [ ] Downloads DA boundary shapefile
- [ ] Joins with population and income data
- [ ] Returns GeoDataFrame with density per DA
**Files**: `src/data/census_loader.py`

### TASK-013: OSMnx graph loader with fallback
**Priority**: Medium | **Estimate**: 2h
**Description**: Load real Toronto road network via OSMnx when network available, fall back to synthetic otherwise.
**Acceptance Criteria**:
- [ ] Tries OSMnx first with timeout
- [ ] Falls back to synthetic grid on failure
- [ ] Caches downloaded graph to disk
**Files**: `src/env/graph_builder.py`

### TASK-014: Metrolinx planned lines data
**Priority**: Low | **Estimate**: 2h
**Description**: Manually encode Ontario Line, Scarborough Extension, Eglinton West as reference lines for comparison.
**Acceptance Criteria**:
- [ ] GeoJSON or dict of planned station locations
- [ ] Function to overlay on map
- [ ] Function to compute "similarity" between agent proposal and planned
**Files**: `src/data/metrolinx.py`

---

## Epic 4: FastHTML Visualization
**Goal**: Interactive web demo to showcase trained agent

### TASK-015: Basic FastHTML app scaffold
**Priority**: High | **Estimate**: 2h
**Description**: Minimal FastHTML app with routing and static file serving.
**Acceptance Criteria**:
- [ ] Home page loads
- [ ] Static CSS/JS served
- [ ] HTMX included for interactivity
**Files**: `src/viz/app.py`

### TASK-016: Leaflet map component
**Priority**: High | **Estimate**: 3h
**Description**: Embed Leaflet map centered on Toronto. Show base road network and existing TTC.
**Acceptance Criteria**:
- [ ] Map renders in browser
- [ ] Existing TTC lines shown in blue
- [ ] Zoom/pan works
**Files**: `src/viz/components.py`, `src/viz/map_render.py`

### TASK-017: Step/Reset controls with HTMX
**Priority**: High | **Estimate**: 2h
**Description**: Buttons to step the agent and reset the environment. Use HTMX for partial page updates.
**Acceptance Criteria**:
- [ ] "Step" button calls `/step` endpoint, updates map + metrics
- [ ] "Reset" button calls `/reset` endpoint, resets env
- [ ] No full page reload
**Files**: `src/viz/app.py`

### TASK-018: Metrics display panel
**Priority**: Medium | **Estimate**: 1h
**Description**: Show current TTT, improvement %, equity score, budget remaining.
**Acceptance Criteria**:
- [ ] Updates after each step
- [ ] Visual indicator (green/red) for improvement direction
**Files**: `src/viz/components.py`

### TASK-019: Agent proposal rendering
**Priority**: High | **Estimate**: 2h
**Description**: Show agent's proposed new transit lines in red on the map.
**Acceptance Criteria**:
- [ ] New stations shown as red markers
- [ ] New lines shown as red polylines
- [ ] Popup on click shows cost and TTT impact
**Files**: `src/viz/map_render.py`

### TASK-020: Metrolinx comparison overlay
**Priority**: Low | **Estimate**: 2h
**Description**: Toggle to show planned Metrolinx lines alongside agent proposals.
**Acceptance Criteria**:
- [ ] Checkbox to enable/disable overlay
- [ ] Planned lines shown in green/dashed
- [ ] Legend distinguishes existing/agent/planned
**Files**: `src/viz/components.py`, `src/viz/map_render.py`

---

## Epic 5: Polish & Demo
**Goal**: Deployable demo for sharing

### TASK-021: Dockerfile
**Priority**: Low | **Estimate**: 1h
**Description**: Containerize the app for easy deployment.
**Files**: `Dockerfile`, `docker-compose.yml`

### TASK-022: README with demo GIF
**Priority**: Low | **Estimate**: 1h
**Description**: User-facing README with installation, usage, and a GIF of the demo.
**Files**: `README.md`

### TASK-023: Pre-trained checkpoint
**Priority**: Medium | **Estimate**: 1h (after training done)
**Description**: Include a trained checkpoint so users can run demo without training.
**Files**: `checkpoints/pretrained.pt`

---

## Suggested Sprint Plan

### Sprint 1 (MVP Environment + Training)
- TASK-001: Flat observation space
- TASK-002: Action masking
- TASK-006: PufferLib wrapper
- TASK-007: Basic training script
- TASK-005: Env tests

### Sprint 2 (Visualization)
- TASK-015: FastHTML scaffold
- TASK-016: Leaflet map
- TASK-017: Step/Reset controls
- TASK-018: Metrics display
- TASK-019: Agent proposal rendering

### Sprint 3 (Real Data)
- TASK-011: GTFS loader
- TASK-012: Census loader
- TASK-013: OSMnx with fallback
- TASK-003: Extract graph builder
- TASK-004: Extract demand model

### Sprint 4 (Polish)
- TASK-014: Metrolinx data
- TASK-020: Comparison overlay
- TASK-009: Evaluation script
- TASK-021: Dockerfile
- TASK-022: README

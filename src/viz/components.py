"""
UI Components
=============
Reusable FastHTML components for the visualization app.
"""

from fasthtml.common import *


def create_controls():
    """
    Create control panel with Step/Reset buttons.

    Returns:
        FastHTML component
    """
    return Div(
        H3("Controls"),
        Div(
            Button(
                "Step",
                hx_post="/step",
                hx_target="#map-container, #metrics-panel",
                hx_swap="innerHTML",
                cls="btn-step",
            ),
            Button(
                "Reset",
                hx_post="/reset",
                hx_target="#map-container, #metrics-panel",
                hx_swap="innerHTML",
                cls="btn-reset",
            ),
            cls="control-buttons",
        ),
        Div(
            Label(
                Input(
                    type="checkbox",
                    hx_post="/toggle-metrolinx",
                    hx_target="#map-container",
                    hx_swap="innerHTML",
                ),
                " Show Metrolinx Plans",
            ),
            cls="control-options",
        ),
        cls="control-group",
    )


def create_metrics_panel(env, info):
    """
    Create metrics display panel.

    Args:
        env: TorontoTransitEnv instance
        info: Latest step info dict

    Returns:
        FastHTML component
    """
    # Compute metrics
    baseline_ttt = info.get("baseline_ttt", env.baseline_ttt)
    current_ttt = env.current_ttt
    improvement_pct = (baseline_ttt - current_ttt) / baseline_ttt * 100

    equity_score = env._compute_equity_score()
    budget_remaining = env.budget
    budget_used = env.initial_budget - budget_remaining
    steps = env.steps
    new_stations = len(env.new_transit_nodes)

    # Determine if improvement is positive or negative
    improvement_class = "improvement" if improvement_pct > 0 else "decline"

    return Div(
        H3("Metrics"),
        # TTT Improvement
        Div(
            Div("Travel Time Improvement", cls="metric-label"),
            Div(
                f"{improvement_pct:+.1f}%",
                cls=f"metric-value {improvement_class}",
            ),
            cls="metric",
        ),
        # Equity Score
        Div(
            Div("Equity Score", cls="metric-label"),
            Div(f"{equity_score:.2f}", cls="metric-value"),
            cls="metric",
        ),
        # Budget
        Div(
            Div("Budget Used", cls="metric-label"),
            Div(f"${budget_used:.0f}M / ${env.initial_budget:.0f}M", cls="metric-value"),
            cls="metric",
        ),
        # Progress
        Div(
            Div("Progress", cls="metric-label"),
            Div(f"Step {steps} / {env.max_steps}", cls="metric-value"),
            cls="metric",
        ),
        # New Stations
        Div(
            Div("New Stations", cls="metric-label"),
            Div(f"{new_stations}", cls="metric-value"),
            cls="metric",
        ),
        cls="metrics-container",
    )


def create_legend():
    """
    Create map legend.

    Returns:
        HTML string for legend
    """
    return """
    <div class="map-legend" style="
        position: absolute;
        bottom: 30px;
        right: 10px;
        background: white;
        padding: 10px;
        border-radius: 4px;
        box-shadow: 0 1px 5px rgba(0,0,0,0.4);
        z-index: 1000;
    ">
        <div style="font-weight: bold; margin-bottom: 5px;">Legend</div>
        <div><span style="color: #3498db;">●</span> Existing Transit</div>
        <div><span style="color: #e74c3c;">●</span> New Stations</div>
        <div><span style="color: #27ae60;">●</span> Metrolinx Plans</div>
    </div>
    """


if __name__ == "__main__":
    print("Testing components...")

    # Test controls
    controls = create_controls()
    print("Controls created:", controls)

    # Test metrics (requires env)
    from src.env.toronto_env import TorontoTransitEnv

    env = TorontoTransitEnv(seed=42)
    obs, info = env.reset()

    metrics = create_metrics_panel(env, info)
    print("Metrics created:", metrics)

    print("\nComponents tests passed!")

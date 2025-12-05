"""
FastHTML Visualization App
===========================
Interactive web demo for Toronto Transit RL agent.
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fasthtml.common import *
from src.env.toronto_env import TorontoTransitEnv
from src.training.evaluate import RandomBaseline
from src.viz.components import create_metrics_panel, create_controls
from src.viz.map_render import render_map_html


# Global state (in production, use session management)
class AppState:
    def __init__(self):
        self.env = TorontoTransitEnv(
            grid_size=(30, 20),  # Smaller for demo
            n_zones=50,
            budget=5000.0,
            max_steps=15,
            seed=42,
        )
        self.policy = RandomBaseline()
        self.obs, self.info = self.env.reset(seed=42)
        self.show_metrolinx = False


app, rt = fast_app()
state = AppState()


@rt("/")
def get():
    """Main page."""
    return Titled(
        "Toronto Transit RL Demo",
        Div(
            H2("Interactive Transit Network Design"),
            P("Watch an RL agent propose new transit lines for Toronto."),
            cls="header",
        ),
        Div(
            # Map container
            Div(id="map-container", cls="map-section"),
            # Controls and metrics
            Div(
                create_controls(),
                Div(id="metrics-panel", cls="metrics-section"),
                cls="sidebar",
            ),
            cls="main-container",
        ),
        # Initialize map on page load
        Script(src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"),
        Link(
            rel="stylesheet",
            href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css",
        ),
        Script("htmx.on('htmx:afterSwap', function(evt) { if (window.updateMap) updateMap(); });"),
        # Trigger initial map and metrics load
        Script("htmx.trigger('#map-container', 'load'); htmx.trigger('#metrics-panel', 'load');"),
        Style("""
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 0;
                background: #f5f5f5;
            }
            .header {
                background: #2c3e50;
                color: white;
                padding: 1.5rem;
                text-align: center;
            }
            .main-container {
                display: flex;
                height: calc(100vh - 120px);
            }
            .map-section {
                flex: 1;
                padding: 1rem;
            }
            #map {
                width: 100%;
                height: 100%;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .sidebar {
                width: 350px;
                background: white;
                padding: 1.5rem;
                box-shadow: -2px 0 4px rgba(0,0,0,0.1);
                overflow-y: auto;
            }
            .control-group {
                margin-bottom: 1.5rem;
            }
            button {
                background: #3498db;
                color: white;
                border: none;
                padding: 0.75rem 1.5rem;
                border-radius: 4px;
                cursor: pointer;
                font-size: 1rem;
                margin: 0.5rem 0.5rem 0.5rem 0;
                transition: background 0.3s;
            }
            button:hover {
                background: #2980b9;
            }
            button:disabled {
                background: #95a5a6;
                cursor: not-allowed;
            }
            .metric {
                padding: 0.75rem;
                margin: 0.5rem 0;
                background: #ecf0f1;
                border-radius: 4px;
            }
            .metric-label {
                font-weight: bold;
                color: #7f8c8d;
                font-size: 0.875rem;
            }
            .metric-value {
                font-size: 1.5rem;
                color: #2c3e50;
                margin-top: 0.25rem;
            }
            .improvement {
                color: #27ae60;
            }
            .decline {
                color: #e74c3c;
            }
            label {
                display: block;
                margin: 0.5rem 0;
                cursor: pointer;
            }
        """),
    )


@rt("/map")
def get_map():
    """Render the map."""
    map_html = render_map_html(state.env, show_metrolinx=state.show_metrolinx)
    return Div(Raw(map_html), id="map")


@rt("/metrics")
def get_metrics():
    """Render metrics panel."""
    return create_metrics_panel(state.env, state.info)


@rt("/step", methods=["POST"])
def post_step():
    """Execute one step of the agent."""
    if state.env.steps >= state.env.max_steps or state.env.budget < state.env.station_cost:
        return (
            get_map(),
            create_metrics_panel(state.env, state.info),
        )

    # Get action from policy
    action_mask = state.env.get_action_mask()
    action = state.policy.predict(state.obs, action_mask)

    # Step environment
    state.obs, reward, terminated, truncated, state.info = state.env.step(action)

    # Return updated map and metrics
    return (
        get_map(),
        create_metrics_panel(state.env, state.info),
    )


@rt("/reset", methods=["POST"])
def post_reset():
    """Reset the environment."""
    state.obs, state.info = state.env.reset(seed=np.random.randint(0, 10000))

    return (
        get_map(),
        create_metrics_panel(state.env, state.info),
    )


@rt("/toggle-metrolinx", methods=["POST"])
def post_toggle_metrolinx():
    """Toggle Metrolinx overlay."""
    state.show_metrolinx = not state.show_metrolinx
    return get_map()


# Initialize on first load
@rt("/init-map")
def get_init_map():
    """Initialize map on page load."""
    return get_map()


@rt("/init-metrics")
def get_init_metrics():
    """Initialize metrics on page load."""
    return create_metrics_panel(state.env, state.info)


# Add HTMX triggers for initialization
@rt("/")
def get():
    """Main page with auto-initialization."""
    return Titled(
        "Toronto Transit RL Demo",
        Div(
            H2("Interactive Transit Network Design"),
            P("Watch an RL agent propose new transit lines for Toronto."),
            cls="header",
        ),
        Div(
            # Map container with auto-load
            Div(
                hx_get="/init-map",
                hx_trigger="load",
                hx_swap="innerHTML",
                id="map-container",
                cls="map-section",
            ),
            # Controls and metrics
            Div(
                create_controls(),
                Div(
                    hx_get="/init-metrics",
                    hx_trigger="load",
                    hx_swap="innerHTML",
                    id="metrics-panel",
                    cls="metrics-section",
                ),
                cls="sidebar",
            ),
            cls="main-container",
        ),
        # Scripts and styles
        Script(src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"),
        Link(
            rel="stylesheet",
            href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css",
        ),
        Style("""
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 0;
                background: #f5f5f5;
            }
            .header {
                background: #2c3e50;
                color: white;
                padding: 1.5rem;
                text-align: center;
            }
            .main-container {
                display: flex;
                height: calc(100vh - 120px);
            }
            .map-section {
                flex: 1;
                padding: 1rem;
            }
            #map {
                width: 100%;
                height: 100%;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .sidebar {
                width: 350px;
                background: white;
                padding: 1.5rem;
                box-shadow: -2px 0 4px rgba(0,0,0,0.1);
                overflow-y: auto;
            }
            .control-group {
                margin-bottom: 1.5rem;
            }
            button {
                background: #3498db;
                color: white;
                border: none;
                padding: 0.75rem 1.5rem;
                border-radius: 4px;
                cursor: pointer;
                font-size: 1rem;
                margin: 0.5rem 0.5rem 0.5rem 0;
                transition: background 0.3s;
            }
            button:hover {
                background: #2980b9;
            }
            button:disabled {
                background: #95a5a6;
                cursor: not-allowed;
            }
            .metric {
                padding: 0.75rem;
                margin: 0.5rem 0;
                background: #ecf0f1;
                border-radius: 4px;
            }
            .metric-label {
                font-weight: bold;
                color: #7f8c8d;
                font-size: 0.875rem;
            }
            .metric-value {
                font-size: 1.5rem;
                color: #2c3e50;
                margin-top: 0.25rem;
            }
            .improvement {
                color: #27ae60;
            }
            .decline {
                color: #e74c3c;
            }
            label {
                display: block;
                margin: 0.5rem 0;
                cursor: pointer;
            }
        """),
    )


def main():
    """Run the app."""
    serve()


if __name__ == "__main__":
    main()

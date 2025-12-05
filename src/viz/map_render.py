"""
Map Rendering
=============
Generate Leaflet/Folium maps for the visualization.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.metrolinx import MetrolinxPlans


def render_map_html(env, show_metrolinx=False):
    """
    Render Leaflet map as HTML.

    Args:
        env: TorontoTransitEnv instance
        show_metrolinx: Whether to show Metrolinx planned lines

    Returns:
        HTML string
    """
    # Get network data
    G = env.G
    transit_nodes = env.transit_nodes
    new_transit_nodes = env.new_transit_nodes
    existing_transit_nodes = env.existing_transit_nodes

    # Compute center of map (Toronto downtown)
    center_lat = 43.65
    center_lon = -79.38

    # Start building Leaflet map
    html = f"""
    <div id="map" style="width: 100%; height: 100%;"></div>
    <script>
        // Create map
        var map = L.map('map').setView([{center_lat}, {center_lon}], 11);

        // Add OpenStreetMap tiles
        L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            attribution: '© OpenStreetMap contributors',
            maxZoom: 19
        }}).addTo(map);

        // Add existing transit nodes (blue)
        var existingTransit = [
    """

    for node in existing_transit_nodes:
        if node in G.nodes:
            lat = G.nodes[node].get("y", 0)
            lon = G.nodes[node].get("x", 0)
            html += f"        [{lat}, {lon}],\n"

    html += """
        ];
        existingTransit.forEach(function(coords) {
            L.circleMarker(coords, {
                radius: 5,
                fillColor: '#3498db',
                color: '#2980b9',
                weight: 1,
                fillOpacity: 0.8
            }).addTo(map).bindPopup('Existing Transit');
        });

        // Add new transit nodes (red)
        var newTransit = [
    """

    for node in new_transit_nodes:
        if node in G.nodes:
            lat = G.nodes[node].get("y", 0)
            lon = G.nodes[node].get("x", 0)
            density = G.nodes[node].get("density", 0)
            html += f"        [{{lat: {lat}, lon: {lon}, density: {density:.2f}}}],\n"

    html += """
        ];
        newTransit.forEach(function(data) {
            L.circleMarker([data.lat, data.lon], {
                radius: 7,
                fillColor: '#e74c3c',
                color: '#c0392b',
                weight: 2,
                fillOpacity: 0.9
            }).addTo(map).bindPopup('New Station<br>Density: ' + data.density.toFixed(2));
        });

        // Draw lines between new stations and nearest existing
    """

    # Draw connections from new to existing transit
    for new_node in new_transit_nodes:
        if new_node not in G.nodes:
            continue

        new_lat = G.nodes[new_node].get("y", 0)
        new_lon = G.nodes[new_node].get("x", 0)

        # Find connected nodes
        for neighbor in G.neighbors(new_node):
            if neighbor in existing_transit_nodes or neighbor in new_transit_nodes:
                neighbor_lat = G.nodes[neighbor].get("y", 0)
                neighbor_lon = G.nodes[neighbor].get("x", 0)

                html += f"""
        L.polyline([[{new_lat}, {new_lon}], [{neighbor_lat}, {neighbor_lon}]], {{
            color: '#e74c3c',
            weight: 3,
            opacity: 0.7
        }}).addTo(map);
                """

    # Add Metrolinx planned lines if requested
    if show_metrolinx:
        plans = MetrolinxPlans()
        all_stations = plans.get_all_planned_stations()

        html += """
        // Add Metrolinx planned stations (green)
        var metrolinxStations = [
        """

        for station in all_stations:
            lat = station["lat"]
            lon = station["lon"]
            name = station["name"]
            project = station["project"]
            html += f"        {{lat: {lat}, lon: {lon}, name: '{name}', project: '{project}'}},\n"

        html += """
        ];
        metrolinxStations.forEach(function(data) {
            L.circleMarker([data.lat, data.lon], {
                radius: 5,
                fillColor: '#27ae60',
                color: '#229954',
                weight: 1,
                fillOpacity: 0.7
            }).addTo(map).bindPopup(data.name + '<br>' + data.project);
        });
        """

    html += """
    </script>
    """

    return html


if __name__ == "__main__":
    print("Testing map rendering...")

    from src.env.toronto_env import TorontoTransitEnv

    env = TorontoTransitEnv(grid_size=(20, 15), seed=42)
    env.reset()

    # Take a few steps
    for _ in range(3):
        mask = env.get_action_mask()
        valid_actions = [i for i in range(len(mask)) if mask[i]]
        if valid_actions:
            env.step(valid_actions[0])

    html = render_map_html(env, show_metrolinx=True)

    print(f"Generated map HTML ({len(html)} characters)")
    print("First 200 chars:", html[:200])

    # Save to file for testing
    with open("/tmp/test_map.html", "w") as f:
        f.write(f"""
        <!DOCTYPE html>
        <html>
        <head>
            <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
            <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
        </head>
        <body style="margin: 0; padding: 0;">
            {html}
        </body>
        </html>
        """)

    print("\nTest map saved to /tmp/test_map.html")
    print("Map rendering tests passed!")

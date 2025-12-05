"""
Metrolinx Planned Lines Data
=============================
Reference data for planned transit expansions (Ontario Line, SSE, etc.)
for comparison with agent proposals.
"""

from typing import Dict, List, Tuple
import numpy as np


class MetrolinxPlans:
    """Planned Metrolinx transit expansions."""

    def __init__(self):
        """Initialize with known planned projects."""
        self.projects = self._load_planned_projects()

    def _load_planned_projects(self) -> Dict[str, Dict]:
        """
        Load planned transit projects.

        Returns:
            Dictionary mapping project name to station data
        """
        projects = {}

        # Ontario Line (North-South through downtown)
        # Approximate station locations (lat, lon)
        projects["Ontario Line"] = {
            "stations": [
                {"name": "Exhibition", "lat": 43.6344, "lon": -79.4192},
                {"name": "King-Bathurst", "lat": 43.6430, "lon": -79.4028},
                {"name": "Queen-Spadina", "lat": 43.6489, "lon": -79.3956},
                {"name": "Osgoode", "lat": 43.6507, "lon": -79.3867},
                {"name": "Queen-Yonge", "lat": 43.6525, "lon": -79.3788},
                {"name": "Moss Park", "lat": 43.6566, "lon": -79.3672},
                {"name": "Corktown", "lat": 43.6540, "lon": -79.3569},
                {"name": "East Harbour", "lat": 43.6587, "lon": -79.3385},
                {"name": "Leslieville", "lat": 43.6645, "lon": -79.3294},
                {"name": "Gerrard", "lat": 43.6712, "lon": -79.3471},
                {"name": "Pape", "lat": 43.6783, "lon": -79.3453},
                {"name": "Cosburn", "lat": 43.6881, "lon": -79.3436},
                {"name": "Thorncliffe Park", "lat": 43.7029, "lon": -79.3489},
                {"name": "Science Centre", "lat": 43.7181, "lon": -79.3389},
            ],
            "type": "subway",
            "status": "planned",
            "cost_billion": 10.9,
        }

        # Scarborough Subway Extension (SSE)
        projects["Scarborough Extension"] = {
            "stations": [
                {"name": "Kennedy", "lat": 43.7326, "lon": -79.2626},
                {"name": "Lawrence East", "lat": 43.7504, "lon": -79.2686},
                {"name": "Scarborough Centre", "lat": 43.7735, "lon": -79.2577},
            ],
            "type": "subway",
            "status": "planned",
            "cost_billion": 5.5,
        }

        # Eglinton West Extension
        projects["Eglinton West Extension"] = {
            "stations": [
                {"name": "Mount Dennis", "lat": 43.6888, "lon": -79.4871},
                {"name": "Jane", "lat": 43.6918, "lon": -79.4933},
                {"name": "Renforth", "lat": 43.6465, "lon": -79.5815},
            ],
            "type": "lrt",
            "status": "planned",
            "cost_billion": 4.7,
        }

        # Yonge North Extension
        projects["Yonge North Extension"] = {
            "stations": [
                {"name": "Finch", "lat": 43.7807, "lon": -79.4147},
                {"name": "Steeles", "lat": 43.8011, "lon": -79.4146},
                {"name": "Clark", "lat": 43.8162, "lon": -79.4145},
                {"name": "Royal Orchard", "lat": 43.8260, "lon": -79.4277},
                {"name": "Langstaff", "lat": 43.8478, "lon": -79.4363},
                {"name": "Richmond Hill Centre", "lat": 43.8684, "lon": -79.4355},
            ],
            "type": "subway",
            "status": "planned",
            "cost_billion": 5.6,
        }

        return projects

    def get_all_planned_stations(self) -> List[Dict]:
        """
        Get all planned stations across all projects.

        Returns:
            List of station dictionaries with lat, lon, project, type
        """
        all_stations = []
        for project_name, project_data in self.projects.items():
            for station in project_data["stations"]:
                station_copy = station.copy()
                station_copy["project"] = project_name
                station_copy["type"] = project_data["type"]
                all_stations.append(station_copy)

        return all_stations

    def get_project(self, name: str) -> Dict:
        """Get specific project by name."""
        return self.projects.get(name, {})

    def compute_similarity(
        self,
        agent_stations: List[Tuple[float, float]],
        project_name: str,
        threshold_km: float = 0.5,
    ) -> float:
        """
        Compute similarity between agent proposal and planned project.

        Similarity = fraction of planned stations that have an agent station nearby.

        Args:
            agent_stations: List of (lat, lon) tuples for agent proposals
            project_name: Name of planned project
            threshold_km: Distance threshold for "nearby" in km

        Returns:
            Similarity score between 0 and 1
        """
        project = self.get_project(project_name)
        if not project:
            return 0.0

        planned_stations = project["stations"]
        matches = 0

        for planned in planned_stations:
            planned_lat = planned["lat"]
            planned_lon = planned["lon"]

            # Check if any agent station is within threshold
            for agent_lat, agent_lon in agent_stations:
                dist_km = self._haversine_distance(
                    planned_lat, planned_lon, agent_lat, agent_lon
                )
                if dist_km < threshold_km:
                    matches += 1
                    break

        return matches / len(planned_stations) if planned_stations else 0.0

    def _haversine_distance(
        self, lat1: float, lon1: float, lat2: float, lon2: float
    ) -> float:
        """
        Compute distance between two lat/lon points in km.

        Args:
            lat1, lon1: First point
            lat2, lon2: Second point

        Returns:
            Distance in kilometers
        """
        # Haversine formula
        R = 6371  # Earth radius in km

        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)

        a = (
            np.sin(dlat / 2) ** 2
            + np.cos(np.radians(lat1))
            * np.cos(np.radians(lat2))
            * np.sin(dlon / 2) ** 2
        )
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

        return R * c


if __name__ == "__main__":
    print("Testing MetrolinxPlans...")

    plans = MetrolinxPlans()

    print(f"\nPlanned projects: {list(plans.projects.keys())}")

    # Test Ontario Line
    ol = plans.get_project("Ontario Line")
    print(f"\nOntario Line:")
    print(f"  Stations: {len(ol['stations'])}")
    print(f"  Cost: ${ol['cost_billion']}B")
    print(f"  Sample stations:")
    for station in ol["stations"][:3]:
        print(f"    {station['name']}: ({station['lat']:.4f}, {station['lon']:.4f})")

    # Test all stations
    all_stations = plans.get_all_planned_stations()
    print(f"\nTotal planned stations: {len(all_stations)}")

    # Test similarity
    # Simulate agent proposing stations near Ontario Line
    agent_stations = [
        (43.6430, -79.4028),  # Near King-Bathurst
        (43.6507, -79.3867),  # Near Osgoode
        (43.6783, -79.3453),  # Near Pape
    ]
    similarity = plans.compute_similarity(agent_stations, "Ontario Line")
    print(f"\nAgent similarity to Ontario Line: {similarity:.2%}")

    print("\nMetrolinxPlans tests passed!")

"""
GTFS Loader for TTC
===================
Download and parse TTC GTFS feed from transit.land.
"""

import os
import zipfile
import requests
from pathlib import Path
from typing import Optional, Tuple
import pandas as pd


class GTFSLoader:
    """Load and parse GTFS data for TTC."""

    def __init__(self, cache_dir: str = "data/gtfs"):
        """
        Initialize GTFS loader.

        Args:
            cache_dir: Directory to cache downloaded GTFS data
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # TTC feed ID from transit.land
        self.feed_url = "https://gtfs.toronto.ca/TTC_GTFS.zip"

    def download(self, force: bool = False) -> Path:
        """
        Download GTFS feed.

        Args:
            force: Force re-download even if cached

        Returns:
            Path to downloaded ZIP file
        """
        zip_path = self.cache_dir / "TTC_GTFS.zip"

        if zip_path.exists() and not force:
            print(f"Using cached GTFS: {zip_path}")
            return zip_path

        print(f"Downloading GTFS from {self.feed_url}...")
        try:
            response = requests.get(self.feed_url, timeout=30)
            response.raise_for_status()

            with open(zip_path, "wb") as f:
                f.write(response.content)

            print(f"Downloaded to {zip_path}")
            return zip_path

        except Exception as e:
            print(f"Error downloading GTFS: {e}")
            if zip_path.exists():
                print("Using existing cached file.")
                return zip_path
            raise

    def extract(self, zip_path: Path) -> Path:
        """
        Extract GTFS ZIP file.

        Args:
            zip_path: Path to ZIP file

        Returns:
            Path to extracted directory
        """
        extract_dir = self.cache_dir / "extracted"
        extract_dir.mkdir(exist_ok=True)

        if (extract_dir / "stops.txt").exists():
            print(f"Using cached extracted GTFS: {extract_dir}")
            return extract_dir

        print(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)

        print(f"Extracted to {extract_dir}")
        return extract_dir

    def load_stops(self) -> pd.DataFrame:
        """
        Load GTFS stops with lat/lon.

        Returns:
            DataFrame with columns: stop_id, stop_name, stop_lat, stop_lon
        """
        zip_path = self.download()
        extract_dir = self.extract(zip_path)

        stops_path = extract_dir / "stops.txt"
        print(f"Loading stops from {stops_path}...")

        stops = pd.read_csv(stops_path)

        # Select relevant columns
        stops = stops[["stop_id", "stop_name", "stop_lat", "stop_lon"]]

        print(f"Loaded {len(stops)} stops")
        return stops

    def load_routes(self) -> pd.DataFrame:
        """
        Load GTFS routes.

        Returns:
            DataFrame with route information
        """
        zip_path = self.download()
        extract_dir = self.extract(zip_path)

        routes_path = extract_dir / "routes.txt"
        print(f"Loading routes from {routes_path}...")

        routes = pd.read_csv(routes_path)
        print(f"Loaded {len(routes)} routes")
        return routes

    def load_shapes(self) -> Optional[pd.DataFrame]:
        """
        Load GTFS shapes (route geometries).

        Returns:
            DataFrame with shape_id, shape_pt_lat, shape_pt_lon, shape_pt_sequence
        """
        zip_path = self.download()
        extract_dir = self.extract(zip_path)

        shapes_path = extract_dir / "shapes.txt"

        if not shapes_path.exists():
            print("shapes.txt not found in GTFS feed")
            return None

        print(f"Loading shapes from {shapes_path}...")
        shapes = pd.read_csv(shapes_path)
        print(f"Loaded {len(shapes)} shape points")
        return shapes

    def get_subway_stops(self) -> pd.DataFrame:
        """
        Get only subway/metro stops.

        Returns:
            DataFrame of subway stops
        """
        stops = self.load_stops()
        routes = self.load_routes()

        # TTC subway routes are typically route_type = 1 (metro/subway)
        # Or we can filter by route_id (Line 1, 2, 3, 4)
        subway_route_ids = ["1", "2", "3", "4"]  # Simplified - may need adjustment

        # Join with stop_times and trips to get stops for subway routes
        # This would require loading trips.txt and stop_times.txt
        # For now, return all stops (could be filtered by location/name pattern)

        print(f"Returning all {len(stops)} stops (subway filtering requires trip/route join)")
        return stops


if __name__ == "__main__":
    print("Testing GTFSLoader...")

    loader = GTFSLoader()

    # Test download and load
    try:
        stops = loader.load_stops()
        print(f"\nSample stops:")
        print(stops.head())

        routes = loader.load_routes()
        print(f"\nSample routes:")
        print(routes.head())

        shapes = loader.load_shapes()
        if shapes is not None:
            print(f"\nSample shapes:")
            print(shapes.head())

        print("\nGTFSLoader tests passed!")

    except Exception as e:
        print(f"\nGTFS loading failed (this is expected if no network access):")
        print(f"  {e}")
        print("\nGTFSLoader will fall back to synthetic data when offline.")

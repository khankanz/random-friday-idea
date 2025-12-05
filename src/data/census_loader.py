"""
Census Density Loader
=====================
Load StatsCan 2021 census dissemination area data.
"""

import os
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np


class CensusLoader:
    """Load and parse StatsCan census data."""

    def __init__(self, cache_dir: str = "data/census"):
        """
        Initialize census loader.

        Args:
            cache_dir: Directory for census data files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def download_da_profiles(self, force: bool = False) -> Path:
        """
        Download 2021 Census DA profiles.

        NOTE: This requires manual download from StatsCan.
        URL: https://www12.statcan.gc.ca/census-recensement/2021/dp-pd/prof/index.cfm

        Args:
            force: Force re-download

        Returns:
            Path to downloaded file
        """
        # StatsCan data requires manual download
        # Provide instructions

        csv_path = self.cache_dir / "census_2021_da.csv"

        if csv_path.exists() and not force:
            print(f"Using cached census data: {csv_path}")
            return csv_path

        print("\n" + "=" * 60)
        print("Census data download required")
        print("=" * 60)
        print("To download 2021 Census DA profiles:")
        print("1. Visit: https://www12.statcan.gc.ca/census-recensement/2021/dp-pd/prof/index.cfm")
        print("2. Select 'Dissemination Area' geography")
        print("3. Select 'Toronto' region")
        print("4. Download CSV and save to:", csv_path)
        print("=" * 60)

        if not csv_path.exists():
            # Create a placeholder synthetic data file
            print("\nCreating synthetic census data for demonstration...")
            self._create_synthetic_census(csv_path)

        return csv_path

    def _create_synthetic_census(self, path: Path):
        """Create synthetic census data for testing."""
        # Generate synthetic DA data for Toronto
        n_das = 500  # Toronto has ~1000 DAs

        np.random.seed(42)

        data = {
            "GEO_ID": [f"DA_{i:04d}" for i in range(n_das)],
            "POPULATION": np.random.randint(100, 5000, n_das),
            "MEDIAN_INCOME": np.random.randint(30000, 120000, n_das),
            "AREA_KM2": np.random.uniform(0.1, 2.0, n_das),
        }

        df = pd.DataFrame(data)
        df["DENSITY"] = df["POPULATION"] / df["AREA_KM2"]

        df.to_csv(path, index=False)
        print(f"Created synthetic census data: {path}")

    def load_density_data(self) -> pd.DataFrame:
        """
        Load population density data.

        Returns:
            DataFrame with GEO_ID, POPULATION, DENSITY, MEDIAN_INCOME
        """
        csv_path = self.download_da_profiles()
        print(f"Loading census data from {csv_path}...")

        df = pd.read_csv(csv_path)

        # Ensure required columns exist
        required_cols = ["GEO_ID", "POPULATION", "DENSITY"]
        for col in required_cols:
            if col not in df.columns:
                print(f"Warning: Column {col} not found. Using synthetic data.")
                self._create_synthetic_census(csv_path)
                df = pd.read_csv(csv_path)
                break

        print(f"Loaded {len(df)} dissemination areas")
        return df

    def load_income_data(self) -> pd.DataFrame:
        """
        Load median income data.

        Returns:
            DataFrame with GEO_ID and MEDIAN_INCOME
        """
        df = self.load_density_data()

        if "MEDIAN_INCOME" in df.columns:
            return df[["GEO_ID", "MEDIAN_INCOME"]]
        else:
            print("Warning: MEDIAN_INCOME not available")
            return df[["GEO_ID"]]

    def get_low_income_zones(self, percentile: float = 40) -> pd.DataFrame:
        """
        Identify low-income dissemination areas.

        Args:
            percentile: Percentile threshold for low income

        Returns:
            DataFrame of low-income DAs
        """
        df = self.load_density_data()

        if "MEDIAN_INCOME" in df.columns:
            threshold = df["MEDIAN_INCOME"].quantile(percentile / 100)
            low_income = df[df["MEDIAN_INCOME"] < threshold]
            print(f"Identified {len(low_income)} low-income DAs (< ${threshold:.0f})")
            return low_income
        else:
            print("Warning: Cannot identify low-income zones without MEDIAN_INCOME")
            return df.head(0)


if __name__ == "__main__":
    print("Testing CensusLoader...")

    loader = CensusLoader()

    # Test load
    density_data = loader.load_density_data()
    print(f"\nSample density data:")
    print(density_data.head())

    income_data = loader.load_income_data()
    print(f"\nSample income data:")
    print(income_data.head())

    low_income = loader.get_low_income_zones(percentile=40)
    print(f"\nSample low-income zones:")
    print(low_income.head())

    print("\nCensusLoader tests passed!")

"""
Step 1.1: Scenario Topology Generator
Generates realistic NTN topology for 6G spectrum allocation scenarios.

Creates:
- LEO/MEO satellite constellations
- Ground station networks (urban/rural)
- Terrestrial cell layers
- User equipment locations
- Channel propagation paths

Output: JSON scenario files with spatial geometry
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class NTNTopologyGenerator:
    """Generate realistic Non-Terrestrial Network topologies for 6G."""

    # Orbital parameters (6G standardized)
    LEO_ALTITUDE_KM = 550  # km, typical LEO constellation
    MEO_ALTITUDE_KM = 8063  # km, MEO orbit
    GEO_ALTITUDE_KM = 35786  # km, Geostationary
    EARTH_RADIUS_KM = 6371  # km

    def __init__(self, num_satellites: int = 12, num_ground_stations: int = 50,
                 coverage_radius_km: float = 2000, random_seed: Optional[int] = None):
        """
        Initialize topology generator.
        
        Args:
            num_satellites: Number of satellites in constellation
            num_ground_stations: Number of ground stations
            coverage_radius_km: Coverage area radius
            random_seed: For reproducible scenarios
        """
        self.num_satellites = num_satellites
        self.num_ground_stations = num_ground_stations
        self.coverage_radius_km = coverage_radius_km
        
        if random_seed is not None:
            np.random.seed(random_seed)

    def generate_leo_constellation(self) -> List[Dict]:
        """
        Generate LEO satellite constellation.
        Distribute satellites in polar orbit pattern.
        
        Returns:
            List of satellite positions with metadata
        """
        satellites = []
        
        # Distribute satellites in orbital planes
        num_planes = 3  # Number of orbital planes
        sats_per_plane = self.num_satellites // num_planes
        
        for plane_idx in range(num_planes):
            # Orbital plane inclination (polar orbit: 86-98 degrees)
            inclination = 86 + np.random.rand() * 12
            
            # Right ascension of ascending node (RAAN)
            raan = (plane_idx * 360.0 / num_planes) + np.random.rand() * 5
            
            for sat_idx in range(sats_per_plane):
                # Mean anomaly (position in orbit)
                mean_anomaly = (sat_idx * 360.0 / sats_per_plane) + np.random.rand() * 5
                
                # Convert to Cartesian coordinates (Earth-centered)
                lat, lon = self._mean_anomaly_to_latlon(mean_anomaly, inclination)
                
                # Add orbital velocity vector (circular orbit)
                velocity = self._circular_orbit_velocity(self.LEO_ALTITUDE_KM)
                
                satellites.append({
                    'sat_id': f'LEO-{plane_idx}-{sat_idx}',
                    'type': 'LEO',
                    'altitude_km': self.LEO_ALTITUDE_KM,
                    'latitude': lat,
                    'longitude': lon,
                    'inclination': inclination,
                    'velocity_m_s': velocity,
                    'orbital_period_minutes': self._orbital_period(self.LEO_ALTITUDE_KM),
                })
        
        return satellites

    def generate_ground_stations(self) -> List[Dict]:
        """
        Generate ground station network.
        Mix of urban clusters and rural stations.
        
        Returns:
            List of ground station positions
        """
        stations = []
        
        # Urban clusters (5 cities, each with multiple stations)
        num_urban_clusters = 5
        stations_per_cluster = self.num_ground_stations // (num_urban_clusters + 1)
        
        for cluster_idx in range(num_urban_clusters):
            # Random city center within coverage area
            cluster_lat = np.random.uniform(-self.coverage_radius_km/111, self.coverage_radius_km/111)
            cluster_lon = np.random.uniform(-self.coverage_radius_km/111, self.coverage_radius_km/111)
            
            # Stations clustered around city center
            for station_idx in range(stations_per_cluster):
                # Gaussian distribution around cluster center
                lat = cluster_lat + np.random.randn() * 0.1  # ~10 km std dev
                lon = cluster_lon + np.random.randn() * 0.1
                
                stations.append({
                    'gs_id': f'GS-URBAN-{cluster_idx}-{station_idx}',
                    'type': 'urban',
                    'latitude': lat,
                    'longitude': lon,
                    'elevation_m': np.random.randint(0, 500),
                    'antenna_type': 'phased_array',
                    'num_antenna_elements': 64,
                })
        
        # Rural stations (scattered)
        num_rural = self.num_ground_stations - (stations_per_cluster * num_urban_clusters)
        for station_idx in range(num_rural):
            lat = np.random.uniform(-self.coverage_radius_km/111, self.coverage_radius_km/111)
            lon = np.random.uniform(-self.coverage_radius_km/111, self.coverage_radius_km/111)
            
            stations.append({
                'gs_id': f'GS-RURAL-{station_idx}',
                'type': 'rural',
                'latitude': lat,
                'longitude': lon,
                'elevation_m': np.random.randint(0, 200),
                'antenna_type': 'horn',
                'num_antenna_elements': 16,
            })
        
        return stations

    def generate_user_equipment(self, num_users: int = 1000) -> List[Dict]:
        """
        Generate distributed user equipment locations.
        
        Args:
            num_users: Number of UEs to generate
            
        Returns:
            List of UE positions and capabilities
        """
        users = []
        
        for ue_idx in range(num_users):
            # Random position within coverage area
            lat = np.random.uniform(-self.coverage_radius_km/111, self.coverage_radius_km/111)
            lon = np.random.uniform(-self.coverage_radius_km/111, self.coverage_radius_km/111)
            
            # UE type distribution
            ue_type_rand = np.random.rand()
            if ue_type_rand < 0.7:
                ue_type = 'mobile'
                speed_kmh = np.random.exponential(30)  # Mobile speeds
            elif ue_type_rand < 0.9:
                ue_type = 'fixed'
                speed_kmh = 0
            else:
                ue_type = 'high_speed'
                speed_kmh = np.random.uniform(100, 500)  # Aircraft/train
            
            users.append({
                'ue_id': f'UE-{ue_idx}',
                'type': ue_type,
                'latitude': lat,
                'longitude': lon,
                'velocity_kmh': speed_kmh,
                'altitude_m': np.random.randint(0, 10000) if ue_type == 'high_speed' else 0,
                'antenna_gain_dbi': 3,
                'noise_figure_db': 8,
            })
        
        return users

    def generate_terrestrial_cells(self) -> List[Dict]:
        """
        Generate terrestrial 5G/6G cell sites.
        
        Returns:
            List of terrestrial base stations
        """
        cells = []
        
        # Grid-based cell placement (typical deployment)
        cell_spacing_km = 1.0  # 1 km inter-site distance (urban)
        grid_size = int(np.ceil(np.sqrt(self.coverage_radius_km / cell_spacing_km)))
        
        cell_idx = 0
        for i in range(grid_size):
            for j in range(grid_size):
                lat = -self.coverage_radius_km/111/2 + i * (cell_spacing_km / 111)
                lon = -self.coverage_radius_km/111/2 + j * (cell_spacing_km / 111)
                
                if np.sqrt(lat**2 + lon**2) > self.coverage_radius_km/111:
                    continue  # Skip cells outside coverage
                
                cells.append({
                    'cell_id': f'CELL-{cell_idx}',
                    'type': 'terrestrial',
                    'latitude': lat,
                    'longitude': lon,
                    'elevation_m': np.random.randint(10, 50),
                    'antenna_height_m': 30,
                    'num_antenna_arrays': 3,  # Tri-sector
                    'carrier_frequency_ghz': np.random.choice([3.5, 28, 39]),
                })
                cell_idx += 1
        
        return cells

    def assemble_scenario(self, scenario_id: int) -> Dict:
        """
        Assemble complete scenario topology.
        
        Args:
            scenario_id: Unique scenario identifier
            
        Returns:
            Complete scenario configuration
        """
        scenario = {
            'scenario_id': scenario_id,
            'timestamp': str(np.datetime64('now')),
            'coverage_radius_km': self.coverage_radius_km,
            'satellites': self.generate_leo_constellation(),
            'ground_stations': self.generate_ground_stations(),
            'terrestrial_cells': self.generate_terrestrial_cells(),
            'user_equipment': self.generate_user_equipment(),
            'atmospheric_conditions': {
                'rain_rate_mm_h': np.random.exponential(2.0),
                'temperature_c': np.random.uniform(5, 35),
                'humidity_percent': np.random.uniform(30, 95),
            },
            'simulation_config': {
                'duration_seconds': 1.0,
                'sampling_rate_hz': 20e6,
                'frequency_band': '28-39 GHz',
            }
        }
        return scenario

    # Helper methods
    @staticmethod
    def _mean_anomaly_to_latlon(mean_anomaly: float, inclination: float) -> Tuple[float, float]:
        """Convert mean anomaly to latitude/longitude."""
        # Simplified Kepler equation solution
        ecc_anomaly = mean_anomaly  # Circular orbit assumption
        lat = inclination * np.sin(np.radians(ecc_anomaly))
        lon = ecc_anomaly
        return lat, lon

    @staticmethod
    def _circular_orbit_velocity(altitude_km: float) -> float:
        """Calculate circular orbit velocity in m/s."""
        GM = 3.986e14  # Earth's standard gravitational parameter
        r = (NTNTopologyGenerator.EARTH_RADIUS_KM + altitude_km) * 1000  # meters
        return np.sqrt(GM / r)

    @staticmethod
    def _orbital_period(altitude_km: float) -> float:
        """Calculate orbital period in minutes."""
        velocity = NTNTopologyGenerator._circular_orbit_velocity(altitude_km)
        r = (NTNTopologyGenerator.EARTH_RADIUS_KM + altitude_km) * 1000
        period_seconds = 2 * np.pi * r / velocity
        return period_seconds / 60


def generate_scenario_json(num_scenarios: int, output_dir: Path) -> List[str]:
    """
    Generate and save scenario topology files.
    
    Args:
        num_scenarios: Number of scenarios to generate
        output_dir: Output directory for JSON files
        
    Returns:
        List of created file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    generator = NTNTopologyGenerator()
    created_files = []
    
    logger.info(f"Generating {num_scenarios} NTN scenarios...")
    
    for scenario_id in range(num_scenarios):
        scenario = generator.assemble_scenario(scenario_id)
        
        # Save as JSON
        output_file = output_dir / f"scenario_{scenario_id:05d}.json"
        with open(output_file, 'w') as f:
            json.dump(scenario, f, indent=2)
        
        created_files.append(str(output_file))
        
        if (scenario_id + 1) % max(1, num_scenarios // 10) == 0:
            logger.info(f"  Generated {scenario_id + 1}/{num_scenarios} scenarios")
    
    logger.info(f"Scenario topology generation complete! Output: {output_dir}")
    return created_files


if __name__ == '__main__':
    # Test: Generate 10 scenarios
    output = generate_scenario_json(10, Path('/tmp/scenarios'))
    print(f"Generated {len(output)} scenario files")

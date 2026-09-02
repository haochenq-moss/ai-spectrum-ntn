"""
Step 1.2: Channel Generation
Simulates wireless channel state information (CSI) using Sionna for 6G NTN.

Creates:
- Path loss models (free space + atmospheric)
- Fading channels (Rayleigh/Rician)
- Delay profiles
- Doppler effects from satellite motion
- Rain attenuation model

Output: HDF5 files with CSI matrices
"""

import numpy as np
import h5py
import json
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)

try:
    import sionna
    SIONNA_AVAILABLE = True
except ImportError:
    SIONNA_AVAILABLE = False
    logger.warning("Sionna not available; using fallback channel model")


@dataclass
class ChannelConfig:
    """Channel simulation configuration."""
    frequency_ghz: float = 28.0
    bandwidth_mhz: float = 400.0
    num_subcarriers: int = 512
    num_ofdm_symbols: int = 14
    num_antennas_tx: int = 64
    num_antennas_rx: int = 64
    max_links_per_scenario: int = 16
    rician_k_factor: float = 5.0
    rain_rate_mm_h: float = 2.0
    temperature_c: float = 20.0


class ChannelSimulator:
    """Simulate realistic wireless channels for NTN scenarios."""

    def __init__(self, config: Optional[ChannelConfig] = None):
        """Initialize channel simulator."""
        self.config = config or ChannelConfig()
        self.wavelength_m = 3e8 / (self.config.frequency_ghz * 1e9)

    def calculate_free_space_path_loss(self, distance_m: float) -> float:
        """
        Friis free space path loss formula.
        
        Args:
            distance_m: Distance in meters
            
        Returns:
            Path loss in dB
        """
        if distance_m <= 0:
            return 0
        
        # Friis formula: PL = 20*log10(4*pi*d/lambda)
        numerator = 4 * np.pi * distance_m
        denominator = self.wavelength_m
        path_loss_db = 20 * np.log10(numerator / denominator)
        
        return path_loss_db

    def calculate_atmospheric_attenuation(self, distance_m: float, 
                                        rain_rate_mm_h: float) -> float:
        """
        Calculate atmospheric attenuation at mmWave frequencies.
        Includes oxygen/water vapor absorption and rain attenuation.
        
        Args:
            distance_m: Distance in meters
            rain_rate_mm_h: Rain rate in mm/h
            
        Returns:
            Attenuation in dB
        """
        distance_km = distance_m / 1000
        
        # Oxygen/water vapor absorption (ITU-R P.676)
        # Approximation for 28 GHz
        specific_absorption_db_km = 0.05  # Typical for 28 GHz in clear sky
        
        # Rain attenuation (ITU-R P.838)
        # For 28 GHz: k ≈ 0.032, alpha ≈ 1.23
        k_rain = 0.032
        alpha_rain = 1.23
        rain_attenuation_db_km = k_rain * (rain_rate_mm_h ** alpha_rain)
        
        # Total attenuation
        total_attenuation_db = (specific_absorption_db_km + rain_attenuation_db_km) * distance_km
        
        return total_attenuation_db

    def calculate_doppler_shift(self, relative_velocity_m_s: float) -> float:
        """
        Calculate Doppler frequency shift due to satellite motion.
        
        Args:
            relative_velocity_m_s: Relative velocity between satellite and ground
            
        Returns:
            Doppler frequency shift in Hz
        """
        c = 3e8  # Speed of light
        carrier_freq_hz = self.config.frequency_ghz * 1e9
        
        # Doppler formula: f_d = (v/c) * f_c
        doppler_hz = (relative_velocity_m_s / c) * carrier_freq_hz
        
        return doppler_hz

    def generate_rician_channel(self, num_samples: int, 
                               k_factor: Optional[float] = None) -> np.ndarray:
        """
        Generate Rician fading channel matrix.
        
        Args:
            num_samples: Number of time samples
            k_factor: Rician K-factor (signal power / diffuse power)
            
        Returns:
            Complex channel matrix of shape (num_antennas_rx, num_antennas_tx, num_samples)
        """
        if k_factor is None:
            k_factor = self.config.rician_k_factor
        
        # Generate channel for all antenna combinations
        h_shape = (self.config.num_antennas_rx, self.config.num_antennas_tx, num_samples)
        
        if SIONNA_AVAILABLE:
            # Use Sionna's Rician generator if available
            # Placeholder - Sionna API would be used here
            pass
        
        # Fallback: Generate Rician channel manually
        # Line-of-sight (LoS) component
        los_power = k_factor / (1 + k_factor)
        los_component = np.sqrt(los_power) * np.ones(h_shape, dtype=complex)
        
        # Non-line-of-sight (NLoS) Rayleigh component
        nlos_power = 1 / (1 + k_factor)
        rayleigh_real = np.random.randn(*h_shape) * np.sqrt(nlos_power / 2)
        rayleigh_imag = np.random.randn(*h_shape) * np.sqrt(nlos_power / 2)
        nlos_component = rayleigh_real + 1j * rayleigh_imag
        
        # Combined Rician channel
        h_rician = los_component + nlos_component
        
        return h_rician

    def generate_delay_profile(self, num_taps: int = 25) -> np.ndarray:
        """
        Generate power delay profile (PDP) for channel.
        Models multipath propagation.
        
        Args:
            num_taps: Number of delay taps
            
        Returns:
            Power delay profile (normalized)
        """
        # Exponential decay profile (typical for outdoor NTN)
        rms_delay_spread_us = 0.5  # microseconds
        
        # Tap delays (linearly spaced)
        max_delay_us = 5 * rms_delay_spread_us
        tap_delays_us = np.linspace(0, max_delay_us, num_taps)
        
        # Exponential power decay
        power_profile = np.exp(-tap_delays_us / rms_delay_spread_us)
        
        # Normalize
        power_profile /= np.sum(power_profile)
        
        return power_profile

    def generate_channel_matrix(self, num_time_samples: int, 
                               path_loss_db: float,
                               doppler_hz: float,
                               rain_attenuation_db: float) -> np.ndarray:
        """
        Generate complete channel matrix with all impairments.
        
        Args:
            num_time_samples: Number of time samples
            path_loss_db: Free-space path loss
            doppler_hz: Doppler frequency shift
            rain_attenuation_db: Rain attenuation
            
        Returns:
            Channel matrix (num_antennas_rx, num_antennas_tx, num_time_samples)
        """
        # Generate fading component
        h_fading = self.generate_rician_channel(num_time_samples)
        
        # Apply path loss (linear scale)
        h_fading_scaled = h_fading * 10**(-path_loss_db / 20)
        
        # Apply rain attenuation
        h_fading_scaled *= 10**(-rain_attenuation_db / 20)
        
        # Apply Doppler (frequency shift in time domain)
        if doppler_hz != 0:
            t = np.arange(num_time_samples) / 1e6  # Time in seconds (1 MHz sampling)
            doppler_phase = 2 * np.pi * doppler_hz * t
            doppler_modulation = np.exp(1j * doppler_phase)
            h_fading_scaled *= doppler_modulation[np.newaxis, np.newaxis, :]
        
        return h_fading_scaled

    def generate_csi_for_scenario(self, scenario: Dict) -> Dict[str, np.ndarray]:
        """
        Generate CSI for all satellite-ground links in a scenario.
        
        Args:
            scenario: Scenario topology dictionary
            
        Returns:
            Dictionary with CSI for each satellite-ground link
        """
        csi_data = {}
        
        satellites = scenario.get('satellites', [])
        ground_stations = scenario.get('ground_stations', [])
        
        rain_rate = scenario.get('atmospheric_conditions', {}).get('rain_rate_mm_h', 0)
        
        # Store a representative, bounded link set so each scenario remains practical.
        max_links = self.config.max_links_per_scenario
        for sat in satellites:
            sat_id = sat['sat_id']
            sat_lat, sat_lon = sat['latitude'], sat['longitude']
            sat_velocity = sat.get('velocity_m_s', 0)
            
            for gs in ground_stations:
                if len(csi_data) >= max_links:
                    return csi_data

                gs_id = gs['gs_id']
                gs_lat, gs_lon = gs['latitude'], gs['longitude']
                
                # Calculate link geometry (simplified haversine distance)
                distance_m = self._great_circle_distance(
                    (sat_lat, sat_lon), (gs_lat, gs_lon), sat['altitude_km'] * 1000
                )
                
                # Calculate impairments
                path_loss_db = self.calculate_free_space_path_loss(distance_m)
                atmospheric_db = self.calculate_atmospheric_attenuation(distance_m, rain_rate)
                doppler_hz = self.calculate_doppler_shift(sat_velocity)
                
                # Generate channel matrix
                num_time_samples = 100  # 100 time samples
                h_channel = self.generate_channel_matrix(
                    num_time_samples, path_loss_db, doppler_hz, atmospheric_db
                )
                
                link_key = f"{sat_id}_{gs_id}"
                csi_data[link_key] = {
                    'channel_matrix': h_channel,
                    'path_loss_db': path_loss_db,
                    'doppler_hz': doppler_hz,
                    'atmospheric_attenuation_db': atmospheric_db,
                    'distance_m': distance_m,
                }
        
        return csi_data

    @staticmethod
    def _great_circle_distance(pos1: Tuple[float, float], pos2: Tuple[float, float], 
                              altitude_m: float = 0) -> float:
        """
        Calculate distance between two points considering altitude.
        Simplified spherical Earth model.
        """
        lat1, lon1 = np.radians(pos1)
        lat2, lon2 = np.radians(pos2)
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        
        earth_radius = 6371e3  # meters
        horizontal_dist = earth_radius * c
        
        # Include altitude difference
        distance = np.sqrt(horizontal_dist**2 + altitude_m**2)
        
        return distance


def save_csi_to_h5(csi_data: Dict[str, Dict], scenario_id: int, 
                   output_dir: Path) -> str:
    """
    Save channel state information to HDF5 file.
    
    Args:
        csi_data: Dictionary with CSI data
        scenario_id: Scenario identifier
        output_dir: Output directory
        
    Returns:
        Path to created HDF5 file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"scenario_{scenario_id:05d}_csi.h5"
    
    with h5py.File(output_file, 'w') as f:
        for link_key, csi_link_data in csi_data.items():
            group = f.create_group(link_key)
            
            # Store channel matrix
            group.create_dataset('channel_matrix', 
                                data=csi_link_data['channel_matrix'],
                                dtype=np.complex64, compression='gzip')
            
            # Store metadata
            group.attrs['path_loss_db'] = csi_link_data['path_loss_db']
            group.attrs['doppler_hz'] = csi_link_data['doppler_hz']
            group.attrs['atmospheric_attenuation_db'] = csi_link_data['atmospheric_attenuation_db']
            group.attrs['distance_m'] = csi_link_data['distance_m']
    
    logger.info(f"Saved CSI to {output_file}")
    return str(output_file)


if __name__ == '__main__':
    # Test: Generate CSI for a sample scenario
    config = ChannelConfig()
    simulator = ChannelSimulator(config)
    
    # Dummy scenario for testing
    test_scenario = {
        'satellites': [{'sat_id': 'LEO-0', 'latitude': 0, 'longitude': 0, 
                       'altitude_km': 550, 'velocity_m_s': 7600}],
        'ground_stations': [{'gs_id': 'GS-0', 'latitude': 0, 'longitude': 0}],
        'atmospheric_conditions': {'rain_rate_mm_h': 2.0}
    }
    
    csi_data = simulator.generate_csi_for_scenario(test_scenario)
    print(f"Generated CSI for {len(csi_data)} links")

"""
Step 1.4: Interference Modeling
Models interference in NTN scenarios including co-channel and adjacent-channel.

Creates:
- Co-channel interference (CCI) matrices
- Adjacent-channel interference (ACI)
- Inter-satellite interference (ISI)
- Terrestrial-satellite interference
- Interference power spectral density

Output: NPY files with interference matrices
"""

import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

logger = logging.getLogger(__name__)


class InterferenceModel:
    """Model interference in NTN scenarios."""

    # Interference parameters
    ISOLATION_ADJACENT_CHANNEL_DB = 45  # Typical filter isolation
    SAT_TO_SAT_COUPLING_DB = -90  # Free space coupling between satellites
    TERRESTRIAL_SAT_COUPLING_DB = -70  # Worse than sat-to-sat due to ground proximity

    def __init__(self, bandwidth_mhz: float = 400, frequency_ghz: float = 28,
                 num_subcarriers: int = 512):
        """
        Initialize interference model.
        
        Args:
            bandwidth_mhz: Bandwidth in MHz
            frequency_ghz: Carrier frequency in GHz
            num_subcarriers: Number of OFDM subcarriers
        """
        self.bandwidth_mhz = bandwidth_mhz
        self.frequency_ghz = frequency_ghz
        self.num_subcarriers = num_subcarriers
        self.subcarrier_spacing_khz = bandwidth_mhz * 1000 / num_subcarriers

    def calculate_co_channel_interference(self, interference_power_dbm: float,
                                        num_interferers: int = 5) -> float:
        """
        Calculate co-channel interference power.
        Multiple interferers add incoherently.
        
        Args:
            interference_power_dbm: Power from single interferer in dBm
            num_interferers: Number of co-channel interferers
            
        Returns:
            Total co-channel interference in dBm
        """
        # Incoherent addition: 10*log10(sum of linear powers)
        linear_power = 10**(interference_power_dbm / 10)
        total_power_dbm = 10 * np.log10(num_interferers * linear_power)
        
        return total_power_dbm

    def calculate_adjacent_channel_interference(self, transmit_power_dbm: float,
                                              num_adjacent_channels: int = 2) -> float:
        """
        Calculate adjacent-channel interference (ACI).
        Depends on filter isolation.
        
        Args:
            transmit_power_dbm: Transmit power in dBm
            num_adjacent_channels: Number of adjacent channels
            
        Returns:
            ACI power in dBm
        """
        # ACI = Tx Power - Filter Isolation
        aci_per_channel_dbm = transmit_power_dbm - self.ISOLATION_ADJACENT_CHANNEL_DB
        
        # Multiple adjacent channels add
        total_aci_dbm = 10 * np.log10(num_adjacent_channels) + aci_per_channel_dbm
        
        return total_aci_dbm

    def calculate_inter_satellite_interference(self, sat_tx_power_dbm: float,
                                             distance_m: float,
                                             frequency_ghz: Optional[float] = None) -> float:
        """
        Calculate interference from other satellites (Inter-Satellite Interference).
        
        Args:
            sat_tx_power_dbm: Satellite transmit power
            distance_m: Distance between satellites
            frequency_ghz: Frequency (uses class default if None)
            
        Returns:
            ISI power at receiver in dBm
        """
        if frequency_ghz is None:
            frequency_ghz = self.frequency_ghz
        
        # Free-space path loss between satellites
        wavelength_m = 3e8 / (frequency_ghz * 1e9)
        path_loss_db = 20 * np.log10(4 * np.pi * distance_m / wavelength_m)
        
        # Add coupling loss (satellites are far apart, not pointed at each other)
        coupling_db = self.SAT_TO_SAT_COUPLING_DB
        
        # ISI power at receiver
        isi_dbm = sat_tx_power_dbm - path_loss_db + coupling_db
        
        return isi_dbm

    def calculate_terrestrial_satellite_interference(self, terrestrial_tx_power_dbm: float,
                                                   distance_m: float) -> float:
        """
        Calculate interference from terrestrial systems to satellite.
        
        Args:
            terrestrial_tx_power_dbm: Terrestrial transmit power
            distance_m: Distance from terrestrial system to satellite
            
        Returns:
            Interference power in dBm
        """
        # Path loss
        wavelength_m = 3e8 / (self.frequency_ghz * 1e9)
        path_loss_db = 20 * np.log10(4 * np.pi * distance_m / wavelength_m)
        
        # Terrestrial uses different antennas (not pointing at satellite)
        coupling_db = self.TERRESTRIAL_SAT_COUPLING_DB
        
        # Interference at satellite
        interference_dbm = terrestrial_tx_power_dbm - path_loss_db + coupling_db
        
        return interference_dbm

    def generate_cci_matrix(self, num_subcarriers: int, 
                          num_channels: int = 16) -> np.ndarray:
        """
        Generate co-channel interference matrix across subcarriers/channels.
        
        Args:
            num_subcarriers: Number of subcarriers
            num_channels: Number of frequency channels
            
        Returns:
            CCI matrix of shape (num_channels, num_subcarriers)
        """
        # Generate random CCI per channel
        cci_matrix = np.zeros((num_channels, num_subcarriers))
        
        for ch_idx in range(num_channels):
            # Interference power (varies per channel)
            num_interferers = np.random.poisson(3)  # 3 interferers on average
            base_interference_dbm = -90
            interference_dbm = self.calculate_co_channel_interference(
                base_interference_dbm, num_interferers
            )
            
            # Spatially correlated: not all subcarriers equally affected
            for subcarrier_idx in range(num_subcarriers):
                # Distance from interfering channel center
                subcarrier_offset = np.abs(subcarrier_idx - num_subcarriers//2)
                
                # Selectivity: nearby subcarriers see more interference
                selectivity = np.exp(-subcarrier_offset / 50)
                
                cci_matrix[ch_idx, subcarrier_idx] = interference_dbm * selectivity
                
                # Add random variation
                cci_matrix[ch_idx, subcarrier_idx] += np.random.normal(0, 2)
        
        return cci_matrix

    def generate_aci_spectrum(self, num_subcarriers: int) -> np.ndarray:
        """
        Generate adjacent-channel interference spectrum.
        
        Args:
            num_subcarriers: Number of subcarriers
            
        Returns:
            ACI profile across subcarriers
        """
        aci_spectrum = np.zeros(num_subcarriers)
        
        # ACI from adjacent channels (left and right)
        tx_power_dbm = 20  # 100 mW typical
        aci_left_dbm = self.calculate_adjacent_channel_interference(tx_power_dbm, 1)
        aci_right_dbm = self.calculate_adjacent_channel_interference(tx_power_dbm, 1)
        
        # ACI rolloff: decreases away from channel edges
        center = num_subcarriers // 2
        
        for idx in range(num_subcarriers):
            distance_from_center = np.abs(idx - center)
            
            # Gaussian rolloff from channel edges
            if idx < center:
                rolloff = np.exp(-(distance_from_center - center) / 50)
                aci_spectrum[idx] = aci_left_dbm * rolloff
            else:
                rolloff = np.exp(-(distance_from_center - center) / 50)
                aci_spectrum[idx] = aci_right_dbm * rolloff
            
            # Add jitter
            aci_spectrum[idx] += np.random.normal(0, 1)
        
        return aci_spectrum

    def generate_multi_link_interference(self, scenario: Dict,
                                       num_time_samples: int = 100) -> np.ndarray:
        """
        Generate interference matrix for all active links.
        
        Args:
            scenario: Scenario topology
            num_time_samples: Number of time samples
            
        Returns:
            Interference matrix (num_frequencies, num_time_samples)
        """
        interference_matrix = np.zeros((self.num_subcarriers, num_time_samples))
        
        # Get all active links
        satellites = scenario.get('satellites', [])
        ground_stations = scenario.get('ground_stations', [])
        terrestrial_cells = scenario.get('terrestrial_cells', [])
        
        # Simulate interference from random active links
        num_active_sat_links = np.random.poisson(len(satellites) * 0.3)
        num_active_terrestrial = np.random.poisson(len(terrestrial_cells) * 0.2)
        
        # Satellite-satellite interference
        for _ in range(num_active_sat_links):
            sat_idx = np.random.randint(0, len(satellites))
            sat = satellites[sat_idx]
            
            # Interfering satellite
            other_sat_idx = np.random.randint(0, len(satellites))
            if other_sat_idx == sat_idx:
                continue
            
            other_sat = satellites[other_sat_idx]
            
            # Simplified distance (both at same altitude)
            distance_m = 500e3  # ~500 km typical inter-sat distance
            
            isi_power_dbm = self.calculate_inter_satellite_interference(
                20, distance_m
            )
            
            # Add to interference matrix
            affected_subcarriers = np.random.choice(
                self.num_subcarriers, 
                size=int(0.1 * self.num_subcarriers),
                replace=False
            )
            
            for subcarrier in affected_subcarriers:
                interference_matrix[subcarrier, :] += 10**(isi_power_dbm / 10)
        
        # Terrestrial-satellite interference
        for _ in range(num_active_terrestrial):
            terrestrial = terrestrial_cells[np.random.randint(0, len(terrestrial_cells))]
            
            # Random distance to satellite
            distance_m = np.random.uniform(100e3, 2000e3)
            
            interference_dbm = self.calculate_terrestrial_satellite_interference(
                30, distance_m  # 1W transmit power
            )
            
            # Add narrow-band interference (in-band)
            affected_subcarriers = np.random.choice(
                self.num_subcarriers,
                size=int(0.05 * self.num_subcarriers),
                replace=False
            )
            
            for subcarrier in affected_subcarriers:
                interference_matrix[subcarrier, :] += 10**(interference_dbm / 10)
        
        # Convert back to dB and add noise
        interference_matrix = 10 * np.log10(np.maximum(interference_matrix, 1e-12))
        interference_matrix += np.random.normal(0, 0.5, interference_matrix.shape)
        
        return interference_matrix


def save_interference_to_npy(interference_data: Dict[str, np.ndarray],
                            scenario_id: int, output_dir: Path) -> List[str]:
    """
    Save interference matrices to NPY files.
    
    Args:
        interference_data: Dictionary with interference arrays
        scenario_id: Scenario identifier
        output_dir: Output directory
        
    Returns:
        List of created file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    created_files = []
    
    for data_name, data_array in interference_data.items():
        output_file = output_dir / f"scenario_{scenario_id:05d}_{data_name}.npy"
        np.save(output_file, data_array)
        created_files.append(str(output_file))
    
    logger.info(f"Saved interference data to {output_dir}")
    return created_files


if __name__ == '__main__':
    # Test: Generate interference matrix
    model = InterferenceModel()
    
    test_scenario = {
        'satellites': [{'sat_id': f'LEO-{i}'} for i in range(12)],
        'ground_stations': [{'gs_id': f'GS-{i}'} for i in range(50)],
        'terrestrial_cells': [{'cell_id': f'CELL-{i}'} for i in range(30)],
    }
    
    interference_matrix = model.generate_multi_link_interference(test_scenario)
    print(f"Generated interference matrix of shape {interference_matrix.shape}")

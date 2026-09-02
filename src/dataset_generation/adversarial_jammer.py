"""
Step 1.5: Adversarial Jammer Modeling
Models intelligent adversarial jamming for robustness evaluation.

Creates:
- Jammer locations and power levels
- Jamming signal characteristics
- Time-varying jamming patterns
- Adaptive jamming (responds to system allocation)
- Jamming power spectral density

Output: NPY files with jamming signals and attack vectors
"""

import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class JammingType(Enum):
    """Types of jamming attacks."""
    BARRAGE = 'barrage'  # Broadband noise
    SWEPT = 'swept'  # Frequency-swept tone
    REACTIVE = 'reactive'  # Responds to transmission
    TARGETED = 'targeted'  # Focuses on specific user
    INTERMITTENT = 'intermittent'  # On-off pattern


class JammerModel:
    """Model adversarial jamming attacks in NTN scenarios."""

    def __init__(self, bandwidth_mhz: float = 400, frequency_ghz: float = 28,
                 num_subcarriers: int = 512, sampling_rate_hz: float = 1.0):
        """
        Initialize jammer model.
        
        Args:
            bandwidth_mhz: Signal bandwidth
            frequency_ghz: Carrier frequency
            num_subcarriers: Number of subcarriers
            sampling_rate_hz: Sampling rate
        """
        self.bandwidth_mhz = bandwidth_mhz
        self.frequency_ghz = frequency_ghz
        self.num_subcarriers = num_subcarriers
        self.sampling_rate_hz = sampling_rate_hz

    def place_jammers(self, scenario: Dict, num_jammers: int = 3,
                     jammer_power_dbm: float = 10) -> List[Dict]:
        """
        Place adversarial jammers in scenario.
        
        Args:
            scenario: Scenario topology
            num_jammers: Number of jammers to place
            jammer_power_dbm: Jammer transmit power
            
        Returns:
            List of jammer positions and characteristics
        """
        jammers = []
        
        # Randomly place jammers near coverage area
        coverage_radius_km = scenario.get('coverage_radius_km', 2000)
        
        for jammer_idx in range(num_jammers):
            # Random location within coverage area
            lat = np.random.uniform(-coverage_radius_km/111, coverage_radius_km/111)
            lon = np.random.uniform(-coverage_radius_km/111, coverage_radius_km/111)
            
            # Random jammer type
            jamming_type = np.random.choice(list(JammingType))
            
            jammers.append({
                'jammer_id': f'JAMMER-{jammer_idx}',
                'latitude': lat,
                'longitude': lon,
                'altitude_m': np.random.randint(100, 500),
                'transmit_power_dbm': jammer_power_dbm,
                'jamming_type': jamming_type.value,
                'antenna_gain_dbi': 10,
                'coverage_radius_km': np.random.uniform(10, 100),
            })
        
        return jammers

    def generate_barrage_jamming(self, num_samples: int, jammer_power_dbm: float,
                                jam_factor: float = 1.0) -> np.ndarray:
        """
        Generate barrage (broadband noise) jamming.
        Covers entire frequency band.
        
        Args:
            num_samples: Number of time samples
            jammer_power_dbm: Jammer power in dBm
            jam_factor: Jamming intensity factor (0-1)
            
        Returns:
            Jamming signal (num_subcarriers, num_samples)
        """
        # Gaussian noise across all subcarriers
        jammer_power_linear = 10**(jammer_power_dbm / 10) / 1000  # Convert to watts
        
        jamming_signal = np.random.randn(self.num_subcarriers, num_samples) * \
                        np.sqrt(jammer_power_linear * jam_factor)
        
        return jamming_signal

    def generate_swept_jamming(self, num_samples: int, jammer_power_dbm: float,
                              num_sweeps: int = 3, jam_factor: float = 1.0) -> np.ndarray:
        """
        Generate frequency-swept jamming.
        Jammer sweeps across frequency band.
        
        Args:
            num_samples: Number of time samples
            jammer_power_dbm: Jammer power in dBm
            num_sweeps: Number of sweeps across band
            jam_factor: Jamming intensity factor
            
        Returns:
            Jamming signal (num_subcarriers, num_samples)
        """
        jamming_signal = np.zeros((self.num_subcarriers, num_samples))
        jammer_power_linear = 10**(jammer_power_dbm / 10) / 1000
        
        # Generate sweeps
        samples_per_sweep = num_samples // num_sweeps
        
        for sweep_idx in range(num_sweeps):
            start_sample = sweep_idx * samples_per_sweep
            end_sample = (sweep_idx + 1) * samples_per_sweep
            
            # Random sweep direction
            if np.random.rand() > 0.5:
                # Sweep low to high
                sweep_subcarriers = np.arange(self.num_subcarriers)
            else:
                # Sweep high to low
                sweep_subcarriers = np.arange(self.num_subcarriers)[::-1]
            
            # Linear interpolation to map samples to subcarriers
            for sample_idx in range(start_sample, end_sample):
                position_in_sweep = (sample_idx - start_sample) / samples_per_sweep
                active_subcarrier = int(position_in_sweep * self.num_subcarriers)
                
                if active_subcarrier < self.num_subcarriers:
                    # Concentrated power on active subcarrier
                    width = 5  # Subcarrier width
                    for sc in range(max(0, active_subcarrier - width),
                                   min(self.num_subcarriers, active_subcarrier + width)):
                        jamming_signal[sc, sample_idx] += \
                            np.sqrt(jammer_power_linear * jam_factor)
        
        return jamming_signal

    def generate_reactive_jamming(self, legitimate_signal: np.ndarray,
                                 jammer_power_dbm: float,
                                 reaction_delay_samples: int = 5,
                                 jam_factor: float = 1.0) -> np.ndarray:
        """
        Generate reactive jamming that responds to legitimate signal.
        Jammer detects transmission and generates interference.
        
        Args:
            legitimate_signal: Original transmission signal
            jammer_power_dbm: Jammer power
            reaction_delay_samples: Samples delay for jammer to react
            jam_factor: Jamming intensity factor
            
        Returns:
            Jamming signal (num_subcarriers, num_samples)
        """
        jammer_power_linear = 10**(jammer_power_dbm / 10) / 1000
        num_samples = legitimate_signal.shape[1]
        
        jamming_signal = np.zeros_like(legitimate_signal, dtype=complex)
        
        # Detect active subcarriers in legitimate signal
        signal_power = np.abs(legitimate_signal) ** 2
        active_threshold = np.percentile(signal_power, 70)
        
        for sample_idx in range(num_samples):
            # Delay for reaction
            if sample_idx < reaction_delay_samples:
                continue
            
            # Get signal from previous samples
            prev_sample_idx = sample_idx - reaction_delay_samples
            prev_signal_power = signal_power[:, prev_sample_idx]
            
            # Active subcarriers
            active_subcarriers = np.where(prev_signal_power > active_threshold)[0]
            
            # Generate jamming on active subcarriers
            for sc in active_subcarriers:
                jamming_signal[sc, sample_idx] = \
                    np.sqrt(jammer_power_linear * jam_factor) * \
                    (1 + 1j) / np.sqrt(2)  # Random phase
        
        return jamming_signal

    def generate_targeted_jamming(self, num_samples: int, jammer_power_dbm: float,
                                 target_subcarrier: int = 256,
                                 jam_factor: float = 1.0) -> np.ndarray:
        """
        Generate targeted jamming focused on specific subcarriers.
        
        Args:
            num_samples: Number of time samples
            jammer_power_dbm: Jammer power
            target_subcarrier: Primary target subcarrier index
            jam_factor: Jamming intensity factor
            
        Returns:
            Jamming signal (num_subcarriers, num_samples)
        """
        jamming_signal = np.zeros((self.num_subcarriers, num_samples), dtype=complex)
        jammer_power_linear = 10**(jammer_power_dbm / 10) / 1000
        
        # Target subcarriers around primary target
        target_width = 10
        target_subcarriers = np.arange(
            max(0, target_subcarrier - target_width),
            min(self.num_subcarriers, target_subcarrier + target_width)
        )
        
        # Concentrated power on target subcarriers
        for sc in target_subcarriers:
            # Random phase coherent with jammer oscillator
            phase = 2 * np.pi * np.random.rand()
            
            # Power decreases away from center
            distance_from_center = np.abs(sc - target_subcarrier)
            power_factor = np.exp(-distance_from_center**2 / (target_width**2))
            
            jamming_signal[sc, :] = np.sqrt(jammer_power_linear * jam_factor * power_factor) * \
                                   np.exp(1j * phase)
        
        return jamming_signal

    def generate_intermittent_jamming(self, num_samples: int, jammer_power_dbm: float,
                                    on_probability: float = 0.5,
                                    jam_factor: float = 1.0) -> np.ndarray:
        """
        Generate intermittent (on-off) jamming.
        Jammer alternates between active and silent periods.
        
        Args:
            num_samples: Number of time samples
            jammer_power_dbm: Jammer power when active
            on_probability: Probability jammer is active at each time
            jam_factor: Jamming intensity factor
            
        Returns:
            Jamming signal (num_subcarriers, num_samples)
        """
        # Generate on-off pattern
        on_off_pattern = np.random.rand(num_samples) < on_probability
        
        # Generate broadband jamming
        jamming_signal = self.generate_barrage_jamming(num_samples, jammer_power_dbm, jam_factor)
        
        # Zero out inactive periods
        jamming_signal *= on_off_pattern[np.newaxis, :]
        
        return jamming_signal

    def generate_scenario_jamming(self, scenario: Dict,
                                num_jammers: int = 3,
                                num_time_samples: int = 100) -> Dict:
        """
        Generate jamming signals for all jammers in scenario.
        
        Args:
            scenario: Scenario topology
            num_jammers: Number of jammers
            num_time_samples: Number of time samples
            
        Returns:
            Dictionary with jammer positions and jamming signals
        """
        # Place jammers
        jammers = self.place_jammers(scenario, num_jammers)
        
        jamming_data = {
            'scenario_id': scenario.get('scenario_id', 0),
            'jammers': jammers,
            'jamming_signals': {}
        }
        
        # Generate jamming for each jammer
        for jammer in jammers:
            jammer_id = jammer['jammer_id']
            jamming_type = JammingType[jammer['jamming_type'].upper()]
            power_dbm = jammer['transmit_power_dbm']
            jam_factor = np.random.uniform(0.3, 1.0)  # Intensity variation
            
            # Generate appropriate jamming signal
            if jamming_type == JammingType.BARRAGE:
                jamming_sig = self.generate_barrage_jamming(num_time_samples, power_dbm, jam_factor)
            elif jamming_type == JammingType.SWEPT:
                jamming_sig = self.generate_swept_jamming(num_time_samples, power_dbm, jam_factor=jam_factor)
            elif jamming_type == JammingType.TARGETED:
                target_sc = np.random.randint(0, self.num_subcarriers)
                jamming_sig = self.generate_targeted_jamming(
                    num_time_samples, power_dbm, target_sc, jam_factor
                )
            elif jamming_type == JammingType.INTERMITTENT:
                jamming_sig = self.generate_intermittent_jamming(
                    num_time_samples, power_dbm, jam_factor=jam_factor
                )
            else:  # REACTIVE
                # Need legitimate signal for reactive jamming
                dummy_signal = np.random.randn(self.num_subcarriers, num_time_samples)
                jamming_sig = self.generate_reactive_jamming(
                    dummy_signal, power_dbm, jam_factor=jam_factor
                )
            
            jamming_data['jamming_signals'][jammer_id] = jamming_sig
        
        return jamming_data


def save_jamming_to_npy(jamming_data: Dict, scenario_id: int,
                       output_dir: Path) -> List[str]:
    """
    Save jamming data to NPY files.
    
    Args:
        jamming_data: Dictionary with jamming signals and metadata
        scenario_id: Scenario identifier
        output_dir: Output directory
        
    Returns:
        List of created file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    created_files = []
    
    # Save jamming signals
    for jammer_id, jamming_signal in jamming_data['jamming_signals'].items():
        output_file = output_dir / f"scenario_{scenario_id:05d}_{jammer_id}_jamming.npy"
        np.save(output_file, jamming_signal)
        created_files.append(str(output_file))
    
    logger.info(f"Saved jamming data for {len(jamming_data['jamming_signals'])} jammers")
    return created_files


if __name__ == '__main__':
    # Test: Generate jamming signals
    jammer = JammerModel()
    
    test_scenario = {
        'scenario_id': 0,
        'coverage_radius_km': 2000
    }
    
    jamming_data = jammer.generate_scenario_jamming(test_scenario, num_jammers=3)
    print(f"Generated jamming for {len(jamming_data['jammers'])} jammers")
    for jammer_id, sig in jamming_data['jamming_signals'].items():
        print(f"  {jammer_id}: {sig.shape}")

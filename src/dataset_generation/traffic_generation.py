"""
Step 1.3: Traffic Generation
Generates realistic user traffic patterns for NTN scenarios.

Creates:
- Downlink/uplink traffic patterns
- Bursty traffic (file transfers, video streaming)
- Real-time traffic (VoIP, video conferencing)
- Background traffic (IoT, telemetry)
- Quality of Service (QoS) requirements

Output: CSV files with traffic time series
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class TrafficType(Enum):
    """User traffic types."""
    REAL_TIME = 'real_time'  # VoIP, video conferencing
    BURSTY = 'bursty'  # File transfer, web browsing
    BACKGROUND = 'background'  # IoT, telemetry
    VIDEO_STREAMING = 'video_streaming'  # Netflix, YouTube


class QoSRequirement:
    """QoS requirement for traffic flow."""
    
    def __init__(self, traffic_type: TrafficType, min_bitrate_kbps: float = 0,
                 target_latency_ms: float = 100, max_loss_percent: float = 5):
        self.traffic_type = traffic_type
        self.min_bitrate_kbps = min_bitrate_kbps
        self.target_latency_ms = target_latency_ms
        self.max_loss_percent = max_loss_percent


class TrafficGenerator:
    """Generate realistic traffic patterns for NTN scenarios."""

    # Traffic model parameters
    REAL_TIME_BITRATE_KBPS = 64  # VoIP typical
    VIDEO_STREAMING_BITRATE_KBPS = 2500  # Average HD video
    FILE_TRANSFER_SIZE_MB = 100  # Typical file size
    
    def __init__(self, num_users: int = 1000, duration_seconds: float = 1.0, 
                 sampling_rate_hz: float = 1.0):
        """
        Initialize traffic generator.
        
        Args:
            num_users: Number of users/UEs
            duration_seconds: Simulation duration
            sampling_rate_hz: Traffic sampling rate
        """
        self.num_users = num_users
        self.duration_seconds = duration_seconds
        self.sampling_rate_hz = sampling_rate_hz
        self.num_samples = int(duration_seconds * sampling_rate_hz)
        self.time_vector = np.linspace(0, duration_seconds, self.num_samples)

    def generate_real_time_traffic(self) -> Tuple[np.ndarray, QoSRequirement]:
        """
        Generate real-time traffic (VoIP, video conferencing).
        Constant bit rate with periodic activity.
        
        Returns:
            (traffic vector in Kbps, QoS requirement)
        """
        # VoIP: 64 kbps, active 50% of time
        activity_pattern = np.random.poisson(0.5, self.num_samples).astype(float)
        traffic = activity_pattern * self.REAL_TIME_BITRATE_KBPS
        
        # Add small noise
        traffic += np.random.normal(0, 5, self.num_samples)
        traffic = np.maximum(traffic, 0)
        
        qos = QoSRequirement(
            TrafficType.REAL_TIME,
            min_bitrate_kbps=self.REAL_TIME_BITRATE_KBPS * 0.8,
            target_latency_ms=150,
            max_loss_percent=1
        )
        
        return traffic, qos

    def generate_video_streaming_traffic(self) -> Tuple[np.ndarray, QoSRequirement]:
        """
        Generate video streaming traffic (Netflix, YouTube).
        Variable bit rate with adaptive bitrate algorithm.
        
        Returns:
            (traffic vector in Kbps, QoS requirement)
        """
        # Base video bitrate with fluctuations
        traffic = np.full(
            self.num_samples,
            self.VIDEO_STREAMING_BITRATE_KBPS,
            dtype=float,
        )
        
        # Simulate adaptive bitrate (quality changes)
        # Random quality changes every ~10 seconds
        num_segments = int(self.duration_seconds / 10)
        for _ in range(num_segments):
            seg_start = np.random.randint(0, self.num_samples - 50)
            seg_end = min(seg_start + 50, self.num_samples)
            
            # Quality shift: ±20%
            quality_factor = np.random.uniform(0.8, 1.2)
            traffic[seg_start:seg_end] *= quality_factor
        
        # Add periodic buffering (stalls)
        num_stalls = int(self.duration_seconds * 0.5)
        for _ in range(num_stalls):
            stall_pos = np.random.randint(0, self.num_samples - 5)
            traffic[stall_pos:stall_pos+5] = 0  # Buffering gap
        
        qos = QoSRequirement(
            TrafficType.VIDEO_STREAMING,
            min_bitrate_kbps=self.VIDEO_STREAMING_BITRATE_KBPS * 0.5,
            target_latency_ms=500,
            max_loss_percent=2
        )
        
        return traffic, qos

    def generate_bursty_traffic(self) -> Tuple[np.ndarray, QoSRequirement]:
        """
        Generate bursty traffic (file downloads, web browsing).
        Poisson-modulated traffic with idle periods.
        
        Returns:
            (traffic vector in Kbps, QoS requirement)
        """
        # Poisson burst arrivals
        burst_rate = 2  # bursts per second
        traffic = np.zeros(self.num_samples)
        
        # Generate burst times
        num_bursts = np.random.poisson(burst_rate * self.duration_seconds)
        burst_times = np.random.uniform(0, self.duration_seconds, num_bursts)
        
        for burst_time in burst_times:
            # Burst properties
            burst_duration_s = np.random.exponential(0.1)  # 100 ms average burst
            burst_bitrate_kbps = np.random.uniform(500, 5000)  # 500-5000 Kbps
            burst_intensity = np.random.uniform(0.5, 1.0)  # Intensity modulation
            
            # Apply burst to traffic vector
            start_idx = int(burst_time * self.sampling_rate_hz)
            end_idx = int((burst_time + burst_duration_s) * self.sampling_rate_hz)
            end_idx = min(end_idx, self.num_samples)
            
            traffic[start_idx:end_idx] = burst_bitrate_kbps * burst_intensity
        
        qos = QoSRequirement(
            TrafficType.BURSTY,
            min_bitrate_kbps=0,  # No minimum
            target_latency_ms=1000,
            max_loss_percent=5
        )
        
        return traffic, qos

    def generate_background_traffic(self) -> Tuple[np.ndarray, QoSRequirement]:
        """
        Generate background/IoT traffic.
        Low-rate periodic reporting.
        
        Returns:
            (traffic vector in Kbps, QoS requirement)
        """
        # Periodic small packets (e.g., sensor data every 10 seconds)
        report_interval_s = 10
        packet_size_bits = 256  # Small IoT packet
        
        traffic = np.zeros(self.num_samples)
        
        # Reporting times
        report_times = np.arange(0, self.duration_seconds, report_interval_s)
        
        for report_time in report_times:
            idx = int(report_time * self.sampling_rate_hz)
            if idx < self.num_samples:
                # Packet transmission time (~1 ms)
                packet_duration = 0.001 * self.sampling_rate_hz
                traffic[idx:int(idx+packet_duration)] = packet_size_bits / 0.001 / 1000  # Kbps
        
        # Add some random jitter
        traffic += np.random.exponential(0.1, self.num_samples)
        
        qos = QoSRequirement(
            TrafficType.BACKGROUND,
            min_bitrate_kbps=0,
            target_latency_ms=5000,  # Relaxed latency
            max_loss_percent=10  # Higher loss tolerance
        )
        
        return traffic, qos

    def generate_user_traffic_mix(self) -> List[Dict]:
        """
        Generate traffic for all users with mixed traffic types.
        
        Returns:
            List of user traffic dictionaries
        """
        user_traffic_list = []
        
        for user_id in range(self.num_users):
            # Randomly assign traffic type to user
            traffic_type_rand = np.random.rand()
            
            if traffic_type_rand < 0.3:
                traffic, qos = self.generate_real_time_traffic()
            elif traffic_type_rand < 0.5:
                traffic, qos = self.generate_video_streaming_traffic()
            elif traffic_type_rand < 0.7:
                traffic, qos = self.generate_bursty_traffic()
            else:
                traffic, qos = self.generate_background_traffic()
            
            # Separate uplink/downlink
            # Assume 20% uplink, 80% downlink on average
            uplink_ratio = np.random.uniform(0.15, 0.25)
            
            user_traffic = {
                'ue_id': f'UE-{user_id}',
                'traffic_type': qos.traffic_type.value,
                'downlink_traffic_kbps': traffic * (1 - uplink_ratio),
                'uplink_traffic_kbps': traffic * uplink_ratio,
                'total_traffic_kbps': traffic,
                'time_vector_s': self.time_vector,
                'qos_min_bitrate_kbps': qos.min_bitrate_kbps,
                'qos_target_latency_ms': qos.target_latency_ms,
                'qos_max_loss_percent': qos.max_loss_percent,
            }
            
            user_traffic_list.append(user_traffic)
        
        return user_traffic_list

    def generate_scenario_traffic(self, scenario: Dict) -> Dict:
        """
        Generate traffic for all UEs in a scenario.
        
        Args:
            scenario: Scenario topology
            
        Returns:
            Dictionary with traffic for all UEs
        """
        num_ues = len(scenario.get('user_equipment', []))
        
        # Update to match scenario UE count
        original_num_users = self.num_users
        self.num_users = num_ues
        
        traffic_data = self.generate_user_traffic_mix()
        
        # Restore
        self.num_users = original_num_users
        
        return {
            'scenario_id': scenario.get('scenario_id', 0),
            'duration_seconds': self.duration_seconds,
            'sampling_rate_hz': self.sampling_rate_hz,
            'user_traffic': traffic_data,
            'total_network_demand_kbps': sum([t['total_traffic_kbps'].sum() 
                                             for t in traffic_data]) / len(traffic_data),
        }


def save_traffic_to_csv(scenario_traffic: Dict, scenario_id: int, 
                       output_dir: Path) -> str:
    """
    Save traffic data to CSV file.
    
    Args:
        scenario_traffic: Traffic dictionary from generator
        scenario_id: Scenario identifier
        output_dir: Output directory
        
    Returns:
        Path to created CSV file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"scenario_{scenario_id:05d}_traffic.csv"
    
    # Flatten user traffic data into a single dataframe
    data_rows = []
    
    for user_traffic in scenario_traffic['user_traffic']:
        for t_idx, t_val in enumerate(user_traffic['time_vector_s']):
            row = {
                'scenario_id': scenario_id,
                'ue_id': user_traffic['ue_id'],
                'time_s': t_val,
                'traffic_type': user_traffic['traffic_type'],
                'downlink_kbps': user_traffic['downlink_traffic_kbps'][t_idx],
                'uplink_kbps': user_traffic['uplink_traffic_kbps'][t_idx],
                'total_kbps': user_traffic['total_traffic_kbps'][t_idx],
                'qos_latency_target_ms': user_traffic['qos_target_latency_ms'],
                'qos_max_loss_percent': user_traffic['qos_max_loss_percent'],
            }
            data_rows.append(row)
    
    df = pd.DataFrame(data_rows)
    df.to_csv(output_file, index=False)
    
    logger.info(f"Saved traffic to {output_file}")
    return str(output_file)


if __name__ == '__main__':
    # Test: Generate traffic for a sample scenario
    generator = TrafficGenerator(num_users=100, duration_seconds=10, sampling_rate_hz=1.0)
    
    test_scenario = {
        'scenario_id': 0,
        'user_equipment': [{'ue_id': f'UE-{i}'} for i in range(100)]
    }
    
    traffic_data = generator.generate_scenario_traffic(test_scenario)
    print(f"Generated traffic for {len(traffic_data['user_traffic'])} UEs")
    print(f"Average network demand: {traffic_data['total_network_demand_kbps']:.2f} Kbps")

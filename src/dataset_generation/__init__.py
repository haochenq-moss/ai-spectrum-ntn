"""
Dataset Generation Module for AI Spectrum Management NTN

This module orchestrates the generation of synthetic NTN wireless datasets
with realistic channel conditions, traffic patterns, interference, and
adversarial jamming for training spectrum allocation AI agents.

Components:
- scenario_topology: Generates NTN topology (satellites, ground stations, UEs)
- channel_generation: Simulates wireless channel state information
- traffic_generation: Creates realistic user traffic patterns
- interference_modeling: Models co-channel and adjacent-channel interference
- adversarial_jammer: Generates adversarial jamming attacks
- dataset_assembler: Orchestrates complete dataset generation
"""

from .scenario_topology import NTNTopologyGenerator, generate_scenario_json
from .channel_generation import ChannelSimulator, ChannelConfig, save_csi_to_h5
from .traffic_generation import TrafficGenerator, TrafficType, QoSRequirement, save_traffic_to_csv
from .interference_modeling import InterferenceModel, save_interference_to_npy
from .adversarial_jammer import JammerModel, JammingType, save_jamming_to_npy
from .dataset_assembler import DatasetAssembler, generate_dataset

__all__ = [
    'NTNTopologyGenerator',
    'generate_scenario_json',
    'ChannelSimulator',
    'ChannelConfig',
    'save_csi_to_h5',
    'TrafficGenerator',
    'TrafficType',
    'QoSRequirement',
    'save_traffic_to_csv',
    'InterferenceModel',
    'save_interference_to_npy',
    'JammerModel',
    'JammingType',
    'save_jamming_to_npy',
    'DatasetAssembler',
    'generate_dataset',
]

__version__ = '0.1.0'

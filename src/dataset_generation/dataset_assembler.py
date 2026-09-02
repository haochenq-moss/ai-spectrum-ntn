"""
Step 1.6: Dataset Assembler
Orchestrates all dataset generation components and creates final dataset files.

Combines:
- Scenario topologies (JSON)
- Channel state information (HDF5)
- Traffic patterns (CSV)
- Interference matrices (NPY)
- Jamming signals (NPY)

Output: Complete dataset ready for Step 2 (Data Processing)
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from .scenario_topology import NTNTopologyGenerator, generate_scenario_json
from .channel_generation import ChannelSimulator, ChannelConfig, save_csi_to_h5
from .traffic_generation import TrafficGenerator, save_traffic_to_csv
from .interference_modeling import InterferenceModel, save_interference_to_npy
from .adversarial_jammer import JammerModel, save_jamming_to_npy

logger = logging.getLogger(__name__)


class DatasetAssembler:
    """Orchestrate complete dataset generation for NTN spectrum management."""

    def __init__(self, output_base_dir: Path = Path('data/raw'),
                 num_scenarios: int = 100,
                 enable_channel_sim: bool = True,
                 enable_interference: bool = True,
                 enable_jamming: bool = True):
        """
        Initialize dataset assembler.
        
        Args:
            output_base_dir: Base directory for output files
            num_scenarios: Number of scenarios to generate
            enable_channel_sim: Whether to generate channel data
            enable_interference: Whether to generate interference
            enable_jamming: Whether to generate jamming signals
        """
        self.output_base_dir = Path(output_base_dir)
        self.num_scenarios = num_scenarios
        self.enable_channel_sim = enable_channel_sim
        self.enable_interference = enable_interference
        self.enable_jamming = enable_jamming
        
        # Create output directories
        self.scenario_dir = self.output_base_dir / 'scenarios'
        self.csi_dir = self.output_base_dir / 'csi'
        self.traffic_dir = self.output_base_dir / 'traffic'
        self.interference_dir = self.output_base_dir / 'interference'
        self.jamming_dir = self.output_base_dir / 'jamming'
        
        for dir_path in [self.scenario_dir, self.csi_dir, self.traffic_dir,
                        self.interference_dir, self.jamming_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def generate_single_scenario(self, scenario_id: int) -> Dict[str, List[str]]:
        """
        Generate complete dataset for a single scenario.
        
        Args:
            scenario_id: Scenario identifier
            
        Returns:
            Dictionary with paths to all generated files
        """
        generated_files = {'scenario_id': scenario_id, 'files': {}}
        
        try:
            # Step 1.1: Generate scenario topology
            logger.info(f"[Scenario {scenario_id}] Generating topology...")
            topo_gen = NTNTopologyGenerator()
            scenario = topo_gen.assemble_scenario(scenario_id)
            
            # Save scenario JSON
            scenario_file = self.scenario_dir / f"scenario_{scenario_id:05d}.json"
            with open(scenario_file, 'w') as f:
                json.dump(scenario, f, indent=2)
            generated_files['files']['scenario_json'] = str(scenario_file)
            
            # Step 1.2: Generate channel state information
            if self.enable_channel_sim:
                logger.info(f"[Scenario {scenario_id}] Generating CSI...")
                csi_config = ChannelConfig()
                csi_sim = ChannelSimulator(csi_config)
                csi_data = csi_sim.generate_csi_for_scenario(scenario)
                
                csi_file = save_csi_to_h5(csi_data, scenario_id, self.csi_dir)
                generated_files['files']['csi_h5'] = csi_file
            
            # Step 1.3: Generate traffic patterns
            logger.info(f"[Scenario {scenario_id}] Generating traffic...")
            num_ues = len(scenario.get('user_equipment', []))
            traffic_gen = TrafficGenerator(num_users=num_ues, duration_seconds=1.0,
                                          sampling_rate_hz=1.0)
            scenario_traffic = traffic_gen.generate_scenario_traffic(scenario)
            
            traffic_file = save_traffic_to_csv(scenario_traffic, scenario_id, self.traffic_dir)
            generated_files['files']['traffic_csv'] = traffic_file
            
            # Step 1.4: Generate interference matrix
            if self.enable_interference:
                logger.info(f"[Scenario {scenario_id}] Generating interference...")
                intf_model = InterferenceModel()
                intf_matrix = intf_model.generate_multi_link_interference(scenario)
                
                intf_file = self.interference_dir / f"scenario_{scenario_id:05d}_interference.npy"
                np.save(intf_file, intf_matrix)
                generated_files['files']['interference_npy'] = str(intf_file)
            
            # Step 1.5: Generate adversarial jamming
            if self.enable_jamming:
                logger.info(f"[Scenario {scenario_id}] Generating jamming...")
                jammer_model = JammerModel()
                jamming_data = jammer_model.generate_scenario_jamming(scenario, num_jammers=3)
                
                jamming_files = save_jamming_to_npy(jamming_data, scenario_id, self.jamming_dir)
                generated_files['files']['jamming_npy'] = jamming_files
            
            logger.info(f"[Scenario {scenario_id}] ✓ Complete")
            
        except Exception as e:
            logger.error(f"[Scenario {scenario_id}] Error: {e}")
            generated_files['error'] = str(e)
        
        return generated_files

    def generate_full_dataset(self) -> Dict:
        """
        Generate complete dataset with all scenarios.
        
        Returns:
            Summary statistics and file manifest
        """
        logger.info(f"Starting dataset generation for {self.num_scenarios} scenarios...")
        
        start_time = time.time()
        all_manifests = []
        
        for scenario_id in range(self.num_scenarios):
            manifest = self.generate_single_scenario(scenario_id)
            all_manifests.append(manifest)
            
            # Progress indicator
            if (scenario_id + 1) % max(1, self.num_scenarios // 10) == 0:
                elapsed = time.time() - start_time
                rate = (scenario_id + 1) / elapsed
                eta_seconds = (self.num_scenarios - scenario_id - 1) / rate
                logger.info(f"  Progress: {scenario_id + 1}/{self.num_scenarios} "
                           f"({rate:.1f} scenarios/sec, ETA: {eta_seconds:.0f}s)")
        
        # Compute statistics
        total_time = time.time() - start_time
        scenarios_per_hour = (self.num_scenarios / total_time) * 3600
        
        # Count generated files
        total_files = 0
        file_types = {}
        for manifest in all_manifests:
            if 'error' not in manifest:
                for file_type, file_path in manifest['files'].items():
                    total_files += 1
                    file_types[file_type] = file_types.get(file_type, 0) + 1
        
        summary = {
            'total_scenarios': self.num_scenarios,
            'generation_time_seconds': total_time,
            'scenarios_per_hour': scenarios_per_hour,
            'total_files_generated': total_files,
            'file_types': file_types,
            'output_base_dir': str(self.output_base_dir),
            'manifests': all_manifests,
        }
        
        logger.info(f"\n{'='*60}")
        logger.info("DATASET GENERATION COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"Total Scenarios: {self.num_scenarios}")
        logger.info(f"Generation Time: {total_time:.1f} seconds")
        logger.info(f"Rate: {scenarios_per_hour:.0f} scenarios/hour")
        logger.info(f"Total Files: {total_files}")
        for file_type, count in file_types.items():
            logger.info(f"  - {file_type}: {count}")
        logger.info(f"Output Directory: {self.output_base_dir}")
        logger.info(f"{'='*60}\n")
        
        return summary

    def save_manifest(self, summary: Dict, filename: str = 'dataset_manifest.json'):
        """
        Save dataset manifest to JSON file.
        
        Args:
            summary: Summary dictionary from generate_full_dataset
            filename: Output filename
        """
        manifest_file = self.output_base_dir / filename
        
        with open(manifest_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Saved manifest to {manifest_file}")


def generate_dataset(num_scenarios: int = 100, output_dir: str = 'data/raw',
                    enable_channel_sim: bool = True,
                    enable_interference: bool = True,
                    enable_jamming: bool = True) -> Dict:
    """
    Main entry point for dataset generation.
    
    Args:
        num_scenarios: Number of scenarios to generate
        output_dir: Output directory path
        enable_channel_sim: Enable channel simulation
        enable_interference: Enable interference modeling
        enable_jamming: Enable jamming generation
        
    Returns:
        Summary statistics dictionary
    """
    assembler = DatasetAssembler(
        output_base_dir=Path(output_dir),
        num_scenarios=num_scenarios,
        enable_channel_sim=enable_channel_sim,
        enable_interference=enable_interference,
        enable_jamming=enable_jamming
    )
    
    summary = assembler.generate_full_dataset()
    assembler.save_manifest(summary)
    
    return summary


if __name__ == '__main__':
    # Test: Generate small dataset
    logging.basicConfig(level=logging.INFO)
    
    summary = generate_dataset(
        num_scenarios=5,
        output_dir=Path('/tmp/test_dataset'),
        enable_channel_sim=True,
        enable_interference=True,
        enable_jamming=True
    )
    
    print("\n" + "="*60)
    print("Dataset generation test complete!")
    print("="*60)

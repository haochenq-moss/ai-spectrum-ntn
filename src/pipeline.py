"""
Main orchestrator for the 10-step AI Spectrum Management pipeline.
Coordinates dataset generation → processing → optimization → training → agent → evaluation.
"""

from typing import Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class SpectrumManagementPipeline:
    """
    Orchestrates the full 10-step pipeline:
    1. Dataset Generation
    2. Data Processing
    3. Ground-Truth Optimization
    4. Neural Model Training
    5. AI Agent Construction
    6. Tool Layer (wireless functions)
    7. NemoIR Workflow Compilation
    8. Evaluation & Benchmarking
    9. Online Execution
    10. AutoResearch Loop
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize pipeline with configuration."""
        self.config = self.load_config(config_path) if config_path else {}
        self.results = {}
        
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML or JSON."""
        import yaml
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def step_1_generate_dataset(self, num_scenarios: int = 100, output_dir: str = 'data/raw/'):
        """Step 1: Generate raw NTN wireless dataset."""
        logger.info(f"Step 1️⃣  Generating {num_scenarios} NTN scenarios...")
        
        from src.dataset_generation.dataset_assembler import generate_dataset
        dataset = generate_dataset(
            num_scenarios=num_scenarios,
            output_dir=output_dir,
            scenario_config=self.config.get('dataset', {})
        )
        
        self.results['dataset'] = dataset
        logger.info(f"✅ Generated {len(dataset)} scenarios")
        return dataset
    
    def step_2_process_data(self, input_dir: str = 'data/raw/', output_dir: str = 'data/processed/'):
        """Step 2: Process raw data to learning states."""
        logger.info("Step 2️⃣  Processing raw data to learning states...")
        
        from src.data_processing.preprocessing_pipeline import process_all
        processed_data = process_all(
            input_dir=input_dir,
            output_dir=output_dir,
            feature_config=self.config.get('features', {})
        )
        
        self.results['processed_data'] = processed_data
        logger.info(f"✅ Processed {len(processed_data)} learning states")
        return processed_data
    
    def step_3_ground_truth_optimization(self, method: str = 'adversarial_wf', 
                                        num_scenarios: int = 100,
                                        output_dir: str = 'data/ground_truth/'):
        """Step 3: Generate optimal labels via ground-truth optimization."""
        logger.info(f"Step 3️⃣  Generating ground-truth labels using {method}...")
        
        from src.optimization.teacher_dataset import generate_teacher_dataset
        labels = generate_teacher_dataset(
            method=method,
            num_scenarios=num_scenarios,
            output_dir=output_dir,
            optimization_config=self.config.get('optimization', {})
        )
        
        self.results['ground_truth_labels'] = labels
        logger.info(f"✅ Generated {len(labels)} optimal actions")
        return labels
    
    def step_4_train_models(self, models: Optional[list] = None, epochs: int = 100, gpu_id: int = 0):
        """Step 4: Train neural models (supervised, RL, predictors)."""
        if models is None:
            models = ['supervised', 'channel_predictor', 'jammer_detector']
        
        logger.info(f"Step 4️⃣  Training models: {models}")
        
        trained_models = {}
        
        for model_name in models:
            logger.info(f"  → Training {model_name}...")
            
            if model_name == 'supervised':
                from src.models.supervised_learning import train_supervised
                model = train_supervised(
                    epochs=epochs,
                    device=f'cuda:{gpu_id}',
                    training_config=self.config.get('training', {})
                )
            elif model_name == 'rl':
                from src.models.reinforcement_learning import train_rl
                model = train_rl(
                    num_episodes=epochs,
                    device=f'cuda:{gpu_id}',
                    training_config=self.config.get('training', {})
                )
            elif model_name == 'channel_predictor':
                from src.models.channel_predictor import train_predictor
                model = train_predictor(
                    epochs=epochs,
                    device=f'cuda:{gpu_id}',
                    training_config=self.config.get('training', {})
                )
            elif model_name == 'jammer_detector':
                from src.models.jammer_detector import train_detector
                model = train_detector(
                    epochs=epochs,
                    device=f'cuda:{gpu_id}',
                    training_config=self.config.get('training', {})
                )
            else:
                raise ValueError(f"Unknown model: {model_name}")
            
            trained_models[model_name] = model
            logger.info(f"  ✅ {model_name} trained")
        
        self.results['trained_models'] = trained_models
        return trained_models
    
    def step_5_build_agent(self, agent_type: str = 'llm', memory_size: int = 1000):
        """Step 5: Build AI agent with reasoning and planning."""
        logger.info(f"Step 5️⃣  Building {agent_type} agent with RAG memory...")
        
        from src.agent.ai_agent import AIAgent
        agent = AIAgent(
            agent_type=agent_type,
            memory_size=memory_size,
            trained_models=self.results.get('trained_models', {}),
            agent_config=self.config.get('agent', {})
        )
        
        self.results['agent'] = agent
        logger.info("✅ Agent built and ready")
        return agent
    
    def step_7_compile_workflow(self, workflow_file: str = 'src/workflows/agent_workflow.nemo',
                               output_file: str = 'compiled_agent.ir',
                               optimize: str = 'aggressive'):
        """Step 7: Compile NemoIR workflow."""
        logger.info(f"Step 7️⃣  Compiling NemoIR workflow ({optimize} optimization)...")
        
        from src.workflows.compile_and_run import compile_workflow
        compile_workflow(workflow_file, output_file, optimize)
        
        logger.info(f"✅ Workflow compiled to {output_file}")
        self.results['compiled_workflow'] = output_file
        return output_file
    
    def step_8_benchmark(self, baselines: Optional[list] = None, num_scenarios: int = 100,
                        output_dir: str = 'experiments/results/'):
        """Step 8: Run comprehensive benchmarking."""
        if baselines is None:
            baselines = ['classical_wf', 'adversarial_wf', 'diffract', 'rl', 'llm', 'proposed']
        
        logger.info(f"Step 8️⃣  Benchmarking {baselines}...")
        
        from src.evaluation.benchmarks import run_benchmark
        results = run_benchmark(
            baselines=baselines,
            num_scenarios=num_scenarios,
            output_dir=output_dir,
            agent=self.results.get('agent'),
            compiled_workflow=self.results.get('compiled_workflow')
        )
        
        self.results['benchmark_results'] = results
        logger.info("✅ Benchmarking complete")
        return results
    
    def step_9_online_execution(self, duration_seconds: int = 3600, 
                               agent_type: str = 'compiled',
                               log_file: str = 'logs/online_execution.log'):
        """Step 9: Run agent for real-time online spectrum allocation."""
        logger.info(f"Step 9️⃣  Running online execution for {duration_seconds}s...")
        
        from src.evaluation.online_executor import OnlineExecutor
        executor = OnlineExecutor(
            agent_type=agent_type,
            agent=self.results.get('agent'),
            compiled_workflow=self.results.get('compiled_workflow'),
            log_file=log_file
        )
        
        execution_results = executor.run(duration_seconds=duration_seconds)
        self.results['online_execution'] = execution_results
        logger.info("✅ Online execution complete")
        return execution_results
    
    def step_10_launch_autoresearch(self, duration_seconds: int = 28800, gpu_id: int = 0):
        """Step 10: Launch autonomous research loop."""
        logger.info(f"Step 🔟  Launching AutoResearch loop for {duration_seconds}s...")
        
        from research.experiment_runner import launch_autoresearch_loop
        research_results = launch_autoresearch_loop(
            duration_seconds=duration_seconds,
            gpu_id=gpu_id,
            program_file='research/program.md',
            results_file='research/results.tsv',
            learnings_file='research/learnings.md'
        )
        
        self.results['autoresearch'] = research_results
        logger.info("✅ AutoResearch loop complete")
        return research_results
    
    def run_full_pipeline(self, stages: Optional[list] = None):
        """Run the full 10-step pipeline."""
        if stages is None:
            stages = list(range(1, 11))
        
        logger.info("🚀 Starting AI Spectrum Management Pipeline")
        logger.info(f"   Stages to run: {stages}")
        
        try:
            if 1 in stages:
                self.step_1_generate_dataset(
                    num_scenarios=self.config.get('dataset', {}).get('num_scenarios', 100)
                )
            
            if 2 in stages:
                self.step_2_process_data()
            
            if 3 in stages:
                self.step_3_ground_truth_optimization(
                    method=self.config.get('optimization', {}).get('method', 'adversarial_wf')
                )
            
            if 4 in stages:
                self.step_4_train_models(epochs=self.config.get('training', {}).get('epochs', 100))
            
            if 5 in stages:
                self.step_5_build_agent(
                    agent_type=self.config.get('agent', {}).get('type', 'llm')
                )
            
            if 7 in stages:
                self.step_7_compile_workflow()
            
            if 8 in stages:
                self.step_8_benchmark(
                    num_scenarios=self.config.get('evaluation', {}).get('num_scenarios', 100)
                )
            
            if 9 in stages:
                self.step_9_online_execution(
                    duration_seconds=self.config.get('online', {}).get('duration', 3600)
                )
            
            if 10 in stages:
                self.step_10_launch_autoresearch(
                    duration_seconds=self.config.get('autoresearch', {}).get('duration', 28800)
                )
            
            logger.info("✅ Pipeline complete!")
            return self.results
        
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}", exc_info=True)
            raise


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='AI Spectrum Management Pipeline')
    parser.add_argument('--config', help='Path to configuration YAML')
    parser.add_argument('--stages', nargs='+', type=int, help='Stages to run')
    parser.add_argument('--log-level', default='INFO', help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run pipeline
    pipeline = SpectrumManagementPipeline(config_path=args.config)
    results = pipeline.run_full_pipeline(stages=args.stages)
    
    print("\n📊 Pipeline Results:")
    for stage, result in results.items():
        print(f"  {stage}: ✅")


if __name__ == '__main__':
    main()

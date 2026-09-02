"""
AI Spectrum Management for NTN - Unified CLI
Integrating RAG-Practice, AutoResearch, and NemoIR frameworks
"""

import click
import sys
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))


@click.group()
def cli():
    """AI Agentic Spectrum Management CLI"""
    pass


@cli.command()
@click.option('--num-scenarios', default=100, help='Number of scenarios to generate')
@click.option('--output', default='data/raw/', help='Output directory')
@click.option('--enable-channel', is_flag=True, default=True, help='Generate channel data')
@click.option('--enable-interference', is_flag=True, default=True, help='Generate interference')
@click.option('--enable-jamming', is_flag=True, default=True, help='Generate jamming signals')
def generate_dataset(num_scenarios: int, output: str, enable_channel: bool,
                    enable_interference: bool, enable_jamming: bool):
    """Generate raw NTN wireless dataset (Step 1)"""
    click.echo(f"📊 Generating {num_scenarios} NTN scenarios...")
    click.echo(f"  Channel simulation: {'✓' if enable_channel else '✗'}")
    click.echo(f"  Interference model: {'✓' if enable_interference else '✗'}")
    click.echo(f"  Adversarial jamming: {'✓' if enable_jamming else '✗'}")
    
    try:
        from dataset_generation.dataset_assembler import generate_dataset as gen
        summary = gen(num_scenarios, output, enable_channel, enable_interference, enable_jamming)
        click.echo(f"✅ Dataset generation complete!")
        click.echo(f"   Rate: {summary['scenarios_per_hour']:.0f} scenarios/hour")
        click.echo(f"   Output: {summary['output_base_dir']}")
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        raise


@cli.command()
@click.option('--input', default='data/raw/', help='Raw data directory')
@click.option('--output', default='data/processed/', help='Processed data directory')
def process_data(input: str, output: str):
    """Process raw data to learning states (Step 2)"""
    click.echo("🔧 Processing raw data...")
    from data_processing.preprocessing_pipeline import process_all
    process_all(input, output)
    click.echo("✅ Data processing complete!")


@cli.command()
@click.option('--method', default='adversarial_wf', 
              type=click.Choice(['classical_wf', 'adversarial_wf', 'diffract']))
@click.option('--num-scenarios', default=1000, help='Scenarios to optimize')
@click.option('--output', default='data/ground_truth/', help='Output directory')
def generate_labels(method: str, num_scenarios: int, output: str):
    """Generate optimal labels via ground-truth optimization (Step 3)"""
    click.echo(f"🎯 Generating ground-truth labels using {method}...")
    from optimization.teacher_dataset import generate_teacher_dataset
    generate_teacher_dataset(method, num_scenarios, output)
    click.echo("✅ Ground-truth labels generated!")


@cli.command()
@click.option('--model', default='supervised', 
              type=click.Choice(['supervised', 'rl', 'channel_predictor', 'traffic_predictor', 'jammer_detector']))
@click.option('--epochs', default=100, help='Training epochs')
@click.option('--device', default='cpu', help='Training device, such as cpu or cuda:0')
def train_model(model: str, epochs: int, device: str):
    """Train neural models (Step 4)"""
    click.echo(f"🤖 Training {model} model for {epochs} epochs...")
    
    if model == 'supervised':
        from models.supervised_learning import train_supervised
        train_supervised(epochs=epochs, device=device)
    elif model == 'rl':
        from models.reinforcement_learning import train_rl
        train_rl(num_episodes=epochs, device=device)
    elif model == 'channel_predictor':
        from models.channel_predictor import train_predictor
        train_predictor(epochs=epochs, device=device)
    elif model == 'traffic_predictor':
        from models.traffic_predictor import train_traffic_predictor
        train_traffic_predictor(epochs=epochs, device=device)
    else:  # jammer_detector
        from models.jammer_detector import train_detector
        train_detector(epochs=epochs, device=device)
    
    click.echo("✅ Model training complete!")


@cli.command()
@click.option('--agent-type', default='llm', 
              type=click.Choice(['llm', 'rl', 'hybrid']))
@click.option('--memory-size', default=1000, help='RAG memory size')
def build_agent(agent_type: str, memory_size: int):
    """Build AI agent with reasoning (Step 5)"""
    click.echo(f"🧠 Building {agent_type} agent with RAG memory (size={memory_size})...")
    from agent.ai_agent import build_agent as build
    agent = build(agent_type=agent_type, memory_size=memory_size)
    click.echo("✅ Agent built and ready for deployment!")
    return agent


@cli.command()
@click.option('--workflow', default='src/workflows/agent_workflow.nemo')
@click.option('--output', default='compiled_agent.ir')
@click.option('--optimize', default='aggressive', 
              type=click.Choice(['none', 'conservative', 'aggressive']))
def compile_workflow(workflow: str, output: str, optimize: str):
    """Compile agent workflow to NemoIR (Step 7)"""
    click.echo(f"🔨 Compiling NemoIR workflow with {optimize} optimization...")
    click.echo(f"   Input: {workflow}")
    click.echo(f"   Output: {output}")
    
    try:
        from workflows.compile_and_run import compile_workflow as compile_nemo_workflow
        compile_nemo_workflow(workflow, output, optimize)
        click.echo("✅ Workflow compiled successfully!")
    except (FileNotFoundError, RuntimeError) as error:
        click.echo(f"❌ Compilation failed: {error}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--baselines', default='all',
              type=click.Choice(['all', 'classical_wf', 'adversarial_wf', 'diffract', 'rl', 'llm', 'proposed']))
@click.option('--num-scenarios', default=100, help='Scenarios to evaluate')
@click.option('--output', default='experiments/results/')
def benchmark(baselines: str, num_scenarios: int, output: str):
    """Run comprehensive benchmarking (Step 8)"""
    click.echo(f"📈 Benchmarking {baselines} on {num_scenarios} scenarios...")
    from evaluation.benchmarks import run_benchmark
    results = run_benchmark(
        baselines=baselines if baselines != 'all' else None,
        num_scenarios=num_scenarios,
        output_dir=output
    )
    click.echo("✅ Benchmarking complete! Results saved to:")
    click.echo(f"   {output}")
    for baseline, metrics in results.items():
        click.echo(f"   {baseline}: spectral_eff={metrics['spectral_efficiency']['mean']:.2f}")


@cli.command()
@click.option('--agent', default='compiled', type=click.Choice(['compiled', 'python']))
@click.option('--duration', default=3600, help='Runtime duration (seconds)')
@click.option('--log', default='logs/online_execution.log')
def run_online(agent: str, duration: int, log: str):
    """Run agent for real-time online spectrum allocation (Step 9)"""
    click.echo(f"🚀 Starting online execution (agent={agent}, duration={duration}s)...")
    click.echo(f"   Logging to: {log}")
    
    from evaluation.online_executor import OnlineExecutor
    executor = OnlineExecutor(agent_type=agent, log_file=log)
    executor.run(duration_seconds=duration)
    
    click.echo("✅ Online execution complete!")


@cli.command()
@click.option('--duration', default=28800, help='Autoresearch runtime (seconds, default=8hrs)')
@click.option('--gpu', default=0, help='GPU device ID')
@click.option('--num-scenarios', default=3, help='Scenarios per experiment')
@click.option('--max-trials', default=4, help='Maximum experiments to run')
@click.option('--epochs', default=100, help='Supervised-training epochs per experiment')
def launch_autoresearch(duration: int, gpu: int, num_scenarios: int, max_trials: int, epochs: int):
    """Launch autonomous research loop (Step 10 + continuous improvement)"""
    click.echo(f"🔬 Launching AutoResearch loop (duration={duration}s on GPU {gpu})...")
    click.echo("   This will autonomously:")
    click.echo("   - Train a supervised allocation model per hypothesis")
    click.echo("   - Compare held-out MSE, spectral efficiency, and latency")
    click.echo("   - Record keep/discard results and learnings")
    click.echo("   - Reserve GPU training for the Slurm submission script")
    click.echo("")
    
    from research.experiment_runner import launch_autoresearch_loop
    summary = launch_autoresearch_loop(
        duration_seconds=duration,
        gpu_id=gpu,
        num_scenarios=num_scenarios,
        max_trials=max_trials,
        epochs=epochs,
    )
    
    click.echo(f"✅ AutoResearch complete: {summary['trials']} model trials across {summary['num_scenarios']} scenarios.")


@cli.command()
@click.option('--full', is_flag=True, help='Run full end-to-end pipeline')
@click.option('--stages', default='1,2,3,4,5,6,7,8', help='Stages to run (comma-separated)')
def pipeline(full: bool, stages: str):
    """Run full 10-step pipeline end-to-end"""
    if full:
        stage_list = list(range(1, 11))
    else:
        stage_list = [int(s) for s in stages.split(',')]
    
    click.echo("🚀 Starting AI Spectrum Management Pipeline")
    click.echo(f"   Stages: {stage_list}")
    click.echo("")
    
    # Stage 1: Dataset
    if 1 in stage_list:
        click.echo("Step 1️⃣  Dataset Generation...")
        ctx = click.Context(generate_dataset)
        ctx.invoke(generate_dataset, num_scenarios=100, output='data/raw/')
    
    # Stage 2: Processing
    if 2 in stage_list:
        click.echo("Step 2️⃣  Data Processing...")
        ctx = click.Context(process_data)
        ctx.invoke(process_data, input='data/raw/', output='data/processed/')
    
    # Stage 3: Optimization
    if 3 in stage_list:
        click.echo("Step 3️⃣  Ground-Truth Optimization...")
        ctx = click.Context(generate_labels)
        ctx.invoke(generate_labels, method='adversarial_wf', num_scenarios=100, output='data/ground_truth/')
    
    # Stage 4: Model Training
    if 4 in stage_list:
        click.echo("Step 4️⃣  Model Training (Supervised)...")
        ctx = click.Context(train_model)
        ctx.invoke(train_model, model='supervised', epochs=50, gpu=0)
    
    # Stage 5: Agent Building
    if 5 in stage_list:
        click.echo("Step 5️⃣  AI Agent Construction...")
        ctx = click.Context(build_agent)
        ctx.invoke(build_agent, agent_type='llm', memory_size=1000)
    
    # Stage 7: Workflow Compilation
    if 7 in stage_list:
        click.echo("Step 7️⃣  NemoIR Workflow Compilation...")
        ctx = click.Context(compile_workflow)
        ctx.invoke(compile_workflow, workflow='src/workflows/agent_workflow.nemo', 
                  output='compiled_agent.ir', optimize='aggressive')
    
    # Stage 8: Benchmarking
    if 8 in stage_list:
        click.echo("Step 8️⃣  Evaluation & Benchmarking...")
        ctx = click.Context(benchmark)
        ctx.invoke(benchmark, baselines='all', num_scenarios=100, output='experiments/results/')
    
    click.echo("")
    click.echo("✅ Pipeline complete!")


if __name__ == '__main__':
    cli()

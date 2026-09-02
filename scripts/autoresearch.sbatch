#!/bin/bash
#SBATCH --job-name=ai-spectrum-autoresearch
#SBATCH --output=logs/autoresearch-%j.out
#SBATCH --error=logs/autoresearch-%j.err
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00

set -euo pipefail
cd /home/msai/qinh0007/ai-spectrum-ntn
mkdir -p logs
uv run src/cli.py launch-autoresearch --duration 600 --num-scenarios 3 --max-trials 4 --epochs 100 --gpu 0
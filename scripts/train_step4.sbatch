#!/bin/bash
#SBATCH --job-name=ai-spectrum-step4
#SBATCH --output=logs/step4-%j.out
#SBATCH --error=logs/step4-%j.err
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

set -euo pipefail
cd /home/msai/qinh0007/ai-spectrum-ntn
mkdir -p logs
uv run src/cli.py train-model --model supervised --epochs 100 --device cuda:0
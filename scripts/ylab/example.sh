#!/bin/bash
#YBATCH -r a6000_1
#SBATCH --nodes 1
#SBATCH -J ja_mgsm
#SBATCH --time=168:00:00
#SBATCH --output outputs/%j.out
#SBATCH --error errors/%j.err

. /etc/profile.d/modules.sh
module load cuda/11.7
module load cudnn/cuda-11.x/8.9.0
module load nccl/cuda-11.7/2.14.3
module load openmpi/4.0.5

source venv/bin/activate
modle_name=$1

python code_benchmark/test_prompt.py \
    --llm $modle_name \
    --device cuda \
    --folder vmlu_v1.5/test.jsonl

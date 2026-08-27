#!/bin/bash
#SBATCH --account=def-shwemaha_gpu
#SBATCH --gres=gpu:a100:3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=24G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=baseline_full_%j.out

module load python/3.11
source ~/envs/lnfmm/bin/activate
export WANDB_MODE=offline
cd $SCRATCH/lnfmm

if [ -f model_checkpoint_t2i.pt ]; then
    echo "Found existing checkpoint, attempting resume"
    python train.py --config params_t2i --resume model_checkpoint_t2i.pt
else
    echo "No checkpoint found, starting fresh"
    python train.py --config params_t2i
fi

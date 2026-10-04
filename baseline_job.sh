#!/bin/bash
#SBATCH --account=def-shwemaha
#SBATCH --gres=gpu:a100:3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=24G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --job-name=t2i_baseline
#SBATCH --output=/home/dowolabi/scratch/cross_modal_translation/runs/logs/%x_%j.out

CODE_DIR=/home/dowolabi/projects/def-shwemaha/dowolabi/cross_modal_translation/lnfmm
RUN_DIR=/home/dowolabi/scratch/cross_modal_translation/runs/t2i_baseline

module load python/3.11
source ~/envs/lnfmm/bin/activate
export WANDB_MODE=offline
export WANDB_DIR=$RUN_DIR

mkdir -p "$RUN_DIR"
cd "$RUN_DIR" || exit 1

# First run only: snapshot the config and vocab into the run folder
if [ ! -f params.json ]; then
    cp "$CODE_DIR/params.json" "$CODE_DIR/vocab_coco_full.pkl" .
fi

if [ -f model_checkpoint_t2i.pt ]; then
    echo "Found existing checkpoint, attempting resume"
    python "$CODE_DIR/train.py" --config params_t2i --resume model_checkpoint_t2i.pt
else
    echo "No checkpoint found, starting fresh"
    python "$CODE_DIR/train.py" --config params_t2i
fi

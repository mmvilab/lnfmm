#!/bin/bash
#SBATCH --account=aip-shwemaha
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --job-name=t2i_1gpu
#SBATCH --output=/home/dowolabi/scratch/cross_modal_translation/runs/logs/%x_%j.out

# Single-GPU variant of baseline_job.sh. Uses params_1gpu.json (num_gpus=1) and its own run folder,
# so it never touches the 3-GPU baseline checkpoint.

CODE_DIR=/home/dowolabi/projects/aip-shwemaha/dowolabi/cross_modal_translation/lnfmm

# TEST_RUN=1 sbatch t2i_1gpu_job.sh  -> separate folder, so it can't touch the real checkpoint
if [ "${TEST_RUN:-0}" = "1" ]; then
    RUN_DIR=/home/dowolabi/scratch/cross_modal_translation/runs/t2i_1gpu_test
else
    RUN_DIR=/home/dowolabi/scratch/cross_modal_translation/runs/t2i_1gpu
fi

module load python/3.11
source "$CODE_DIR/lnfmm_env/bin/activate" || exit 1
export WANDB_MODE=offline
export WANDB_DIR=$RUN_DIR
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$RUN_DIR"
cd "$RUN_DIR" || exit 1

# First run only: snapshot the config and vocab, so resumes keep the original settings
if [ ! -f params.json ]; then
    cp "$CODE_DIR/params_1gpu.json" params.json
    cp "$CODE_DIR/vocab_coco_full.pkl" .
fi

# Sanity check: GPUs visible to PyTorch must match num_gpus in the config
echo "Job $SLURM_JOB_ID on $(hostname), RUN_DIR=$RUN_DIR"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
python - <<'EOF' || exit 1
import json, torch
cfg = json.load(open('params.json'))['params_t2i']
n = torch.cuda.device_count()
print(f"torch sees {n} GPU(s); config num_gpus={cfg['num_gpus']}, batch_size={cfg['batch_size']}, test_run={cfg.get('test_run', False)}")
if n != int(cfg['num_gpus']):
    raise SystemExit("num_gpus in params.json does not match allocated GPUs")
EOF

if [ -f model_checkpoint_t2i.pt ]; then
    echo "Found existing checkpoint, attempting resume"
    python "$CODE_DIR/train.py" --config params_t2i --resume model_checkpoint_t2i.pt
else
    echo "No checkpoint found, starting fresh"
    python "$CODE_DIR/train.py" --config params_t2i
fi

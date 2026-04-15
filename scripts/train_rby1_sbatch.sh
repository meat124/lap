#!/bin/bash
#SBATCH --job-name=PuttingCupintotheDish_demo50_uniform_eef
#SBATCH --partition=gigabyte_a6000
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=/lustre/meat124/lap/logs/%x_%j.out
#SBATCH --error=/lustre/meat124/lap/logs/%x_%j.err

# ---------------------------------------------------------------
# Usage:
#   sbatch scripts/train_rby1_sbatch.sh                      # all episodes, default batch size
#   BATCH_SIZE=16 sbatch scripts/train_rby1_sbatch.sh        # override batch size
#   sbatch scripts/train_rby1_sbatch.sh --num-train-steps 20001
#
# Episode count control (edit NUM_EPISODES below, or pass via CLI):
#   NUM_EPISODES=100 sbatch scripts/train_rby1_sbatch.sh     # first 100 episodes
#   sbatch scripts/train_rby1_sbatch.sh --data.num-episodes 50
#
# Batch size control (edit BATCH_SIZE below, or set as env var):
#   BATCH_SIZE=16 sbatch scripts/train_rby1_sbatch.sh        # e.g. for memory-limited GPUs
#   BATCH_SIZE=64 sbatch scripts/train_rby1_sbatch.sh        # larger batch
#
# EEF-space training (end-effector action space, action_dim=7):
#   CONFIG_NAME=lap_rby1_eef REPO_ID=PuttingCupintotheDishV2_eef \
#     sbatch scripts/train_rby1_sbatch.sh
#
# Prerequisites:
#   1. Download the LAP-3B checkpoint once:
#        cd /lustre/meat124/lap
#        uv run huggingface-cli download lihzha/LAP-3B \
#            --local-dir ./checkpoints/lap --repo-type model
#   2. Ensure the rby1 LeRobot dataset is at:
#        /lustre/meat124/rby1_demo/LeRobotDataset_v2/PuttingCupintotheDishV2
#   3. For EEF training, convert the dataset first:
#        python scripts/convert_rby1_joint_to_eef.py \
#            --input-dir /lustre/meat124/rby1_demo/LeRobotDataset_v2/PuttingCupintotheDishV2 \
#            --output-dir /lustre/meat124/rby1_demo/LeRobotDataset_v2/PuttingCupintotheDishV2_eef
# ---------------------------------------------------------------

set -euo pipefail

# --- configurable paths ---
LAP_DIR=/lustre/meat124/lap
DATASET_ROOT=/lustre/meat124/rby1_demo/LeRobotDataset_v2

# Config name: lap_rby1 (joint-space, 8D) or lap_rby1_eef (EEF-space, 7D)
CONFIG_NAME=${CONFIG_NAME:-"lap_rby1_eef"}

# repo_id: LeRobot dataset folder name under DATASET_ROOT
REPO_ID=${REPO_ID:-"PuttingCupintotheDishV2_50_eef"}
# asset_id: norm stats folder name under assets/<config>/
# Reuse the full-dataset norm stats unless a dedicated one exists.
ASSET_ID=${ASSET_ID:-"PuttingCupintotheDishV2_50_eef"}

# --- dataset size control ---
# Set NUM_EPISODES to limit training to the first N episodes.
# Leave empty (default) to use all available episodes.
# Examples: NUM_EPISODES=50, NUM_EPISODES=100, NUM_EPISODES=300
NUM_EPISODES=${NUM_EPISODES:-""}

# --- batch size control ---
# Set BATCH_SIZE to override the default (32) from the lap_rby1 config.
# Leave empty to use the config default.
# Examples: BATCH_SIZE=16 (memory-limited), BATCH_SIZE=64 (multi-GPU)
BATCH_SIZE=${BATCH_SIZE:-"32"}

# HF_LEROBOT_HOME must be exported BEFORE Python starts so that the lerobot
# module resolves the dataset root at import time.
export HF_LEROBOT_HOME="$DATASET_ROOT"

# Output / checkpoint dir is derived from the SLURM job name so that each run
# gets a unique, identifiable directory (matching WandB run name).
OUTPUT_DIR="checkpoints/lap_rby1/${SLURM_JOB_NAME}"

cd "$LAP_DIR"

# create log dir if needed
mkdir -p "$LAP_DIR/logs"

echo "=============================="
echo "Job ID       : $SLURM_JOB_ID"
echo "Job Name     : $SLURM_JOB_NAME"
echo "Config       : $CONFIG_NAME"
echo "Dataset root : $DATASET_ROOT"
echo "Repo ID      : $REPO_ID"
echo "Asset ID     : $ASSET_ID"
echo "Num episodes : ${NUM_EPISODES:-all}"
echo "Batch size   : ${BATCH_SIZE:-default}"
echo "Checkpoint   : $OUTPUT_DIR"
echo "Node         : $(hostname)"
echo "Start        : $(date)"
echo "=============================="

# --- download pretrained weights if not already present ---
CHECKPOINT_PARAMS="$LAP_DIR/checkpoints/lap/params"
if [ ! -d "$CHECKPOINT_PARAMS" ]; then
    echo "LAP-3B checkpoint not found. Downloading from HuggingFace..."
    uv run huggingface-cli download lihzha/LAP-3B \
        --local-dir ./checkpoints/lap --repo-type model
    echo "Download complete."
fi

# --- compute normalization stats if not already present ---
NORM_STATS_FILE="$LAP_DIR/assets/${CONFIG_NAME}/${ASSET_ID}/norm_stats.json"
if [ ! -f "$NORM_STATS_FILE" ]; then
    echo "Norm stats not found. Computing from dataset..."
    mkdir -p "$LAP_DIR/assets/${CONFIG_NAME}"
    HF_LEROBOT_HOME="$DATASET_ROOT" uv run scripts/compute_norm_stats.py "$CONFIG_NAME" --repo-id "${ASSET_ID}"
    echo "Norm stats computed."
fi

# --- train ---
# JAX_PLATFORMS=cuda selects the CUDA backend.
# All extra arguments passed via sbatch are forwarded to train.py.
# --data.num-episodes / --batch-size are only passed when the corresponding
# variables are non-empty.
JAX_PLATFORMS=cuda \
uv run --group cuda scripts/train.py "$CONFIG_NAME" \
    --exp-name="$SLURM_JOB_NAME" \
    --data.repo-id "${REPO_ID}" \
    --data.assets.asset-id "${ASSET_ID}" \
    ${NUM_EPISODES:+--data.num-episodes "${NUM_EPISODES}"} \
    ${BATCH_SIZE:+--batch-size "${BATCH_SIZE}"} \
    "$@"

echo "=============================="
echo "End          : $(date)"
echo "=============================="

#!/bin/bash
#SBATCH --job-name=sft    # Name of the job
#SBATCH --output=logs/sft_%j.out   # Output file (%j will be replaced by job ID)
#SBATCH --error=logs/sft_%j.err    # Error file
#SBATCH --time=48:00:00          # Maximum runtime in HH:MM:SS
#SBATCH --partition=vision-agrawal-a100
#SBATCH --qos=vision-agrawal-free-cycles
#SBATCH --nodes=1                # Number of nodes
#SBATCH --ntasks-per-node=1      # Number of tasks per node
#SBATCH --cpus-per-task=16       # Number of CPU cores per task
#SBATCH --mem=64G               # Memory per node
#SBATCH --gres=gpu:1             # Number of GPUs required

# Set environment variable
export TF_CPP_MIN_LOG_LEVEL=2

# Run the training script
python3 train_sft.py \
    --game-class="games.go_game.GoBoard9x9" \
    --agent-class="policies.resnet_policy.ResnetPolicyValueNet128" \
    --training-batch-size=1024 \
    --learning-rate=1e-3 \
    --random-seed=42 \
    --ckpt-filename="./sft_agent.ckpt" \
    --num-epochs=10 \
    --dataset-path="./go9x9_data.npz"
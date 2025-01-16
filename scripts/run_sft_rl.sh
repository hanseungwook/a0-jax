#!/bin/bash
#SBATCH --job-name=go_training    # Name of the job
#SBATCH --output=logs/go_%j.out        # Output file (%j will be replaced by job ID)
#SBATCH --error=logs/go_%j.err         # Error file
#SBATCH --time=48:00:00          # Maximum runtime in HH:MM:SS
#SBATCH --partition=vision-pulkitag-h100,vision-pulkitag-a100
#SBATCH --qos=vision-pulkitag-main
#SBATCH --nodes=1                # Number of nodes
#SBATCH --ntasks-per-node=1      # Number of tasks per node
#SBATCH --cpus-per-task=16        # Number of CPU cores per task
#SBATCH --mem=256G                # Memory per node
#SBATCH --gres=gpu:4             # Number of GPUs required

# Set environment variable
export TF_CPP_MIN_LOG_LEVEL=2

# Run the training script
python3 train_agent_rl.py \
    --game-class="games.go_game.GoBoard9x9" \
    --agent-class="policies.resnet_policy.ResnetPolicyValueNet128" \
    --selfplay-batch-size=1024 \
    --training-batch-size=1024 \
    --num-simulations-per-move=32 \
    --num-self-plays-per-iteration=102400 \
    --learning-rate=1e-2 \
    --random-seed=42 \
    --ckpt-filename="./sft_rl_checkpoints/sft_rl_agent.ckpt" \
    --num-iterations=200 \
    --lr-decay-steps=1000000 \
    --kl-coef=1.0
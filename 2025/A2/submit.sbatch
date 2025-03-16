#!/bin/bash
#SBATCH --job-name=jupyter_server
#SBATCH --time=02:00:00              # Adjust time limit as needed
#SBATCH --partition=falcon,gecko   # Adjust based on your cluster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4            # Adjust CPU count
#SBATCH --mem=16G                    # Adjust memory
#SBATCH --gres=gpu:1                  # Remove if not using GPU
#SBATCH --output=joboutput_%j.out # Standard out from your job
#SBATCH --error=joboutput_%j.err  # Standard error from your job

# Load required modules (if applicable)
module load cuda  # Adjust based on your cluster

# Activate your virtual environment
source ~/modules/cs429/CS909-materials/.venv/bin/activate

# Get the node and port information
NODE=$(hostname)
PORT=8888  # Change if needed

# Start Jupyter Notebook without a browser, binding to all interfaces
papermill q3_testing.ipynb test_output.ipynb
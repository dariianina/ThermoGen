#!/bin/bash
#SBATCH --job-name="thermo_unet"
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1,VRAM:48G
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --mail-type=FAIL,INVALID_DEPEND,STAGE_OUT,TIME_LIMIT,REQUEUE
#SBATCH --output=/home/stud/nobis/storage/slurm/logs/slurm-%j.out
#SBATCH --error=/home/stud/nobis/storage/slurm/logs/slurm-%j.out

srun python train_unet.py
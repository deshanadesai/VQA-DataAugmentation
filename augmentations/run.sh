#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=60GB
#SBATCH --time=48:00:00
#SBATCH --job-name=ColorAugmentation
#SBATCH --mail-type=END
#SBATCH --mail-user=dkd266@nyu.edu
#SBATCH --output=slurm_%j.out
python get_colors.py image_id.txt


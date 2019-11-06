#! /bin/bash

#SBATCH -A wangjian 
#SBATCH -p gpu
#SBATCH --gres=gpu:1
module load pytorch
. activate /home/yaoli/project/conda
python main.py -st

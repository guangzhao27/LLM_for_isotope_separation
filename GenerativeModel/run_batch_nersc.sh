#!/bin/bash
#SBATCH --qos=regular
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --constraint=gpu
#SBATCH --gpus-per-node=1
#SBATCH --output=llama2-13b.txt
#SBATCH --account=m4138_g

conda activate
python llama2_13b.py
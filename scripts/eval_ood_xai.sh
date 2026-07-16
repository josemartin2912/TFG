#!/bin/bash


#SBATCH --job-name eval_ood                 # Nombre del proceso

#SBATCH --partition dios  # Cola para ejecutar

#SBATCH --gres=gpu:1                           # Numero de gpus a usar


#SBATCH --output=./eval_resnet50_xai.out



export PATH="/opt/anaconda/anaconda3/bin:$PATH"

export PATH="/opt/anaconda/bin:$PATH"

eval "$(conda shell.bash hook)"

conda activate /mnt/homeGPU/jmartin/xai_env/


export TFHUB_CACHE_DIR=.


/mnt/homeGPU/jmartin/xai_env/bin/python /mnt/homeGPU/jmartin/TFG/scripts/eval_ood.py 

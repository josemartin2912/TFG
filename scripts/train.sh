#!/bin/bash


#SBATCH --job-name entrenamiento                 # Nombre del proceso

#SBATCH --partition dios  # Cola para ejecutar

#SBATCH --gres=gpu:1                           # Numero de gpus a usar

#SBATCH --exclude=atenea,titan,zeus
#SBATCH --output=./train_resnet50.out


export PATH="/opt/anaconda/anaconda3/bin:$PATH"

export PATH="/opt/anaconda/bin:$PATH"

eval "$(conda shell.bash hook)"

conda activate /mnt/homeGPU/jmartin/entorno_jose/

export TFHUB_CACHE_DIR=.


/mnt/homeGPU/jmartin/entorno_jose/bin/python /mnt/homeGPU/jmartin/TFG/scripts/train.py  --config /mnt/homeGPU/jmartin/TFG/configs/train.yml        



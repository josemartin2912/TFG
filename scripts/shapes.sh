#!/bin/bash


#SBATCH --job-name Prueba                 # Nombre del proceso

#SBATCH --partition dios  # Cola para ejecutar

#SBATCH --gres=gpu:0                           # Numero de gpus a usar

        

        export PATH="/opt/anaconda/anaconda3/bin:$PATH"

export PATH="/opt/anaconda/bin:$PATH"

eval "$(conda shell.bash hook)"

conda activate /mnt/homeGPU/Environments/tf2.2py36

export TFHUB_CACHE_DIR=.


python shape_img.py          


mail -s "Proceso finalizado" jmartin02@correo.ugr.es <<< "El proceso ha finalizado"

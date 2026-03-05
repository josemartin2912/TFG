pip install fastai
# Carga del dataset
from pathlib import Path
from fastai.vision.all import *

path = Path('/mnt/homeGPU/isevillano/Github/GleasonMap/Data/TrainTest')
path_train = path/'train'
path_valid = path/'valid'
# Obtenemos las clases
class_names = sorted([d.name for d in path_train.iterdir() if d.is_dir()])
class_to_idx = {name: i for i, name in enumerate(class_names)}
print(class_names)
print("Número de clases:", len(class_names))

# Transformaciones del conjunto de datos y construccion
# de Dataloader

dls = DataBlock(
    blocks = (ImageBlock, CategoryBlock),
    get_items = get_image_files,
    get_y = parent_label,
    splitter = GrandparentSplitter(train_name='train', valid_name='val')
).dataloaders(path, bs=64)


# metricas seleccionadas y funcion de perdida

f_perdida = CrossEntropyFlat()
metricas = accuracy
# Definicion del modelo
import timm
import torch.nn as nn

model = timm.create_model(
    'swin_tiny_patch4_window7_224',
    pretrained=True
)


learn = Learner(
    dls,
    model,
    loss_func=CrossEntropyLossFlat(),
    metrics=metricas
)
# Entrenamiento

fine_tune(epochs = 20)
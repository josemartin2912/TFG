import torch
import argparse
from openood.utils import config, setup_logger
from openood.datasets import get_dataloader
from openood.networks import get_network
from openood.preprocessors import get_preprocessor
from openood.evaluators import get_evaluator
from openood.datasets import get_ood_dataloader
from XAIPostprocessor import XAIPostProcessor
from MahaPostProcessor import MahaPostProcessor
from RMDXAIPostprocessor import RMDXAIPostProcessor
from RMDPostprocessor import RMDPostProcessor
from PIL import Image

img = Image.open("/mnt/homeGPU/isevillano/Github/GleasonMap/Data/TrainTest/train/Estroma/30112463_004033_003809_000224.png")

cfg = config.Config('/mnt/homeGPU/jmartin/TFG/configs/train.yml',
                    '/mnt/homeGPU/jmartin/TFG/configs/eval_ood.yml')


# -------------------------
# Setup
# -------------------------
setup_logger(cfg)
torch.manual_seed(cfg.seed)

# -------------------------
# Preprocessor
# -------------------------
# Para evaluación usamos ID val y OOD
preprocessor = get_preprocessor(cfg, split='val')

# -------------------------
# Dataloaders
# -------------------------
dataloaders = get_dataloader(cfg)
val_loader = dataloaders['val']  
id_data_loaders = {'test': dataloaders['val'], 'train': dataloaders['train']}
ood_data_loaders = get_ood_dataloader(cfg)
# OpenOOD cargará OOD automáticamente según ood_dataset

# -------------------------
# Network
# -------------------------
net = get_network(cfg.network)
#net = net.cuda() #eliminar

# Cargar checkpoint entrenado
if cfg.network.checkpoint is not None:
    state = torch.load(cfg.network.checkpoint)
    net.load_state_dict(state, strict=False)



postprocessor = XAIPostProcessor(cfg) # opcion evaluar mahalanobis en XAI + features
#postprocessor = MahaPostProcessor(cfg) # opcion evaluar mahalanobis en features
#postprocessor = RMDXAIPostProcessor(cfg) # opcion evaluar relative mahalanobis en features + XAI
#postprocessor = RMDPostProcessor(cfg) # opcion evaluar relative mahalanobis en features
postprocessor.setup(net, img, ood_data_loaders)

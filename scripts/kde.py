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
from BestXAIPostProcessor_trans import BestXAIPostProcessor_trans
from BestXAIPostProcessor import BestXAIPostProcessor
import timm
from scipy.stats import gaussian_kde
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------
# Este script se encarga de estimar la funcion de densidad de
# distribuciones de datos generados por un postprocessor. De
# esta forma, veremos la separacion entre distribuciones ID y OOD
# de forma grafica.
# -----------------------------------------------------------

# Configuracion de DeiT
#cfg = config.Config('/mnt/homeGPU/jmartin/TFG/configs/deit.yml',
#                    '/mnt/homeGPU/jmartin/TFG/configs/eval_ood.yml')

# Configuracion de resnet50
cfg = config.Config('/mnt/homeGPU/jmartin/TFG/configs/train.yml',
                    '/mnt/homeGPU/jmartin/TFG/configs/eval_ood.yml')

# Ficheros de resultados de DeiT y resnet50 (resp)
#cfg.output_dir = "/mnt/homeGPU/jmartin/TFG/results/DeiT" 
cfg.output_dir = "/mnt/homeGPU/jmartin/TFG/results/xai" 

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


# -------------------------
# Network: DeiT
# -------------------------
#net = timm.create_model(
#    'deit_base_patch16_224',
#    pretrained=True,
#    num_classes=cfg.num_classes
#)
#net = net.cuda()

#state = torch.load("/mnt/homeGPU/jmartin/TFG/results/DeiT/best_epoch31_acc0.9670.ckpt")
#net.load_state_dict(state, strict=False)
# -------------------------
# Network: Cargada del fichero de config, en este caso resnet50
# -------------------------
net = get_network(cfg.network)
net = net.cuda()

# Cargar checkpoint entrenado
if cfg.network.checkpoint is not None:
    state = torch.load(cfg.network.checkpoint)
    net.load_state_dict(state, strict=False)

# -------------------------
# Evaluator
# -------------------------
evaluator = get_evaluator(cfg)

# -------------------------
# Postprocessors
# -------------------------
#postprocessor = XAIPostProcessor(cfg)
#postprocessor = BestXAIPostProcessor_trans(cfg)
#postprocessor = BestXAIPostProcessor(cfg)
postprocessor = MahaPostProcessor(cfg)
postprocessor.setup(net, id_data_loaders, ood_data_loaders)

# Calculamos scores id
scores_id = []

for batch in id_data_loaders['train']:
    data = batch['data'].cuda()

    pred, score = postprocessor.postprocess(net, data)

    scores_id.extend(score.cpu().numpy())

# Calculamos scores near_ood
scores_near_ood = []

for batch in ood_data_loaders['nearood']['covid']:
    data = batch['data'].cuda()

    pred, score = postprocessor.postprocess(net, data)

    scores_near_ood.extend(score.cpu().numpy())

# Calculamos scores far_ood
scores_far_ood = []

for batch in ood_data_loaders['farood']['cifar10']:
    data = batch['data'].cuda()

    pred, score = postprocessor.postprocess(net, data)

    scores_far_ood.extend(score.cpu().numpy())

# Calculamos las funciones de densidad de los scores con el metodo kde
# (kernel density estimation) 
kde_id = gaussian_kde(scores_id)
kde_covid = gaussian_kde(scores_near_ood)
kde_cifar = gaussian_kde(scores_far_ood)

# Rango común de representacion del eje x
xmin = max(min(min(scores_id), min(scores_near_ood), min(scores_far_ood)), -20000)
xmax = max(max(scores_id), max(scores_near_ood), max(scores_far_ood))

# Añadimos un pequeño margen
margen = 0.05 * (xmax - xmin)
x = np.linspace(xmin - margen, xmax + margen, 1000)

# Plot de las 3 funciones de densidad
plt.figure(figsize=(8,5))

plt.plot(x, kde_id(x), label="ID", linewidth=2)
plt.plot(x, kde_covid(x), label="COVID (Near-OOD)", linewidth=2)
plt.plot(x, kde_cifar(x), label="CIFAR10 (Far-OOD)", linewidth=2)

plt.xlabel("Mahalanobis distance")
plt.ylabel("Density")
plt.title("Kernel Density Estimation of Mahalanobis distances based on features")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

# Guardamos los resultados en funcion del modelo usado.
plt.savefig(
#    "/mnt/homeGPU/jmartin/TFG/results/DeiT/kde.png",
    "/mnt/homeGPU/jmartin/TFG/results/xai/kde.png",
    dpi=300,
    bbox_inches="tight"
)


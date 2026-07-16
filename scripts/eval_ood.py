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

# Este archivo se encarga de realizar la evaluacion OOD. 


# Ficheros de configuracion para DeiT
cfg = config.Config('/mnt/homeGPU/jmartin/TFG/configs/deit.yml',
                    '/mnt/homeGPU/jmartin/TFG/configs/eval_ood.yml')

# Ficheros de configuracion para resnet50
#cfg = config.Config('/mnt/homeGPU/jmartin/TFG/configs/train.yml',
#                   '/mnt/homeGPU/jmartin/TFG/configs/eval_ood.yml')

# Fichero de resultados para DeiT
cfg.output_dir = "/mnt/homeGPU/jmartin/TFG/results/DeiT" 

# Fichero de resultados para resnet. Dependiendo del postprocessor
# utilizado la ruta sera .../results/xai, /maha, /rmd, /xai
#cfg.output_dir = "/mnt/homeGPU/jmartin/TFG/results" 


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
# Network: modelo DeiT de timm
# -------------------------
net = timm.create_model(
    'deit_base_patch16_224',
    pretrained=True,
    num_classes=cfg.num_classes
)
net = net.cuda()

# Cargamos el checkpoint de la epoca 40. Se habia entrenado antes de una interrupcion
# en la epoca 9, por eso aparece que el mejor modelo fue en la epoca 31.
state = torch.load("/mnt/homeGPU/jmartin/TFG/results/DeiT/best_epoch31_acc0.9670.ckpt")
net.load_state_dict(state, strict=False)

# -------------------------
# Network: En este caso resnet50
# -------------------------
#net = get_network(cfg.network)
#net = net.cuda()

# Cargar checkpoint entrenado
#if cfg.network.checkpoint is not None:
#    state = torch.load(cfg.network.checkpoint)
#    net.load_state_dict(state, strict=False)

# -------------------------
# Evaluator
# -------------------------
evaluator = get_evaluator(cfg)

# -------------------------
# Postprocessor
# -------------------------

# Postprocessors disponibles.
# CNN:
# - XAIPostProcessor: calcular distribucion de features y CRP concatenados.
#                     Scores -> distancia de mahalanobis a distribucion [feat,crp]
# - MahaPostProcessor: calcular distribucion de features.
#                     Scores -> distancia de mahalanobis a distribucion [feat]
# - RMDPostProcessor: calcular distribucion de features.
#                     Scores -> distancia relative mahalanobis a distribucion [feat]
# - RMDXAIPostProcessor: calcular distribucion de features y CRP concatenados.
#                     Scores -> distancia relative mahalanobis a distribucion [feat,crp]
# - XAIPostProcessor_norm: calcular distribucion de features y CRP concatenados (normalizados).
#                     Scores -> distancia de mahalanobis a distribucion [feat,crp]
# - BestXAIPostProcessor: calcular distribucion de features.
#                     Scores -> Para cada muestra calcular features y CRP. Tomar 
#                               el p% indices de los mayores valores en valor absoluto
#                               para cada ejemplo. Calcular la distancia de mahalanobis
#                               tomando únicamente las features en las posiciones 
#                               de los indices anteriores
#
# Transformers:
# - BestXAIPostProcessor_trans: calcular distribucion de features.
#                     Scores -> Para cada muestra calcular features y CRP. Tomar 
#                               el p% indices de los mayores valores en valor absoluto
#                               para cada ejemplo. Calcular la distancia de mahalanobis
#                               tomando únicamente las features en las posiciones 
#                               de los indices anteriores

# Usamos el postprocessor BestXAIPostProcessor
postprocessor = BestXAIPostProcessor_trans(cfg)

# Mismo postprocessor para resnet50
# postprocessor = BestXAIPostProcessor(cfg) 
postprocessor.setup(net, id_data_loaders, ood_data_loaders)

# -------------------------
# Evaluación OOD
# -------------------------
print("Evaluando OOD con XAI...")
ood_metrics = evaluator.eval_ood(
    net,
    id_data_loaders=id_data_loaders,
    ood_data_loaders=ood_data_loaders,
    postprocessor=postprocessor
)

print(ood_metrics)
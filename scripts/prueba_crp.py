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
from torchvision import transforms
from zennit.composites import EpsilonPlusFlat
from crp.attribution import CondAttribution
from crp.concepts import ChannelConcept

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

img = Image.open("/mnt/homeGPU/isevillano/Github/GleasonMap/Data/TrainTest/train/Estroma/30112463_004033_003809_000224.png").convert("RGB")

img = transform(img)

img = img.unsqueeze(0)

cfg = config.Config('/mnt/homeGPU/jmartin/TFG/configs/train.yml',
                    '/mnt/homeGPU/jmartin/TFG/configs/eval_ood.yml')
cfg.network.num_gpus = 0
cfg.num_gpus = 0
cfg.dataset.num_gpus = 0


# -------------------------
# Setup
# -------------------------
setup_logger(cfg)
torch.manual_seed(cfg.seed)

# -------------------------
# Preprocessor
# -------------------------
# Para evaluación usamos ID val y OOD
#preprocessor = get_preprocessor(cfg, split='val')

# -------------------------
# Dataloaders
# -------------------------
#dataloaders = get_dataloader(cfg)
#val_loader = dataloaders['val']  
#id_data_loaders = {'test': dataloaders['val'], 'train': dataloaders['train']}
ood_data_loaders = get_ood_dataloader(cfg)
# OpenOOD cargará OOD automáticamente según ood_dataset

# -------------------------
# Network
# -------------------------
net = get_network(cfg.network)
#net = net.cuda() #eliminar

# Cargar checkpoint entrenado
if cfg.network.checkpoint is not None:
    state = torch.load(cfg.network.checkpoint,
    map_location=torch.device('cpu'))
    net.load_state_dict(state, strict=False)

# Elementos para el calculo del vector 
# de explicaciones con CRP. Operamos
# en la capa 4 de resnet50 que es la ultima 
# con convoluciones.
composite = EpsilonPlusFlat()
attribution = CondAttribution(net)
concept = ChannelConcept()
img = img.clone().detach().requires_grad_(True)

          
# Extraemos logits y features
logits, feature = net(img, return_feature=True)

# Calculamos las predicciones de los logits, 
# necesario para el metodo CRP
pred = logits.argmax(dim=1)


# condiciones: propagar relevancia respecto a la clase predicha
conditions = [{"y": [p.item()]} for p in pred]

# Aplicamos CRP
attr = attribution(
    img,
    conditions=conditions,
    composite=composite,
    record_layer=["layer4"]
)

#print(type(attr))
#print(len(attr))
#print(attr)
relevance = concept.attribute(attr.relevances["layer4"], abs_norm=True) 
print(relevance.max())
print(relevance.min())
print(feature.max())
print(feature.min())
print(f"Dimension tensor {feature.shape}")
feat_plus_xai = torch.cat([feature, relevance], dim=-1)

print(f"Dimension tensor {feat_plus_xai.shape}")
# postprocessor = XAIPostProcessor(cfg) # opcion evaluar mahalanobis en XAI + features
#postprocessor = MahaPostProcessor(cfg) # opcion evaluar mahalanobis en features
#postprocessor = RMDXAIPostProcessor(cfg) # opcion evaluar relative mahalanobis en features + XAI
#postprocessor = RMDPostProcessor(cfg) # opcion evaluar relative mahalanobis en features
#postprocessor.setup(net, img, ood_data_loaders)

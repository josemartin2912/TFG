import torch
import argparse
from openood.utils import config, setup_logger
from openood.datasets import get_dataloader
from openood.networks import get_network
from openood.preprocessors import get_preprocessor
from openood.trainers import get_trainer
from openood.evaluators import get_evaluator
from openood.recorders import get_recorder
import timm


#----------------------------------------------------------------------
# Script de entrenamiento
#----------------------------------------------------------------------


# -------------------------
# Load config
# -------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True)
args = parser.parse_args()

config = config.Config(args.config)

# -------------------------
# Setup
# -------------------------
setup_logger(config)
torch.manual_seed(config.seed)

# -------------------------
# Preprocessor
# -------------------------
preprocessor = get_preprocessor(config,  split='train')

# -------------------------
# Dataloaders
# -------------------------
dataloader_dict = get_dataloader(config)
train_loader = dataloader_dict['train']
val_loader   = dataloader_dict['val']

# -------------------------
# Network
# -------------------------

# Modelo del fichero de config. En este caso resnet50
#net = get_network(config.network)

# Modelo DeiT de timm
net = timm.create_model(
    'deit_base_patch16_224',
    pretrained=True,
    num_classes=config.num_classes
)
net = net.cuda()



recorder = get_recorder(config)
evaluator = get_evaluator(config)
# -------------------------
# Trainer
# -------------------------
trainer = get_trainer(net, train_loader, val_loader, config)
print('Entrenando')
for epoch_idx in range(1, config.optimizer.num_epochs + 1):
    
    # Entrenar una época
    net, train_metrics = trainer.train_epoch(epoch_idx)
    
    # Evaluar en validación
    val_metrics = evaluator.eval_acc(net, val_loader, epoch_idx=epoch_idx)
    
    # Guardar modelo y reportar métricas
    recorder.save_model(net, val_metrics)
    recorder.report(train_metrics, val_metrics)

recorder.summary()
import torch

# -----------------------------------------------------------
# Script que descarga el modelo resnet50 de pytorch
# -----------------------------------------------------------
# Descarga y carga los pesos directamente desde la URL
state_dict = torch.hub.load_state_dict_from_url(
    'https://download.pytorch.org/models/resnet50-0676ba61.pth',
    map_location='cpu'
)

del state_dict['fc.weight']
del state_dict['fc.bias']

torch.save(state_dict, '/mnt/homeGPU/jmartin/entorno_jose/resnet50_imagenet_no_fc.ckpt')
print("Checkpoint guardado correctamente")
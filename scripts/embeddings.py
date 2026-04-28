import torch
from torchvision import models
import torch.nn as nn
from openood.datasets import get_dataloader
from openood.datasets import get_ood_dataloader

# modelo base
model = models.resnet50(pretrained=False)

# cambiar la última capa si tu modelo tiene 14 clases
model.fc = torch.nn.Linear(2048, 14)

# cargar checkpoint
state = torch.load("best_epoch37_acc0.9659.ckpt")
model.load_state_dict(state, strict=False)

model.eval().cuda()



# eliminamos la última capa (fc)
feature_extractor = nn.Sequential(*list(model.children())[:-1])
feature_extractor.eval().cuda()

def get_embeddings(dataloader, model):
    all_embeddings = []

    with torch.no_grad():
        for batch in dataloader:
            x = batch['data'].cuda()

            feat = model(x)              
            feat = feat.view(feat.size(0), -1)  

            all_embeddings.append(feat.cpu())

    return torch.cat(all_embeddings, dim=0)

dataloaders = get_dataloader(cfg)
val_loader = dataloaders['val']  
id_data_loaders = {'test': dataloaders['val'], 'train': dataloaders['train']}
ood_data_loaders = get_ood_dataloader(cfg)

id_embeddings = get_embeddings(id_loader, feature_extractor)
ood_embeddings = get_embeddings(ood_loader, feature_extractor)

print(id_embeddings.shape)
print(ood_embeddings.shape)
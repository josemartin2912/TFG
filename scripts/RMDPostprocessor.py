import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from typing import Any
from numpy.linalg import pinv
from zennit.composites import EpsilonPlusFlat
from crp.attribution import CondAttribution
from crp.concepts import ChannelConcept
from openood.postprocessors.base_postprocessor import BasePostprocessor

#----------------------------------------------------------------------
# Postprocessor que calcula la distribucion de features ID y 
# calcula la relative mahalanobis distance a esta distribucion
# para devolver scores.
#----------------------------------------------------------------------

class RMDPostProcessor(BasePostprocessor): 

    def __init__(self, config):
        super().__init__(config)
        self.setup_flag = False
        self.num_classes = self.config.num_classes


    # Metodo que define el estado inicial del postprocessor. Para ello,
    # tomaremos las caracteristicas de nuestro id_loader, calcularemos la media y la
    # covarianza de la distribucion ID, con el objetivo de 
    # realizar los scores a partir de esta distribución.
    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        
        # Si no se ha hecho setup del postprocessor
        if not self.setup_flag:

            # Activamos el modo evaluacion
            net.eval()

            # Lista de listas de features por clase
            class_features = [[] for _ in range(self.num_classes)]           
            for batch in tqdm(id_loader_dict['train'],
                                desc='Setup: ', 
                                position=0, 
                                leave=True): 
                
                # Pasamos los datos a GPU y float
                data = batch['data'].cuda() 
                labels = batch['label'].cuda()
                data = data.float() 
                # Eliminamos el calculo de gradientes. Solo 
                # necesitamos extraer features
                with torch.no_grad():            
                    # Extraemos logits y features
                    _, feature = net(data, return_feature=True)
                
                
                # Introducimos cada feature en la lista de su clase
                for i in range(feature.shape[0]):
                    label = labels[i].item()
                    class_features[label].append(feature[i].detach().cpu())

                del feature
                torch.cuda.empty_cache()

            # La lista de cada clase se organiza en forma de batches y no ejemplos.
            # Con esto, hacemos un stack de todos los batches y pasamos a ejemplos.
            class_features = [
                torch.stack(c, dim=0) for c in class_features
            ]

            # Tomamos todas las caracteristicas
            all_features = torch.cat(class_features, dim=0)

            # Nuestro metodo esta basado en la distancia de mahalanobis.
            # Esta distancia es d(x) = (x - μ)^T Σ^{-1} (x - μ) donde:
            # - x es la muestra a calcular la distancia.
            # - μ es la media de las muestras ID.
            # - Σ es la covarianza entre las muestras ID.
            # Para calcular luego los scores calculamos estos parámetros

            # Calculamos la media de cada clase 
            mean_per_class = torch.stack(
                [c.mean(dim=0) for c in class_features],
                dim=0
            )

            # Calculamos la media global
            mean = all_features.mean(dim=0)

            # Para el calculo de la inversa de la covarianza, se introduce
            # un término regulatorio eps en caso de ser singular
            #  y se calcula su pseusoinversa, por estabilidad numerica
            
            dif = all_features - mean

            cov = ( dif.T @ dif ) / (dif.size(0) - 1) 
            eps = 0.001
            cov_reg = cov + eps + torch.eye(cov.size(0))
            inv_cov = torch.linalg.pinv(cov_reg)

            

            # Pasamos estos valores a cuda
            self.mean_per_class = mean_per_class.cuda()
            self.mean = mean.cuda()
            self.inv_cov = inv_cov.cuda()

            self.setup_flag = True
        else:    
            pass
    
    # Metodo de postprocess. Se calcula la relative mahalanobis
    # distance de cada vector de features a la distribucion ID

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):


        net.eval()

        # Pasamos los datos a GPU
        data = data.cuda().float()

    
        # Calculo de logits y features
        logits, features = net.forward(data, return_feature=True)
        

        # Calculamos la prediccion a partir de los logits.
        pred = logits.argmax(dim=1)


        # Calculamos distancia de mahalanobis. Esta distancia es:
        # d(x) = (x - μ)^T Σ^{-1} (x - μ).
        # diff = (x - μ)
        # left = (x - μ)^T Σ^{-1}
        # scores = (x - μ)^T Σ^{-1} (x - μ)
        diff = features - self.mean                  
        left = diff @ self.inv_cov          
        md = (left * diff).sum(dim=1)

        # Calculamos la relative mahalanobis distance
        rmds = []
        for i in range(self.num_classes):

            # Distancia de mahalanobis de la muestra a cada clase
            diff = features - self.mean_per_class[i]                  
            left = diff @ self.inv_cov          
            md_per_class = ((left * diff).sum(dim=1))

            # Calculamos la RMD que es la MD a cada clase - MD global

            rmd = md_per_class - md

            rmds.append(rmd.unsqueeze(1))

        rmds = torch.cat(rmds, dim=1)

        # Nos quedamos con la menor

        scores = rmds.min(dim=1).values

        return pred, -scores

        


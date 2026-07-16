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
# Postprocessor que calcula la distribucion de features y vectores de XAI 
# CRP ID y calcula la relative mahalanobis distance a esta distribucion
# para devolver scores.
#----------------------------------------------------------------------
class RMDXAIPostProcessor(BasePostprocessor): 

    def __init__(self, config):
        super().__init__(config)
        self.setup_flag = False
        self.num_classes = self.config.num_classes

    # Metodo que define el estado inicial del postprocessor. Para ello,
    # tomaremos las caracteristicas de nuestro id_loader, extraeremos
    # el vector de explicaciones con CRP y calcularemos la media y la
    # covarianza de la distribucion ID, y la media de cada clase con el objetivo de 
    # realizar los scores a partir de esta distribución.
    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        
        # Si no se ha hecho setup del postprocessor
        if not self.setup_flag:

            # Activamos el modo evaluacion
            net.eval()


            # Elementos para el calculo del vector 
            # de explicaciones con CRP. Operamos
            # en la capa 4 de resnet50 que es la ultima 
            # con convoluciones.
            self.composite = EpsilonPlusFlat()
            self.attribution = CondAttribution(net)
            self.concept = ChannelConcept()

            # Lista de listas de features por clase

            class_features_xai = [[] for _ in range(self.num_classes)]           
            for batch in tqdm(id_loader_dict['train'],
                                desc='Setup: ', 
                                position=0, 
                                leave=True): 
                
                # Pasamos los datos a GPU y float
                data = batch['data'].cuda() 
                labels = batch['label'].cuda()
                data = data.float() 
                           
                with torch.no_grad():
                    # Extraemos logits y features
                    logits, feature = net(data, return_feature=True)
                
                    # Calculamos las predicciones de los logits, 
                    # necesario para el metodo CRP
                    pred = logits.argmax(dim=1)

                data = data.detach().requires_grad_(True)
                # condiciones: propagar relevancia respecto a la clase predicha
                conditions = [{"y": [p.item()]} for p in pred]

                # Aplicamos CRP
                attr = self.attribution(
                    data,
                    conditions=conditions,
                    composite=self.composite,
                    record_layer=["layer4"]
                )
                

                relevance = self.concept.attribute(attr.relevances["layer4"], abs_norm=True) 
                relevance = relevance.detach()

                # Calculamos el vector concatenado de feature y CRP
                feat_plus_xai = torch.cat([feature, relevance], dim=-1)

                # Introducimos cada vector en la lista de su clase
                for i in range(feat_plus_xai.shape[0]):
                    label = labels[i].item()
                    class_features_xai[label].append(feat_plus_xai[i].detach().cpu())

                del data
                del logits
                del feature
                del relevance 
                del feat_plus_xai 
                del attr
                torch.cuda.empty_cache()

            # La lista de cada clase se organiza en forma de batches y no ejemplos.
            # Con esto, hacemos un stack de todos los batches y pasamos a ejemplos.        
            class_features_xai = [
                torch.stack(c, dim=0) for c in class_features_xai
            ]

            # Extraemos todos los vectores
            all_features_xai = torch.cat(class_features_xai, dim=0)
            
            # Nuestro metodo esta basado en la distancia de mahalanobis.
            # Esta distancia es d(x) = (x - μ)^T Σ^{-1} (x - μ) donde:
            # - x es la muestra a calcular la distancia.
            # - μ es la media de las muestras ID.
            # - Σ es la covarianza entre las muestras ID.
            # Para calcular luego los scores calculamos estos parámetros
    

            # Calculamos la media de cada clase y media global
            mean_per_class = torch.stack(
                [c.mean(dim=0) for c in class_features_xai],
                dim=0
            )
            mean = all_features_xai.mean(dim=0)

             # Para el calculo de la inversa de la covarianza, se introduce
            # un término regulatorio eps en caso de ser singular
            #  y se calcula su pseusoinversa, por estabilidad numerica
            
            dif = all_features_xai - mean

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
    
    # Metodo de postprocess. A partir de data, extrae features y vectores XAI.
    # Despues los concatena y calcula la distancia relativa de mahalanobis respecto
    # a la distribucion de features + XAI de ID. Devuelve predicciones y score

    def postprocess(self, net: nn.Module, data: Any):


        net.eval()

        # Pasamos los datos a GPU
        data = data.cuda().float()

        with torch.no_grad():
            # Calculo de logits y features
            logits, features = net(data, return_feature=True)
            

            # Calculamos la prediccion a partir de los logits.
            pred = logits.argmax(dim=1)

        data = data.detach().requires_grad_(True)
        # condiciones: propagar relevancia respecto a la clase predicha
        conditions = [{"y": [p.item()]} for p in pred]

        # Calculamos el vector de XAI
        attr = self.attribution(
            data,
            conditions=conditions,
            composite=self.composite,
            record_layer=["layer4"]
        )
        
        relevance = self.concept.attribute(attr.relevances["layer4"], abs_norm=True) 
        relevance = relevance.detach()

        # Concatenamos features y XAI
        feat_plus_xai = torch.cat([features, relevance], dim=-1)

        # Calculamos distancia de mahalanobis. Esta distancia es:
        # d(x) = (x - μ)^T Σ^{-1} (x - μ).
        # diff = (x - μ)
        # left = (x - μ)^T Σ^{-1}
        # scores = (x - μ)^T Σ^{-1} (x - μ)
        diff = feat_plus_xai - self.mean                  
        left = diff @ self.inv_cov          
        md = (left * diff).sum(dim=1)

        # Calculamos la relative mahalanobis distance
        rmds = []
        for i in range(self.num_classes):

            # Distancia de mahalanobis de la muestra a cada clase
            diff = feat_plus_xai - self.mean_per_class[i]                  
            left = diff @ self.inv_cov          
            md_per_class = ((left * diff).sum(dim=1))

            # Calculamos la RMD que es la MD a cada clase - MD global
            rmd = md_per_class - md

            rmds.append(rmd.unsqueeze(1))

        rmds = torch.cat(rmds, dim=1)

        # Nos quedamos con la menor
        scores = rmds.min(dim=1).values

        return pred, -scores
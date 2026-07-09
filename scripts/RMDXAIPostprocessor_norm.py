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

class RMDXAIPostProcessor_norm(BasePostprocessor): 

    def __init__(self, config):
        super().__init__(config)
        self.setup_flag = False


    # Metodo que define el estado inicial del postprocessor. Para ello,
    # tomaremos las caracteristicas de nuestro id_loader, extraeremos
    # el vector de explicaciones con CRP y calcularemos la media y la
    # covarianza de la distribucion ID, con el objetivo de 
    # realizar los scores a partir de esta distribución.
    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        
        # Si no se ha hecho setup del postprocessor
        if not self.setup_flag:

            # Activamos el modo evaluacion
            net.eval()

            print('Extracting id training feature')


            # Elementos para el calculo del vector 
            # de explicaciones con CRP. Operamos
            # en la capa 4 de resnet50 que es la ultima 
            # con convoluciones.
            self.composite = EpsilonPlusFlat()
            self.attribution = CondAttribution(net)
            self.concept = ChannelConcept()

            class_features_xai = [[] for _ in range(14)]           
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
                feat_plus_xai = torch.cat([feature, relevance], dim=-1)

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
                # Guardamos el nuevo tensor

                #feature_xai_id_train.append(feat_plus_xai)
            
            # Concatenamos la lista de vectores de caracteristicas
            # para que no esten agrupadas por batch_size y tener un
            # vector de (N muestras, features.size() )

            #feature_xai_id_train = torch.cat(feature_xai_id_train, dim=0)

            #feat_xai_np = feature_xai_id_train.cpu().numpy()
            class_features_xai = [
                torch.stack(c, dim=0) for c in class_features_xai
            ]

            n_class=[
                len(c) for c in class_features_xai
            ]
            all_features_xai = torch.cat(class_features_xai, dim=0)
            all_features = all_features_xai[:, :2048]
            all_xai = all_features_xai[:, 2048:]

            self.max_vals_feat = all_features.abs().max(dim=0, keepdim=True).values
            features_norm = all_features / (self.max_vals_feat + 1e-8)

            self.max_vals_xai = all_xai.abs().max(dim=0, keepdim=True).values
            xai_norm = all_xai / (self.max_vals_xai + 1e-8)

            feature_xai_id_train = torch.cat([features_norm, xai_norm], dim=1)
            # Nuestro metodo esta basado en la distancia de mahalanobis.
            # Esta distancia es d(x) = (x - μ)^T Σ^{-1} (x - μ) donde:
            # - x es la muestra a calcular la distancia.
            # - μ es la media de las muestras ID.
            # - Σ es la covarianza entre las muestras ID.
            # Para calcular luego los scores calculamos estos parámetros
            
            start = 0
            mean_per_class = []
            for k in range(14):
                Nk = n_class[k]
                mean_per_class.append(feature_xai_id_train[start:start+Nk].mean(dim=0))
                start += Nk

            mean_per_class = torch.stack(mean_per_class, dim=0)

            #mean_per_class = torch.stack(
             #   [c.mean(dim=0) for c in class_features_xai],
              #  dim=0
            #)
            #mean = all_features_xai.mean(dim=0)

            mean = feature_xai_id_train.mean(dim=0)

             # Para el calculo de la inversa de la covarianza, se introduce
            # un término regulatorio eps en caso de ser singular
            #  y se calcula su pseusoinversa, por estabilidad numerica
            
            #dif = all_features_xai - mean
            dif = feature_xai_id_train - mean

            cov = ( dif.T @ dif ) / (dif.size(0) - 1) 
            eps = 0.01
            cov_reg = cov + eps + torch.eye(cov.size(0))
            inv_cov = torch.linalg.pinv(cov_reg)

            

            # Pasamos estos valores a cuda´
            self.max_vals_feat = self.max_vals_feat.cuda()
            self.max_vals_xai = self.max_vals_xai.cuda()            
            self.mean_per_class = mean_per_class.cuda()
            self.mean = mean.cuda()
            self.inv_cov = inv_cov.cuda()

            self.setup_flag = True
        else:    
            pass
    
    # Metodo de postprocess. A partir de data, extrae features y vectores XAI.
    # Despues los concatena y calcula la distancia de mahalanobis respecto
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
        features_norm = features / (self.max_vals_feat + 1e-8)

        xai_norm = relevance / (self.max_vals_xai + 1e-8)
        feat_plus_xai = torch.cat([features_norm, xai_norm], dim=-1)
        print(f"Dimension tensor {feat_plus_xai.shape}")

        # Calculamos distancia de mahalanobis. Esta distancia es:
        # d(x) = (x - μ)^T Σ^{-1} (x - μ).
        # diff = (x - μ)
        # left = (x - μ)^T Σ^{-1}
        # scores = (x - μ)^T Σ^{-1} (x - μ)
        diff = feat_plus_xai - self.mean                  
        left = diff @ self.inv_cov          
        md = (left * diff).sum(dim=1)

        rmds = []
        for i in range(14):
            diff = feat_plus_xai - self.mean_per_class[i]                  
            left = diff @ self.inv_cov          
            md_per_class = ((left * diff).sum(dim=1))

            rmd = md_per_class - md

            rmds.append(rmd.unsqueeze(1))

        rmds = torch.cat(rmds, dim=1)

        scores = rmds.min(dim=1).values

        return pred, -scores

        


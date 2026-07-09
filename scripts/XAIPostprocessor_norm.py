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

class XAIPostProcessor_norm(BasePostprocessor): 

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

            # Vector de caracteristicas y xai id
            feature_xai_id_train = []

            # Elementos para el calculo del vector 
            # de explicaciones con CRP. Operamos
            # en la capa 4 de resnet50 que es la ultima 
            # con convoluciones.
            self.composite = EpsilonPlusFlat()
            self.attribution = CondAttribution(net)
            self.concept = ChannelConcept()

            all_features = []
            all_xai = []
            for batch in tqdm(id_loader_dict['train'],
                                desc='Setup: ', 
                                position=0, 
                                leave=True): 
                
                # Pasamos los datos a GPU y float
                data = batch['data'].cuda() 
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
                
                all_features.append(feature.detach().cpu())

                relevance = self.concept.attribute(attr.relevances["layer4"], abs_norm=True) 
                relevance = relevance.detach()

                all_xai.append(relevance.detach().cpu())


                del logits
                del feature
                del relevance
                del attr
                torch.cuda.empty_cache()
            
            features = torch.cat(all_features, dim=0)
            self.max_vals_feat = features.abs().max(dim=0, keepdim=True).values
            features_norm = features / (self.max_vals_feat + 1e-8)

            xai = torch.cat(all_xai, dim=0)
            self.max_vals_xai = xai.abs().max(dim=0, keepdim=True).values
            xai_norm = xai / (self.max_vals_xai + 1e-8)



            # Concatenamos la lista de vectores de caracteristicas
            # para que no esten agrupadas por batch_size y tener un
            # vector de (N muestras, features.size() )
            feature_xai_id_train = torch.cat([features_norm, xai_norm], dim=1)

            feat_xai_np = feature_xai_id_train.numpy()
            # Nuestro metodo esta basado en la distancia de mahalanobis.
            # Esta distancia es d(x) = (x - μ)^T Σ^{-1} (x - μ) donde:
            # - x es la muestra a calcular la distancia.
            # - μ es la media de las muestras ID.
            # - Σ es la covarianza entre las muestras ID.
            # Para calcular luego los scores calculamos estos parámetros
            mean = feat_xai_np.mean(axis=0)
            cov = np.cov(feat_xai_np, rowvar=False)

            # Para el calculo de la inversa de la covarianza, se introduce
            # un término regulatorio eps en caso de ser singular
            #  y se calcula su pseusoinversa, por estabilidad numerica
            eps = 0.001
            cov_reg = cov + eps * np.eye(cov.shape[0])
            inv_cov = pinv(cov_reg)

            # Pasamos estos valores a torch
            self.max_vals_feat = self.max_vals_feat.cuda()
            self.max_vals_xai = self.max_vals_xai.cuda()
            self.mean = torch.from_numpy(mean).float().cuda()
            self.inv_cov = torch.from_numpy(inv_cov).float().cuda()

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
        
        features = features / (self.max_vals_feat + 1e-8)
        relevance = self.concept.attribute(attr.relevances["layer4"], abs_norm=True) 
        relevance.detach()

        relevance = relevance / (self.max_vals_xai + 1e-8)
        # Concatenamos features y XAI
        feat_plus_xai = torch.cat([features, relevance], dim=-1)
        print(f"Dimension tensor {feat_plus_xai.shape}")

        # Calculamos distancia de mahalanobis. Esta distancia es:
        # d(x) = (x - μ)^T Σ^{-1} (x - μ).
        # diff = (x - μ)
        # left = (x - μ)^T Σ^{-1}
        # scores = (x - μ)^T Σ^{-1} (x - μ)
        diff = feat_plus_xai - self.mean                  
        left = diff @ self.inv_cov          
        scores = (left * diff).sum(dim=1)

        return pred, -scores

        


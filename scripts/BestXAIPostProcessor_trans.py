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

# Postprocessor que selecciona un porcetnaje de mej
class BestXAIPostProcessor_trans(BasePostprocessor): 

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


            # Vector de caracteristicas 
            feature_id_train = []

            # Elementos para el calculo del vector 
            # de explicaciones con CRP. Operamos
            # en el bloque 11 de transformers que es el ultimo
            self.composite = EpsilonPlusFlat()
            self.attribution = CondAttribution(net)
            self.concept = ChannelConcept()

            for batch in tqdm(id_loader_dict['train'],
                                desc='Setup: ', 
                                position=0, 
                                leave=True): 
                
                # Pasamos los datos a GPU y float
                data = batch['data'].cuda() 
                data = data.float() 
                

                with torch.no_grad():
                    # Extraemos logits y features ( token de clase )
                    feature = net.forward_features(data)[:,0]
                    
                
                
                # Guardamos el nuevo tensor
                feature_id_train.append(
                    feature.detach().cpu()
                )

                del feature
                torch.cuda.empty_cache()
            
            # Concatenamos la lista de vectores de caracteristicas
            # para que no esten agrupadas por batch_size y tener un
            # vector de (N muestras, features.size() )
            feature_id_train = torch.cat(feature_id_train, dim=0)

            feat_np = feature_id_train.numpy()
            # Nuestro metodo esta basado en la distancia de mahalanobis.
            # Esta distancia es d(x) = (x - μ)^T Σ^{-1} (x - μ) donde:
            # - x es la muestra a calcular la distancia.
            # - μ es la media de las muestras ID.
            # - Σ es la covarianza entre las muestras ID.
            # Para calcular luego los scores calculamos estos parámetros
            mean = feat_np.mean(axis=0)
            cov = np.cov(feat_np, rowvar=False)

            # Para el calculo de la inversa de la covarianza, se introduce
            # un término regulatorio eps en caso de ser singular
            #  y se calcula su pseusoinversa, por estabilidad numerica
            eps = 0.001
            cov_reg = cov + eps * np.eye(cov.shape[0])
            inv_cov = pinv(cov_reg)

            # Pasamos estos valores a torch
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
            # Calculo de logits y features ( token de clase)
        
            features = net.forward_features(data)[:,0]
            logits = net.head(features)
                
            # Calculamos la prediccion a partir de los logits.
            pred = logits.argmax(dim=1)

        data = data.detach().requires_grad_(True)
        # condiciones: propagar relevancia respecto a la clase predicha
        conditions = [{"y": [p.item()]} for p in pred]

        # Calculamos el vector de XAI
        # Aplicamos CRP
        attr = self.attribution(
            data,
            conditions=conditions,
            composite=self.composite,
            record_layer=["blocks.11"]
        )
        
        relevance = attr.relevances["blocks.11"]

        # Seleccionamos el token de clase
        rel_cls = relevance[:, 0, :]
        relevance = relevance.detach()
        rel_cls = rel_cls.detach()

        # Obtenemos el 25% de mejores features en base a CRP
        best_xai = int(1 * features.shape[1])
        top_idx = torch.topk(rel_cls.abs(), k=best_xai, dim=1).indices
        # Concatenamos features y XAI
        mask = torch.zeros_like(features, dtype=torch.bool)

        mask.scatter_(1, top_idx, True)

        features_masked = features * mask

        # Calculamos distancia de mahalanobis. Esta distancia es:
        # d(x) = (x - μ)^T Σ^{-1} (x - μ).
        # diff = (x - μ)
        # left = (x - μ)^T Σ^{-1}
        # scores = (x - μ)^T Σ^{-1} (x - μ)
        diff = features_masked - self.mean                  
        left = diff @ self.inv_cov          
        scores = (left * diff).sum(dim=1)

        return pred, -scores
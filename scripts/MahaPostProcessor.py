import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from typing import Any
from numpy.linalg import pinv
from openood.postprocessors.base_postprocessor import BasePostprocessor

# -----------------------------------------------------------
# Postprocessor que calcula la distribucion de features ID y
# calcula los scores como la distancia de mahalanobis 
# a esta distribucion
# -----------------------------------------------------------

class MahaPostProcessor(BasePostprocessor): 

    def __init__(self, config):
        super().__init__(config)
        self.setup_flag = False


    # Metodo que define el estado inicial del postprocessor. Para ello,
    # tomaremos las caracteristicas de nuestro id_loader calcularemos la media y la
    # covarianza de la distribucion ID, con el objetivo de 
    # realizar los scores a partir de esta distribución.
    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        
        # Si no se ha hecho setup del postprocessor
        if not self.setup_flag:

            # Activamos el modo evaluacion
            net.eval()

            # Vector de caracteristicas y xai id
            feature_id_train = []

            

            for batch in tqdm(id_loader_dict['train'],
                                desc='Setup: ', 
                                position=0, 
                                leave=True): 
                
                # Pasamos los datos a GPU y float
                data = batch['data'].cuda() 
                data = data.float() 
                # Eliminamos el calculo de gradientes. Solo 
                # necesitamos extraer features

                with torch.no_grad():        

                    # Extraemos logits y features
                    _, feature = net(data, return_feature=True)


                # Guardamos el nuevo tensor
                feature_id_train.append(feature.detach().cpu())
        
                del feature
                torch.cuda.empty_cache()   

            # Concatenamos la lista de vectores de caracteristicas
            # para que no esten agrupadas por batch_size y tener un
            # vector de (N muestras, features.size() )
            feature_id_train = torch.cat(feature_id_train, dim=0)

            feat_xai_np = feature_id_train.numpy()
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
            self.mean = torch.from_numpy(mean).float().cuda()
            self.inv_cov = torch.from_numpy(inv_cov).float().cuda()

            self.setup_flag = True
        else:    
            pass
    
    # Metodo de postprocess. A partir de data, extrae features .
    # Despues calcula la distancia de mahalanobis respecto
    # a la distribucion de features de ID. Devuelve predicciones y score
    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):


        net.eval()

        # Pasamos los datos a GPU
        data = data.cuda().float()

        # Calculo de logits y features
        logits, features = net(data, return_feature=True)
        
        # Calculamos la prediccion a partir de los logits.
        pred = logits.argmax(dim=1)
       
        # Calculamos distancia de mahalanobis. Esta distancia es:
        # d(x) = (x - μ)^T Σ^{-1} (x - μ).
        # diff = (x - μ)
        # left = (x - μ)^T Σ^{-1}
        # scores = (x - μ)^T Σ^{-1} (x - μ)
        diff = features - self.mean                  
        left = diff @ self.inv_cov          
        scores = (left * diff).sum(dim=1)

        return pred, -scores

        


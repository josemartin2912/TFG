class XAIPostProcessor(BasePostprocessor): 

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

            # Eliminamos el calculo de gradientes. Solo 
            # necesitamos extraer features
            with torch.no_grad():
                print('Extracting id training feature')

                # Vector de caracteristicas y xai id
                feature_xai_id_train = []

                # Elementos para el calculo del vector 
                # de explicaciones con CRP. Operamos
                # en la capa 4 de resnet50 que es la ultima 
                # con convoluciones.
                composite = EpsilonPlusFlat()
                attribution = CondAttribution(net)
                concept = ChannelConcept(net.layer4)

                for batch in tqdm(id_loader_dict['train'],
                                   desc='Setup: ', 
                                   position=0, 
                                   leave=True): 
                    
                    # Pasamos los datos a GPU y float
                    data = batch['data'].cuda() 
                    data = data.float() 

                    # Extraemos logits y features
                    logits, feature = net(data, return_feature=True)
                    
                    # Calculamos las predicciones de los logits, 
                    # necesario para el metodo CRP
                    pred = logits.argmax(dim=1)

                    # Aplicamos CRP
                    relevance = attribution(
                        data,
                        condition=pred,
                        composite=composite,
                        concept=concept
                    )
                    relevance = relevance.cpu()

                    feat_plus_xai = torch.cat([feature, relevance], dim=-1)

                    # Pasamos el tensor a numpy y CPU y guradamos en 
                    # la lista de vectores de caracteristicas
                    feature_xai_id_train.append(feat_plus_xai.cpu().numpy())
                
                # Concatenamos la lista de vectores de caracteristicas
                # para que no esten agrupadas por batch_size y tener un
                # vector de (N muestras, features.size() )
                feature_xai_id_train = np.concatenate(feature_xai_id_train, axis=0)

                
                # Nuestro metodo esta basado en la distancia de mahalanobis.
                # Esta distancia es d(x) = (x - μ)^T Σ^{-1} (x - μ) donde:
                # - x es la muestra a calcular la distancia.
                # - μ es la media de las muestras ID.
                # - Σ es la covarianza entre las muestras ID.
                # Para calcular luego los scores calculamos estos parámetros
                self.mean = feature_xai_id_train.mean(axis=0)
                self.cov = np.cov(feature_xai_id_train, rowvar=False)

                # Para el calculo de la inversa de la covarianza, se introduce
                # un término regulatorio eps en caso de ser singular
                #  y se calcula su pseusoinversa, por estabilidad numerica
                eps = 0.001
                cov_reg = self.cov + eps * np.eye(cov.shape[0])
                self.inv_cov = np.linalg.pinv(cov_reg)

                # Pasamos estos valores a torch
                self.mean = torch.from_numpy(self.mean).float().cuda()
                self.inv_cov = torch.from_numpy(self.inv_cov).float().cuda()
        else:    
            pass
    
    # Metodo de postprocess. A partir de data, extrae features y vectores XAI.
    # Despues los concatena y calcula la distancia de mahalanobis respecto
    # a la distribucion de features + XAI de ID. Devuelve predicciones y score
    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):


        net.eval()
        # Calculo de logits y features
        logits, features = net.forward(data, return_feature=True)
        features = features.cpu()

        # Elementos para el calculo del vector 
        # de explicaciones con CRP. Operamos
        # en la capa 4 de resnet50 que es la ultima 
        # con convoluciones.
        composite = EpsilonPlusFlat()
        attribution = CondAttribution(net)
        concept = ChannelConcept(net.layer4)

        
        # Pasamos los datos a GPU
        data = data.cuda()

        # Calculamos la prediccion a partir de los logits.
        pred = logits.argmax(dim=1)

        # Calculamos el vector de XAI
        relevance = attribution(
            data,
            condition=pred,
            composite=composite,
            concept=concept
        )
        relevance = relevance.cpu()

        # Concatenamos features y XAI
        feat_plus_xai = torch.cat([features, relevance], dim=-1)

        # Calculamos distancia de mahalanobis. Esta distancia es:
        # d(x) = (x - μ)^T Σ^{-1} (x - μ).
        # diff = (x - μ)
        # left = (x - μ)^T Σ^{-1}
        # scores = (x - μ)^T Σ^{-1} (x - μ)
        diff = feat_plus_xai - self.mean                  
        left = diff @ self.inv_cov          
        scores = (left * diff).sum(dim=1)

        return pred, torch.from_numpy(scores)

        


# Inteligencia Artificial explicable en medicina: Caso de estudio en cáncer de próstata.
## Resumen/Abstract

Prostate cancer is one of the most prevalent malignancies in men. This work addresses the problem
through the use of Deep Learning models in order to classify prostatic biopsy samples. However,
the main goal is to develop Out-of-Distribution detection methods based on eXplainable AI. In
order to accomplish this task, this work proposes the use of DeiT and ResNet-50 models with a
set of postprocessors based on the Mahalanobis distance and eXplainable AI vectors generated
by the CRP algorithm. Results show that methods employing a low contribution of the CRP
vector outperform state-of-the-art methods such as ViM and MSP. In contrast, incrementing the
contribution of the CRP vectors yields performance that matches the state-of-the-art, resulting in
an effective discriminator of Out-of-Distribution samples. Next, it is employed a smaller amount
of the total features, which resulted in a significant separator for DeiT, saving up to 75% of
total features. Lastly, it is worth noting that the use of all features provided by DeiT for Out-of
Distribution detection, results in perfect separation between both datasets.

El cáncer de próstata es una enfermedad que afecta a gran parte de la población masculina. Este
trabajo aborda el problema mediante el uso de modelos de Deep Learning para clasificar muestras de
biopsias prostáticas. No obstante, el principal objetivo es el desarrollo de métodos de detección Out
of-Distribution empleando vectores de IA explicable. Para ello, se han utilizado los modelos DeiT
y ResNet-50 y una serie de posprocesadores basados en la distancia de Mahalanobis y vectores del
algoritmo de IA explicable CRP. Los resultados muestran que los métodos propuestos que emplean
una contribución reducida del vector CRP superan a métodos del estado del arte como ViM y MSP.
Por el contrario, cuando se incrementa la contribución del vector CRP, el rendimiento obtenido es
comparable al de los métodos del estado del arte, mostrando resultados de un separador razonable.
A continuación, se propone utilizar una cantidad menor de las features totales, que en el modelo
DeiT supuso la obtención de un separador destacable, ahorrando el 75% de features. Por último,
cabe destacar que el uso de la totalidad de features del modelo DeiT para la detección OOD, resulta
en una separación perfecta de ambos conjuntos de datos.

[Memoria completa](TFG_Jose_Maria_Martin.pdf)

## Configs

El directorio [configs](configs) contiene los archivos en formato yml de configuración para los scripts basados en OOD. Estos ficheros contendrán rutas a nuestro dataset, modelo a usar y checkpoints, preprocessors, postprocessors..., etc.
## Scripts
El directorio [scripts](scripts) contiene los scripts de entrenamiento, evaluación OOD, la implementación de nuestros PostProcessors y scripts auxiliares como generar un archivo CSV de métricas a partir de un archivo de salida o generar la estimación de la función de densidad utilizando la técnica KDE sobre distribuciones generadas por postprocessors para ver la separación que aporta dicho postprocessor. De forma muy breve, los postprocessors realizan lo siguiente  ( Consultar cada archivo y sobre todo la [Memoria completa](TFG_Jose_Maria_Martin.pdf) para obtener una descripción más detallada ):

-  [MahaPostProcessor](scripts/MahaPostProcessor.py): Se calcula la distribución $(\mu,\Sigma)$ de las features de ejemplos ID. Se devuelve como score ID la distancia de Mahalanobis a la distribución, es decir, $\forall x \in ID \cup OOD \quad d(x) = (x-\mu)^T\Sigma^{-1}(x-\mu)$. Se devuelve este valor en negativo por una razón. OpenOOD asigna como ID a los scores más altos. Por tanto, como calculamos una distancia, menos distancia implica mayor probabilidad ID. Entonces mayor **-distancia** implica mayor probabilidad de ID.
-  [XAIPostProcessor](scripts/XAIPostProcessor.py): El proceso es exactamente igual que el anterior salvo que la distribución no es únicamente de features sino de vectores formados por la **concatenación de feature y vector de XAI CRP**.
-  [XAIPostProcessor versión normalizada](scripts/XAIPostProcessor_norm.py): Igual que el caso anterior con la salvedad de que se normaliza cada variable al intervalo [0,1] para que los valores de las features no sean demasiado superiores a los de CRP.
-  [RMDPostProcessor](scripts/RMDPostProcessor.py): Similar a [MahaPostProcessor](scripts/MahaPostProcessor.py) pero en este caso se calcula la **Relative Mahalanobis Distance (RMD)**. Este postprocessor calcula la distribución de features ID global y también la distribución de cada clase. La RMD es el mínimo de $d_k(x) - d_{global}(x) \quad k = 1,..,num classes$. De nuevo, se devuelve el valor negativo.
-  [RMDXAIPostProcessor](scripts/RMDXAIPostProcessor.py): Proceso igual al anterior pero la distribución es de **concatenación de feature y vector de XAI CRP** .
-  [BestXAIPostProcessor](scripts/BestXAIPostProcessor.py) y [BestXAIPostProcessor versón transformers](scripts/BestXAIPostProcessor_trans.py): Se calcula la distribución $(\mu,\Sigma)$ de las features de ejemplos ID ( En transformers las features del token de clase ). En el cálculo de los scores, para cada muestra $x$ se calcula su vector CRP. Se calcula un porcentaje, pasado como parámetro $p%$, de los mayores valores en valor absoluto y se guardan los índices de estos mejores valores. Se extraen las features del ejemplo $x$ y se calcula un nuevo vector donde nuevas\_features $[i]$ = features $[i]$ si $i$ está en el conjunto de los índices comentados previamente y 0 en caso contrario. Se calculan los nuevos scores como la distancia de mahalanobis a la distribución ID con los nuevos vectores modificados de features, de modo que se inutiliza el uso de las features menos importantes de acuerdo a CRP, fijando su valor a 0.


## Results
El directorio [results](results) contiene los resultados de nuestros experimentos. Por lo general, el modelo usado es **ResNet-50** salvo en [results/DeiT](results/DeiT) que se usa el modelo **DeiT**. En cuanto a los contenidos:
-  [results](results): Contiene los resultados de entrenamiento de **ResNet-50** ([comparacion de funcion de perdida en train vs val](results/loss_comp.png) y [accuracy en validacion](results/accuracy.png)) y la evaluación OOD para métodos del estado del arte **MSP y  ViM** y la primera propuesta [XAIPostProcessor](scripts/XAIPostProcessor.py).
-  [maha](results/maha): Contiene los resultados de [MahaPostProcessor](scripts/MahaPostProcessor.py).
-  [rmd](results/rmd): Contiene los resultados de [RMDPostProcessor](scripts/RMDPostProcessor.py).
-  [rmdxai](results/rmdxai): Contiene los resultados de [RMDXAIPostProcessor](scripts/RMDXAIPostProcessor.py).
-  [xai](results/xai): Contiene los resultados de [XAIPostProcessor](scripts/XAIPostProcessor.py), [XAIPostProcessor versión normalizada](scripts/XAIPostProcessor_norm.py) y [BestXAIPostProcessor](scripts/BestXAIPostProcessor.py).
-  [DeiT](results/DeiT): Contiene los resultados de entrenamiento de **DeiT** ([comparacion de funcion de perdida en train vs val](results/DeiT/loss_comp.png) y [accuracy en validacion](results/DeiT/accuracy.png)) y la evaluación OOD para el postprocessor [BestXAIPostProcessor (versión transformers)](scripts/BestXAIPostProcessor_trans.py).

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

##configs

La carpeta [configs](configs) contiene los archivos en formato yml de configuración para los scripts basados en OOD. Estos ficheros contendrán rutas a nuestro dataset, modelo a usar y checkpoints, preprocessors, postprocessors..., etc.

##results



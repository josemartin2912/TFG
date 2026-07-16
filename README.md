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

El c´ancer de pr´ostata es una enfermedad que afecta a gran parte de la poblaci´on masculina. Este
trabajo aborda el problema mediante el uso de modelos de Deep Learning para clasificar muestras de
biopsias prost´aticas. No obstante, el principal objetivo es el desarrollo de m´etodos de detecci´on Out
of-Distribution empleando vectores de IA explicable. Para ello, se han utilizado los modelos DeiT
y ResNet-50 y una serie de posprocesadores basados en la distancia de Mahalanobis y vectores del
algoritmo de IA explicable CRP. Los resultados muestran que los m´etodos propuestos que emplean
una contribuci´on reducida del vector CRP superan a m´etodos del estado del arte como ViM y MSP.
Por el contrario, cuando se incrementa la contribuci´on del vector CRP, el rendimiento obtenido es
comparable al de los m´etodos del estado del arte, mostrando resultados de un separador razonable.
A continuaci´on, se propone utilizar una cantidad menor de las features totales, que en el modelo
DeiT supuso la obtenci´on de un separador destacable, ahorrando el 75% de features. Por ´ultimo,
cabe destacar que el uso de la totalidad de features del modelo DeiT para la detecci´on OOD, resulta
en una separaci´on perfecta de ambos conjuntos de datos.

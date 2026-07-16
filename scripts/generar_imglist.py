import os
import argparse

# Script para asignar un formato imglist (formato de OpenOOD) a
# nuestro dataset. Nuestro dataset esta en formato:
#   train:
#           - clase1
#                   - 1.png
#                   - 2.png
#           - clase2
#                   - 3.png
#                   - 4.png
#
# Genera un fichero .txt con entradas:
#  
# clase1/1.png 1
# clase1/2.png 1
# clase2/3.png 2
# clase2/4.png 2
#
# De este modo, se le indica a OpenOOD la clase de cada ejemplo y donde se encuentra.

# Argument parser. Procesa los argumentos de entrada que son el
# directorio de datos a procesar y el nombre del fichero de salida.
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, required=True)
parser.add_argument('--output_txt', type=str, required=True)
args = parser.parse_args()

# Se toman todas las clases (subcarpetas del directorio argumento)
# y se asigna un indice numerico a cada clase
classes = sorted(os.listdir(args.data_dir))
class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

print("Clases encontradas:")
for cls, idx in class_to_idx.items():
    print(f"  {idx}: {cls}")

# Se escribe en el fichero de salida
with open(args.output_txt, 'w') as f:

    # Para cada clase, tomamos la ruta absoluta
    # que es la ruta al directorio argumento mas
    # la clase
    for cls in classes:
        cls_dir = os.path.join(args.data_dir, cls)
        if not os.path.isdir(cls_dir):
            continue

        # Para cada imagen, se escribe en el fichero de salida:
        # clase/imagen.png  indice numerico de clase
        for img in sorted(os.listdir(cls_dir)):
            if img.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
                f.write(f'{cls}/{img} {class_to_idx[cls]}\n')

print(f"\nArchivo generado: {args.output_txt}")
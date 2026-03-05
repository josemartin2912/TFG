from pathlib import Path
from PIL import Image
from collections import Counter

print('Inicio')
path = Path('/mnt/homeGPU/isevillano/Github/GleasonMap/Data/TrainTest/train')

shapes_counter = Counter()

print('Inicio bucle')
# Recorre cada clase
for class_folder in path.iterdir():
    if class_folder.is_dir():
        print('Carpeta')
        for img_path in class_folder.glob("*.*"):
            try:
                with Image.open(img_path) as img:
                    # shape = (width, height, channels)
                    shape = (img.width, img.height, len(img.getbands()))
                    shapes_counter[shape] += 1
            except Exception as e:
                print(f"Error leyendo {img_path}: {e}")

# Imprime los shapes diferentes y cuántas imágenes tienen
print("Shapes distintos en el dataset:")
for shape, count in shapes_counter.items():
    print(f"{shape}: {count} imágenes")
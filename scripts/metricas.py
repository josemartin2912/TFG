import re
import csv

# -----------------------------------------------------------
# Script que procesa los ficheros .out generados por OpenOOD
# y guarda las metricas de entrenamiento en cada epoca en un archivo
# csv.
# -----------------------------------------------------------

# Expresion regular de una linea de resultados en el archivo de metricas

pattern = r'Epoch (\d+) \| Time\s+(\d+)s \| Train Loss ([\d.]+) \| Val Loss ([\d.]+) \| Val Acc ([\d.]+)'

metrics = []

# Comprueba cada linea del fichero .out y si coincide con la expresion
# regular, extrae las metricas
with open('/mnt/homeGPU/jmartin/TFG/scripts/train_deit_pre.out', 'r') as f:
    for line in f:
        match = re.search(pattern, line)
        if match:
            metrics.append({
                'epoch': int(match.group(1)),
                'time': int(match.group(2)),
                'train_loss': float(match.group(3)),
                'val_loss': float(match.group(4)),
                'val_acc': float(match.group(5))
            })

# El bloque esta duplicado porque en el caso de DeiT el entrenamiento se interrumpio
# en la epoca 9 y por tanto hubo dos ficheros .out porque hubo dos entrenamientos.
with open('/mnt/homeGPU/jmartin/TFG/scripts/train_deit.out', 'r') as f:
    for line in f:
        match = re.search(pattern, line)
        if match:
            metrics.append({
                'epoch': int(match.group(1)) + 9,
                'time': int(match.group(2)),
                'train_loss': float(match.group(3)),
                'val_loss': float(match.group(4)),
                'val_acc': float(match.group(5))
            })

# Se guardan las metricas en el csv
with open('/mnt/homeGPU/jmartin/TFG/results/DeiT/metricas.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['epoch', 'time', 'train_loss', 'val_loss', 'val_acc'])
    writer.writeheader()
    writer.writerows(metrics)

print(f"Extraídas {len(metrics)} épocas → /mnt/homeGPU/jmartin/TFG/results/DeiT/metricas.csv")
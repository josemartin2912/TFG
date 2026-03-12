import re
import csv



pattern = r'Epoch (\d+) \| Time\s+(\d+)s \| Train Loss ([\d.]+) \| Val Loss ([\d.]+) \| Val Acc ([\d.]+)'

metrics = []
with open('/mnt/homeGPU/jmartin/TFG/scripts/train_resnet50.out', 'r') as f:
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

with open('/mnt/homeGPU/jmartin/TFG/results/metricas.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['epoch', 'time', 'train_loss', 'val_loss', 'val_acc'])
    writer.writeheader()
    writer.writerows(metrics)

print(f"Extraídas {len(metrics)} épocas → /mnt/homeGPU/jmartin/TFG/results/metricas.csv")
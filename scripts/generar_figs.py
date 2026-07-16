import pandas as pd
import matplotlib.pyplot as plt

# Script que se encarga de generar graficas de comparacion
# de funcion de perdida en train y validacion y accuracy en validacion
# a partir de un csv de metricas.

df = pd.read_csv('/mnt/homeGPU/jmartin/TFG/results/DeiT/metricas.csv')

# 1. Train Loss vs Val Loss
plt.figure(figsize=(8, 5))
plt.plot(df['epoch'], df['train_loss'], label='Train Loss', color='blue')
plt.plot(df['epoch'], df['val_loss'], label='Val Loss', color='orange')
plt.title('Train Loss vs Val Loss')
plt.xlabel('Época')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f'/mnt/homeGPU/jmartin/TFG/results/DeiT/loss_comp.png', dpi=150)
plt.close()


# 2. Val Accuracy
plt.figure(figsize=(8, 5))
plt.plot(df['epoch'], df['val_acc'], label='Val Accuracy', color='green')
plt.title('Accuracy en Validación')
plt.xlabel('Época')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('/mnt/homeGPU/jmartin/TFG/results/DeiT/accuracy.png', dpi=150)
plt.close()

import matplotlib.pyplot as plt

# -----------------------------------------------------------
# Script que calcula graficas de las 4 metricas OOD 
# para diferentes valores de p% de BestXAIPostProcessor
# -----------------------------------------------------------

# Nombres de metricas
metric_names = ["FPR@95", "AUROC", "AUPR_IN", "AUPR_OUT"]

# Resultados para cada valor de p%
results = {
    "25%": [5.05, 98.02, 99.82, 70.43],
    "50%": [6.39, 96.66, 99.77, 48.44],
    "75%": [1.94, 99.09, 99.94, 77.09],
    "100%": [0.0, 100.0, 100.0, 100.0],
}

# Plot y guardar imagen en resultados
plt.figure(figsize=(9, 5))

for label, values in results.items():
    line, = plt.plot(metric_names, values, marker="o", linewidth=2, label=label)

    color = line.get_color()

    for i, y in enumerate(values):
        plt.annotate(
            f"{y:.2f}",
            (metric_names[i], y),
            textcoords="offset points",
            xytext=(0, 8),   # desplazamiento vertical en píxeles
            ha="center",
            fontsize=9,
            color=color
        )

plt.ylim(0, 105)
plt.ylabel("Valor")
plt.xlabel("Métrica")
plt.title(f"DeiT % best features. Comparación de métricas OOD")
plt.grid(True, alpha=0.3)
plt.legend(title=f"% best features")

plt.tight_layout()
plt.savefig(
    "/mnt/homeGPU/jmartin/TFG/results/DeiT/metricas_covid.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()
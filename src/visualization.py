# src/visualization.py
import matplotlib.pyplot as plt
import os

def save_plot(fig, file_name, figures_dir):
    """Guardar la figura en carpeta especificada"""
    os.makedirs(figures_dir, exist_ok=True)
    path = os.path.join(figures_dir, file_name)
    fig.savefig(path)
    print(f"Figura guardada en {path}")
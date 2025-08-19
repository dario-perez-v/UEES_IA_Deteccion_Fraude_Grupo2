# src/utils.py
import pandas as pd
import os

def load_data(file_path):
    """Cargar dataset desde CSV"""
    return pd.read_csv(file_path)

def save_model(model, file_name, models_dir):
    """Guardar modelo con joblib"""
    import joblib
    os.makedirs(models_dir, exist_ok=True)
    path = os.path.join(models_dir, file_name)
    joblib.dump(model, path)
    print(f"Modelo guardado en {path}")
    return path

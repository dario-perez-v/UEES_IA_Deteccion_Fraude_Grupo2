import pandas as pd
import numpy as np
import os
import joblib
import streamlit as st

# --------------------
# Directorios
# --------------------
project_path = os.path.dirname(os.path.abspath(__file__))  # ruta donde está app.py
data_processed = os.path.join(project_path, '..', 'data', 'processed')
models_dir = os.path.join(project_path, '..', 'models', 'trained_models')

# --------------------
# Streamlit: subir CSV
# --------------------
st.title("Dashboard - Detección de Fraude")

uploaded_file = st.file_uploader("Sube tu archivo CSV", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    # Si no sube nada, carga el CSV de ejemplo
    default_csv = os.path.join(data_processed, "creditcard_clean.csv")
    df = pd.read_csv(default_csv)

# Crear columna Amount_log si no existe
if 'Amount_log' not in df.columns and 'Amount' in df.columns:
    df['Amount_log'] = np.log1p(df['Amount'])

# --------------------
# Cargar modelos disponibles
# --------------------
models = {}
for file in os.listdir(models_dir):
    if file.endswith(".pkl"):
        model_name = os.path.splitext(file)[0]
        model_path = os.path.join(models_dir, file)
        models[model_name] = joblib.load(model_path)

st.write("Modelos cargados:", list(models.keys()))

# --------------------
# Columnas usadas en entrenamiento
# --------------------
features_model = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8',
                  'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16',
                  'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24',
                  'V25', 'V26', 'V27', 'V28', 'Amount', 'Amount_log']

X = df[[col for col in features_model if col in df.columns]]

# --------------------
# Predicciones
# --------------------
for name, model in models.items():
    df[name + "_prob"] = model.predict_proba(X)[:,1]

# --------------------
# Mostrar resultados
# --------------------
st.write("Primeras filas del dataset:")
st.dataframe(df.head())

st.write("Distribución de probabilidades de fraude:")
prob_cols = [col for col in df.columns if col.endswith("_prob")]
if prob_cols:
    st.bar_chart(df[prob_cols])
else:
    st.write("No hay probabilidades calculadas aún.")


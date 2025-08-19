import pandas as pd
import numpy as np
import os
import joblib
import streamlit as st

# --------------------
# Directorios
# --------------------
project_path = '/content/drive/MyDrive/Proyecto-IA-DeteccionDeFraudeEnTransacciones'
data_processed = os.path.join(project_path, 'data', 'processed')
models_dir = os.path.join(project_path, 'models', 'trained_models')

# --------------------
# Cargar data
# --------------------
df = pd.read_csv(os.path.join(data_processed, "creditcard_clean.csv"))

# Crear columna Amount_log si no existe
if 'Amount_log' not in df.columns:
    df['Amount_log'] = np.log1p(df['Amount'])

# --------------------
# Cargar modelos
# --------------------
rf_model = joblib.load(os.path.join(models_dir, "rf_model.pkl"))
xgb_model = joblib.load(os.path.join(models_dir, "xgb_model.pkl"))

# --------------------
# Columnas usadas en entrenamiento
# --------------------
features_model = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8',
                  'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16',
                  'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24',
                  'V25', 'V26', 'V27', 'V28', 'Amount', 'Amount_log']

X = df[features_model]

# --------------------
# Predicciones
# --------------------
df['RF_prob'] = rf_model.predict_proba(X)[:,1]
df['XGB_prob'] = xgb_model.predict_proba(X)[:,1]

# --------------------
# Streamlit interface
# --------------------
st.title("Dashboard - Detección de Fraude")

st.write("Primeras filas del dataset:")
st.dataframe(df.head())

st.write("Distribución de probabilidades de fraude:")
st.bar_chart(df[['RF_prob', 'XGB_prob']])

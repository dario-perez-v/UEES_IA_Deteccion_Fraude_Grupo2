import pandas as pd
import numpy as np
import os
import joblib
import streamlit as st

# --------------------
# Directorios
# --------------------
project_path = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(project_path, '../models/trained_models')

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

# --------------------
# Streamlit interface
# --------------------
st.title("Dashboard - Detección de Fraude")

uploaded_file = st.file_uploader("Sube tu archivo CSV", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Crear columna Amount_log si no existe
    if 'Amount_log' not in df.columns:
        df['Amount_log'] = np.log1p(df['Amount'])

    # Seleccionar columnas para predicción
    X = df[features_model]

    # Predicciones
    df['RF_prob'] = rf_model.predict_proba(X)[:,1]
    df['XGB_prob'] = xgb_model.predict_proba(X)[:,1]

    # Mostrar datos
    st.write("Primeras filas del dataset:")
    st.dataframe(df.head())

    st.write("Distribución de probabilidades de fraude:")
    st.bar_chart(df[['RF_prob', 'XGB_prob']])
else:
    st.info("Por favor sube un archivo CSV para continuar.")

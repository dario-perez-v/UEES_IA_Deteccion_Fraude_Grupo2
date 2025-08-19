import pandas as pd
import numpy as np
import os
import joblib
import streamlit as st

# --------------------
# Directorios
# --------------------
project_path = '/content/drive/MyDrive/Proyecto-IA-DeteccionDeFraudeEnTransacciones'
models_dir = os.path.join(project_path, 'models', 'trained_models')

# --------------------
# Cargar modelos
# --------------------
rf_model = joblib.load(os.path.join(models_dir, "rf_model.pkl"))
xgb_model = joblib.load(os.path.join(models_dir, "xgb_model.pkl"))

# --------------------
# Streamlit interface
# --------------------
st.title("Dashboard - Detección de Fraude")

# Subir archivo CSV
uploaded_file = st.file_uploader("Sube tu archivo CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Crear columna Amount_log si no existe
    if 'Amount_log' not in df.columns and 'Amount' in df.columns:
        df['Amount_log'] = np.log1p(df['Amount'])

    # Columnas usadas en entrenamiento (verificamos que existan en el CSV)
    features_model = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8',
                      'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16',
                      'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24',
                      'V25', 'V26', 'V27', 'V28', 'Amount', 'Amount_log']

    features_model = [f for f in features_model if f in df.columns]
    X = df[features_model]

    # Predicciones
    df['RF_prob'] = rf_model.predict_proba(X)[:,1]
    df['XGB_prob'] = xgb_model.predict_proba(X)[:,1]

    # Mostrar tabla
    st.write("Primeras filas del dataset con probabilidades:")
    st.dataframe(df.head())

    # Distribución de probabilidades
    st.write("Distribución de probabilidades de fraude:")
    st.bar_chart(df[['RF_prob', 'XGB_prob']])

else:
    st.warning("Por favor sube un archivo CSV para continuar")

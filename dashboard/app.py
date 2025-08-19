import pandas as pd
import numpy as np
import os
import joblib
import streamlit as st
import shap
import matplotlib.pyplot as plt

# --------------------
# Directorios
# --------------------
project_path = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(project_path, '../models/trained_models')

# --------------------
# Cargar modelos
# --------------------
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
st.title("Dashboard - Detección de Fraude con XGBoost y SHAP")

uploaded_file = st.file_uploader("Sube tu archivo CSV", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Crear columna Amount_log si no existe
    if 'Amount_log' not in df.columns:
        df['Amount_log'] = np.log1p(df['Amount'])

    # Seleccionar columnas para predicción
    X = df[features_model]

    # Predicciones XGBoost
    df['XGB_prob'] = xgb_model.predict_proba(X)[:,1]

    # Muestreo para visualización rápida
    sample_df = df.sample(min(5000, len(df)))

    # Mostrar primeras filas del dataset
    st.write("Primeras filas del dataset:")
    st.dataframe(df.head())

    # Mostrar distribución de probabilidades
    st.write("Distribución de probabilidades de fraude (XGBoost):")
    st.bar_chart(sample_df[['XGB_prob']])

    # --------------------
    # SHAP explainability
    # --------------------
    st.write("Explicabilidad del modelo XGBoost con SHAP:")
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(sample_df[features_model])

    fig, ax = plt.subplots(figsize=(10,6))
    shap.summary_plot(shap_values, sample_df[features_model], show=False)
    fig = plt.gcf()
    st.pyplot(fig)
    plt.close(fig)

else:
    st.info("Por favor sube un archivo CSV para continuar.")



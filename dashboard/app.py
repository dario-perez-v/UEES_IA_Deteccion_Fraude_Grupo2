import pandas as pd
import numpy as np
import os
import joblib
import streamlit as st
import altair as alt

# --------------------
# Directorios
# --------------------
project_path = '/content/drive/MyDrive/Proyecto-IA-DeteccionDeFraudeEnTransacciones'
models_dir = os.path.join(project_path, 'models', 'trained_models')

# --------------------
# Streamlit interface
# --------------------
st.title("Dashboard - Detección de Fraude")
st.write("Carga tu archivo CSV para analizar:")

uploaded_file = st.file_uploader("Elige un archivo CSV", type="csv")

if uploaded_file is not None:
    # --------------------
    # Cargar data
    # --------------------
    df = pd.read_csv(uploaded_file)

    # Crear columna Amount_log si no existe
    if 'Amount_log' not in df.columns and 'Amount' in df.columns:
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

    # --------------------
    # Predicciones
    # --------------------
    X = df[features_model]
    df['RF_prob'] = rf_model.predict_proba(X)[:,1]
    df['XGB_prob'] = xgb_model.predict_proba(X)[:,1]

    # --------------------
    # Mostrar primeras filas
    # --------------------
    st.write("Primeras filas del dataset:")
    st.dataframe(df.head())

    # --------------------
    # Distribución de probabilidades
    # --------------------
    # Muestrear hasta 1000 filas
    df_sample = df[['RF_prob', 'XGB_prob']].sample(min(1000, len(df)))

    # Transformar a formato long
    df_long = df_sample.reset_index().melt(id_vars='index', value_vars=['RF_prob', 'XGB_prob'],
                                          var_name='Modelo', value_name='Probabilidad')

    chart = alt.Chart(df_long).mark_bar(opacity=0.7).encode(
        x=alt.X('Modelo:N', title='Modelo'),
        y=alt.Y('Probabilidad:Q', title='Probabilidad de fraude'),
        color='Modelo:N'
    ).properties(
        width=600,
        height=400,
        title='Distribución de probabilidades de fraude'
    )

    st.altair_chart(chart)

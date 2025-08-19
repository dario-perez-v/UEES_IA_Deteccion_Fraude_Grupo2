import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os
import sys

# Añadir src al path para importar funciones
sys.path.append('../src')
from features.feature_engineering import add_features
from scoring.realtime import score_transaction
from interpretability.shap_analysis import explain_model

# Directorios
models_dir = "../models/trained_models/"
data_dir = "../data/processed/"

# Título
st.title("Dashboard - Detección de Fraude en Transacciones")

# Subida de archivo de transacciones
uploaded_file = st.file_uploader("Sube un CSV de transacciones", type=["csv"])
if uploaded_file:
    df_new = pd.read_csv(uploaded_file)
    df_new = add_features(df_new)
    
    # Cargar modelo
    rf_model = joblib.load(os.path.join(models_dir, "rf_model.pkl"))
    
    # Scoring
    df_new['Fraud_Prob'] = score_transaction(df_new, rf_model)
    
    st.write("Predicciones de fraude:")
    st.dataframe(df_new)
    
    # SHAP explicabilidad
    st.subheader("SHAP Summary Plot")
    explain_model(rf_model, df_new)
    st.image("../reports/figures/shap_summary.png")

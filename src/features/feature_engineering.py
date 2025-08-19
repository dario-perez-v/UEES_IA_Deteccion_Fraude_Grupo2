import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def add_features(df):
    """
    Agrega features derivadas al dataset.
    """
    df = df.copy()
    df['Amount_log'] = np.log1p(df['Amount'])
    # Aquí puedes agregar más transformaciones
    return df

def create_pipeline(model):
    """
    Pipeline para aplicar feature engineering y entrenamiento de modelo
    """
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", model)
    ])
    return pipeline


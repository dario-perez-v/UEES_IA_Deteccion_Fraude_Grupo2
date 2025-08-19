import numpy as np

def score_transaction(transaction_df, model):
    """
    Recibe un DataFrame con una o varias transacciones nuevas
    y devuelve la probabilidad de fraude.
    """
    transaction_df = transaction_df.copy()
    # Aplica las mismas transformaciones que en el entrenamiento
    transaction_df['Amount_log'] = np.log1p(transaction_df['Amount'])
    
    X_new = transaction_df.drop("Class", axis=1, errors='ignore')
    y_pred_proba = model.predict_proba(X_new)[:,1]
    return y_pred_proba

import pandas as pd
import numpy as np
from realtime import score_transaction

def generate_transaction():
    """
    Genera una transacción sintética
    """
    trans = pd.DataFrame([{
        'Time': np.random.randint(0, 172792),
        'V1': np.random.normal(),
        'V2': np.random.normal(),
        'Amount': np.random.uniform(0, 1000)
    }])
    return trans

def simulate_and_score(model):
    transaction = generate_transaction()
    prob = score_transaction(transaction, model)
    print(f"Transacción simulada:\n{transaction}")
    print(f"Probabilidad de fraude: {prob[0]:.4f}")
    return transaction, prob


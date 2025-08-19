# Detección de Fraude en Transacciones Financieras

## Resumen

Este proyecto aborda un desafío clave para el sector fintech: detectar fraudes en transacciones digitales minimizando el impacto sobre clientes legítimos. El objetivo es lograr un equilibrio entre la máxima detección de fraudes y la mínima generación de falsos positivos, ya que cada operación legítima bloqueada deteriora la experiencia del usuario y puede generar pérdidas de clientes.

La solución desarrollada es un prototipo de Red Neuronal Artificial (RNA) implementada íntegramente en NumPy, lo que permitió un control total sobre el flujo de datos, inicialización de pesos, funciones de activación y procesos de entrenamiento. Este enfoque garantiza un entendimiento profundo de la lógica interna del modelo y sienta las bases para una transición a entornos productivos.


## Estructura del proyecto
```
├── dashboard/
│       app.py
├── data/
│   │   creditcard.csv
│   └── processed/
│           creditcard_clean.csv
│           creditcard_ready.csv
├── models/
│   └── trained_models/
│           rf_model.pkl
│           xgb_model.pkl
├── notebooks/
│       data_cleaning.ipynb
│       evaluation.ipynb
│       exploratory.ipynb
│       modeling.ipynb
│       simulation.ipynb
└── src/
    │   utils.py
    │   visualization.py
    ├── features/
    │       feature_engineering.py
    ├── interpretability/
    │       shap_analysis.py
    ├── scoring/
    │      realtime.py
    │      simulator.py
    ├── tables/
    │       metrics_RandomForest.csv
    │       metrics_XGBoost.csv
    ├── visualization/
    │       confusion_matrix_RandomForest.png
    │       confusion_matrix_XGBoost.png
    │       correlation_matrix.png
    │       hist_amount.png
    │       pr_curve_RandomForest.png
    │       pr_curve_XGBoost.png
    │       roc_curve.png
    │       roc_curve_RandomForest.png
    │       roc_curve_XGBoost.png
    │   README.md
    │   requirements.txt
    └─  setup.py
```



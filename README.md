# Detección de Fraude en Transacciones Financieras

## Estructura del proyecto
```
├── dashboard
│       app.py
├── data
│   │   creditcard.csv
│   └── processed
│           creditcard_clean.csv
│           creditcard_ready.csv
├── models
│   └── trained_models
│           rf_model.pkl
│           xgb_model.pkl
├── notebooks
│       data_cleaning.ipynb
│       evaluation.ipynb
│       exploratory.ipynb
│       modeling.ipynb
│       simulation.ipynb
└── src
    │   utils.py
    │   visualization.py
    ├── features
    │       feature_engineering.py
    ├── interpretability
    │       shap_analysis.py
    ├── scoring
    │      realtime.py
    │      simulator.py
    ├── tables
    │       metrics_RandomForest.csv
    │       metrics_XGBoost.csv
    ├── visualization
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
    └─   setup.py
```

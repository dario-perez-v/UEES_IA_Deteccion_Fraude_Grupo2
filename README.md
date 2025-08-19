# Detección de Fraude en Transacciones Financieras

## Introduccion 

El fraude financiero es uno de los grandes retos en la era digital. Cada día, bancos y empresas fintech procesan miles de transacciones, y entre ellas se esconden intentos de fraude que, si no son detectados, pueden generar pérdidas económicas y afectar la confianza de los clientes.

Este proyecto nace con ese desafío: diseñar un modelo de inteligencia artificial que identifique transacciones fraudulentas con la mayor precisión posible, pero sin caer en el error de bloquear operaciones legítimas que afectan la experiencia del usuario.

Nuestra propuesta combina modelos de machine learning con técnicas modernas de interpretabilidad. Además, construimos un prototipo práctico en forma de dashboard interactivo, que muestra cómo estas soluciones podrían aplicarse en un escenario real.


## Estructura del proyecto

La estructura del repositorio está diseñada para ser modular y escalable, facilitando la reproducibilidad de los experimentos y la organización de los componentes.

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

## Metodologia del proyecto

## Limpieza y Preparación de Datos

La primera etapa consistió en limpiar y preparar los datos. En el notebook [data_cleaning](notebooks/exploratory/data_cleaning.ipynb), cargamos el dataset de Credit Card Fraud Detection, tratamos valores nulos y outliers, y generamos variables derivadas como Amount_log. Los datos limpios se guardaron en [creditcard_clean](data/processed/creditcard_clean.csv), que luego sirvieron como entrada para los modelos. Durante este proceso se generaron gráficos y tablas que muestran la distribución de los datos antes y después de la limpieza, todos almacenados en [visualization](src/visualization/).

## Modelado de Machine Learning

En los notebooks de modelado (notebooks/modeling/modeling_rf_xgb.ipynb), entrenamos modelos de Random Forest y XGBoost, comparando sus métricas de desempeño como ROC-AUC y PR-AUC. Los modelos se guardaron en models/trained_models/ y los scripts de entrenamiento están en src/models/train_models.py. Esta etapa permitió seleccionar los modelos más precisos para la detección de fraude.

## Evaluación de Modelos

La evaluación de los modelos se realizó en notebooks/evaluation/evaluate_models.ipynb. Se calcularon probabilidades de fraude, curvas ROC y Precision-Recall, matrices de confusión, y se generaron gráficos comparativos entre modelos. Todas las figuras se guardaron en reports/figures/, listas para su inclusión en reportes y presentaciones.

## Pipeline de Feature Engineering

El pipeline de feature engineering, desarrollado en src/features/feature_engineering.py, permite aplicar transformaciones consistentes a nuevas transacciones, asegurando que las características derivadas coincidan con las usadas en los modelos. Esto facilita la integración del sistema en producción y garantiza que el scoring sea confiable.

## Sistema de Scoring en Tiempo Real

El sistema de scoring en tiempo real, implementado en src/scoring/realtime.py, permite calcular la probabilidad de fraude de cualquier transacción nueva. Esto se complementa con el simulador de transacciones, ubicado en src/scoring/simulator.py y notebooks/simulation/simulation.ipynb, que genera transacciones ficticias para probar el flujo completo y visualizar el comportamiento del sistema mediante tablas y gráficos.

## Interpretabilidad del Modelo con SHAP

Para explicar las predicciones de los modelos, utilizamos SHAP en src/interpretability/shap_analysis.py. Esto permite identificar cuáles variables son más relevantes en cada predicción, generando SHAP summary plots que muestran la importancia de cada feature y garantizan la trazabilidad y transparencia del sistema.

## Dashboard Interactivo

Se desarrolló un dashboard interactivo en Streamlit, ubicado en dashboard/app.py. Este dashboard permite:

Subir archivos CSV de transacciones.

Visualizar las probabilidades de fraude calculadas por los modelos.

Explorar gráficos de ROC y Precision-Recall.

Mostrar SHAP summary plots de manera interactiva.

El dashboard facilita la monitorización en tiempo real y la visualización de patrones de fraude, siendo una herramienta clave para analistas y responsables de seguridad financiera.

Esta organización permite un flujo de trabajo claro, reproducible y escalable, facilitando tanto la ejecución de experimentos como la entrega de resultados a stakeholders.



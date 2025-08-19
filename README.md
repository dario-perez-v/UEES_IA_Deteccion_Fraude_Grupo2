# Detección de Fraude en Transacciones Financieras

## 1. Resumen 

El sector **fintech** ha transformado radicalmente la forma en que los usuarios acceden y utilizan servicios financieros. Pagos digitales, transferencias instantáneas, billeteras electrónicas y crédito en línea son hoy servicios cotidianos. Sin embargo, este crecimiento ha venido acompañado de un reto crítico: **el fraude en transacciones digitales**.

Aunque las operaciones fraudulentas representan un porcentaje marginal del volumen total (≈0.1%), concentran hasta un **60% de las pérdidas financieras** en algunas instituciones. El costo no es solo económico: cada transacción legítima bloqueada o cada falso positivo representa una **fricción directa con el cliente**, deteriora la experiencia, y puede desencadenar la pérdida de usuarios hacia competidores. En un mercado hipercompetitivo, la capacidad de diferenciarse por **confianza, seguridad y experiencia fluida** es vital.

Este proyecto aborda ese desafío mediante el desarrollo de un **prototipo integral de detección de fraude en tiempo real**, construido sobre una arquitectura modular y escalable. El sistema combina **machine learning supervisado (Random Forest, XGBoost)** con técnicas de **explicabilidad (SHAP)**, un **pipeline de features estandarizado**, un **simulador de transacciones** y un **dashboard interactivo** para analistas. Adicionalmente, se exploró el diseño de una **Red Neuronal Artificial (RNA)** implementada desde cero en NumPy, lo que permitió un control absoluto del flujo de datos, inicialización de pesos, funciones de activación y proceso de entrenamiento. Este ejercicio académico no solo afianzó el entendimiento profundo de los mecanismos internos de un modelo, sino que además sentó las bases para migrar hacia entornos productivos con tecnologías avanzadas.

En síntesis, el proyecto propone una solución que equilibra tres dimensiones críticas para las fintech:
- **Precisión predictiva**: detección efectiva de transacciones fraudulentas.  
- **Experiencia del cliente**: reducción de falsos positivos y menor fricción.  
- **Confiabilidad regulatoria**: modelos explicables y auditables.  

El impacto proyectado es significativo: **reducción de pérdidas financieras, incremento en la confianza de los clientes, y cumplimiento regulatorio transparente**.

---


## 2. Contexto de Negocio y Problema

En la economía digital, la **confianza del cliente** es el activo más importante. Para las fintech, mantener un balance adecuado entre **seguridad** y **experiencia de usuario** es un reto constante.  
Los principales problemas identificados son:

- **Alta concentración de pérdidas**: aunque las transacciones fraudulentas son escasas, las pérdidas económicas que generan son desproporcionadas.  
- **Falsos positivos**: los modelos tradicionales suelen ser conservadores, marcando como sospechosas operaciones legítimas. Esto degrada la experiencia del cliente y provoca deserción.  
- **Evolución del fraude**: los atacantes innovan constantemente. Un sistema estático pierde efectividad frente a nuevas modalidades.  
- **Exigencia regulatoria**: las instituciones deben demostrar que las decisiones automáticas son explicables y no discriminatorias.  

Frente a este escenario, una solución de **detección de fraude en tiempo real, explicable y adaptable** se convierte en una ventaja competitiva y estratégica.

---

## 3. Objetivos del Proyecto

El proyecto persigue **cuatro objetivos estratégicos**:

1. **Maximizar la detección de fraude** con modelos predictivos robustos y métricas adecuadas (ROC-AUC, PR-AUC, recall).  
2. **Minimizar la tasa de falsos positivos**, asegurando que clientes legítimos no sean penalizados innecesariamente.  
3. **Explicar las decisiones del modelo**, garantizando transparencia regulatoria y confianza por parte de clientes y auditores.  
4. **Construir un prototipo replicable y escalable**, capaz de integrarse en entornos productivos y evolucionar hacia nuevas técnicas (GNNs, autoencoders, aprendizaje online).  

---

## 4. Estructura del proyecto

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



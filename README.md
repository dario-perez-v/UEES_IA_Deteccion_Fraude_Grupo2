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

---

## 5. Datos y Preparación

- **Dataset base**: Credit Card Fraud Detection (Kaggle).  
- **Procesamiento**:  
  - Tratamiento de valores nulos y outliers.  
  - Generación de variables derivadas como `Amount_log`.  
  - Normalización y partición de datos en entrenamiento y validación.  
- **Almacenamiento estructurado**:  
  - `data/raw` → datos originales.  
  - `data/processed` → datos limpios y listos para modelado.  

Esta estrategia asegura reproducibilidad y calidad en cada etapa.  

---

## Metodologia del proyecto

## Limpieza y Preparación de Datos

La primera etapa consistió en limpiar y preparar los datos. En el notebook [data_cleaning](notebooks/data_cleaning.ipynb), cargamos el dataset de Credit Card Fraud Detection, tratamos valores nulos y outliers, y generamos variables derivadas como Amount_log. Los datos limpios se guardaron en [creditcard_clean](data/processed/creditcard_clean.csv), que luego sirvieron como entrada para los modelos. Durante este proceso se generaron gráficos y tablas que muestran la distribución de los datos antes y después de la limpieza, todos almacenados en [visualization](src/visualization/).

En la fase de exploración de datos realizamos un análisis para entender cómo se distribuían las transacciones y si existían patrones que diferenciaran a las fraudulentas de las legítimas.

<p align="center"> <img src="src/visualization/hist_amount.png" alt="Distribución de montos" width="600"><br> <em>Figura 1. Distribución de montos en las transacciones</em> </p>


También se evaluaron las correlaciones entre variables, lo cual permitió identificar relaciones significativas que podían aportar información útil al modelo.

<p align="center"> <img src="src/visualization/correlation_matrix.png" alt="Matriz de correlación" width="650"><br> <em>Figura 2. Correlaciones entre las variables del dataset</em> </p>


## Modelado de Machine Learning

La fase de modelado se llevó a cabo en el notebook [modeling](notebooks/modeling.ipynb). Se entrenaron modelos supervisados de **Random Forest (RF)** y **XGBoost (XGB)**, utilizando el conjunto de datos previamente procesado. Los scripts de entrenamiento se encuentran en [`src/models/train_models.py`](src/models/train_models.py), y los modelos entrenados fueron almacenados en [`models/trained_models/`](models/trained_models/).  

Random Forest fue considerado como baseline por su robustez y facilidad de interpretación inicial, mientras que XGBoost se evaluó como modelo de referencia por su capacidad de manejar datos desbalanceados y capturar interacciones complejas entre variables. Los resultados confirmaron que XGBoost ofrecía un mejor equilibrio entre recall y precisión, convirtiéndose en el modelo con mayor potencial para despliegue en escenarios reales.


## Evaluación de Modelos

La evaluación de los modelos se realizó en el notebook [evaluation](notebooks/evaluation.ipynb). Se calcularon métricas clave como **ROC-AUC** y **PR-AUC**, además de matrices de confusión para comprender en detalle el comportamiento de cada modelo frente a falsos positivos y falsos negativos.  

Los resultados se visualizaron mediante curvas ROC y Precision-Recall, que permiten analizar el desempeño de los clasificadores en contextos de clases desbalanceadas. Todas las figuras se encuentran almacenadas en [`src/visualization/`](src/visualization/), mientras que las métricas tabulares fueron guardadas en [`src/tables/`](src/tables/).  

En las matrices de confusión se evidenció que **Random Forest** tendía a generar un mayor número de falsos negativos, mientras que **XGBoost** alcanzaba un recall superior, lo que lo posiciona como el modelo más adecuado para un sistema antifraude, donde el costo de no detectar un fraude es mayor que el de una alerta falsa.


## Pipeline de Feature Engineering

El pipeline de **feature engineering**, implementado en [`src/features/feature_engineering.py`](src/features/feature_engineering.py), se diseñó para garantizar que las transformaciones aplicadas durante el entrenamiento se repliquen de forma consistente en cualquier nueva transacción evaluada por el sistema.  

Este componente asegura coherencia en el flujo de datos, evita errores en producción y facilita la escalabilidad del sistema. Es un paso clave para trasladar el modelo desde un entorno experimental hacia un entorno productivo.

## Sistema de Scoring en Tiempo Real

Con el propósito de simular un escenario real de negocio, se implementó un sistema de scoring en tiempo real en [`src/scoring/realtime.py`](src/scoring/realtime.py). Este módulo recibe nuevas transacciones y devuelve la **probabilidad de fraude** en cuestión de milisegundos.  

Para validar su funcionamiento se desarrolló además un **simulador de transacciones** en [`src/scoring/simulator.py`](src/scoring/simulator.py) y en el notebook [simulation](notebooks/simulation.ipynb). El simulador permite generar datos sintéticos y someter al sistema a distintos escenarios de carga, lo que facilita la validación end-to-end y la calibración de umbrales de decisión.


## Interpretabilidad del Modelo con SHAP

La interpretabilidad es un aspecto crítico en soluciones antifraude, tanto para garantizar confianza con los clientes como para cumplir requisitos regulatorios. Por ello se utilizó **SHAP** en [`src/interpretability/shap_analysis.py`](src/interpretability/shap_analysis.py).  

El análisis con SHAP permitió identificar qué variables tenían mayor influencia en las predicciones. Se generaron tanto gráficos **globales** (summary plots que muestran el impacto agregado de las variables) como explicaciones **locales** (caso por caso), ofreciendo una trazabilidad completa de las decisiones del modelo.  


## Dashboard Interactivo

Finalmente, se desarrolló un **dashboard interactivo** en Streamlit, ubicado en [`dashboard/app.py`](dashboard/app.py). Esta interfaz permite:  

- Subir archivos CSV de transacciones.  
- Visualizar las probabilidades de fraude generadas por los modelos.  
- Explorar gráficas de rendimiento como curvas ROC y Precision-Recall.  
- Consultar de manera interactiva los SHAP summary plots para interpretar las decisiones del sistema.  

El dashboard constituye una herramienta clave para **analistas y responsables de seguridad financiera**, ya que facilita la monitorización del sistema en tiempo real y acerca resultados técnicos a perfiles no técnicos.  

En conjunto, esta metodología asegura un flujo de trabajo claro, reproducible y escalable, que cubre desde la preparación inicial de los datos hasta la interpretación y visualización de los resultados, alineándose con las necesidades del negocio y los requisitos técnicos de un sistema antifraude moderno.

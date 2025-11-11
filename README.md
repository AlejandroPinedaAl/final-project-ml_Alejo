# Marketing Campaign Response Prediction

## 📊 Contexto del Negocio

Este proyecto desarrolla un modelo predictivo para identificar qué clientes tienen mayor probabilidad de responder positivamente a una campaña de marketing. El objetivo es maximizar la eficiencia y rentabilidad de futuras campañas mediante la segmentación inteligente de clientes.

## 🎯 Objetivo del Proyecto

Predecir la variable `Response` (1 = acepta la oferta, 0 = rechaza) utilizando técnicas de Machine Learning supervisado, implementando un pipeline completo de MLOps que incluye:

- Análisis Exploratorio de Datos (EDA)
- Ingeniería de Características
- Entrenamiento y Evaluación de Múltiples Modelos
- Monitoreo de Data Drift
- Despliegue como API REST
- Validación de Calidad de Código con SonarCloud

## 📁 Estructura del Proyecto

```
final-project-ml_Alejo/
├── mlops_pipeline/
│   └── src/
│       ├── Cargar_datos.ipynb              # Carga inicial del dataset
│       ├── comprension_eda.ipynb           # Análisis exploratorio
│       ├── ft_engineering.py               # Ingeniería de features
│       ├── model_training_evaluation.py    # Entrenamiento de modelos
│       ├── model_deploy.py                 # API de despliegue
│       ├── model_monitoring.py             # Monitoreo de drift
│       └── heuristic_model.py              # Modelo heurístico base
├── Base_de_datos.csv                       # Dataset principal
├── requirements.txt                        # Dependencias Python
├── config.json                             # Configuración del proyecto
├── setup.bat                               # Script de configuración de entorno
├── .gitignore                              # Archivos ignorados por Git
└── README.md                               # Este archivo
```

## 📊 Dataset

**Fuente**: Marketing Campaign Dataset (Kaggle)

**Descripción**: Dataset con información de clientes y su respuesta a campañas de marketing.

### Variables Principales:

- **Variable Objetivo**: `Response` (1 = acepta, 0 = rechaza)
- **Campañas Anteriores**: AcceptedCmp1-5, Complain
- **Demografía**: Education, Marital, Income, Kidhome, Teenhome, DtCustomer
- **Comportamiento de Compra**: MntWines, MntFruits, MntMeatProducts, MntFishProducts, MntSweetProducts, MntGoldProds
- **Canales**: NumWebPurchases, NumCatalogPurchases, NumStorePurchases, NumDealsPurchases
- **Actividad**: NumWebVisitsMonth, Recency

## 🚀 Instalación y Uso

### Requisitos Previos

- Python 3.9+
- Git

### Instalación

1. Clonar el repositorio:
```bash
git clone https://github.com/AlejandroPinedaAl/final-project-ml_Alejo.git
cd final-project-ml_Alejo
```

2. Configurar entorno virtual e instalar dependencias:
```bash
# En Windows
setup.bat

# En Linux/Mac
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3. Ejecutar notebooks de análisis:
```bash
jupyter notebook
```

### Ejecución de la API

```bash
cd mlops_pipeline/src
uvicorn model_deploy:app --reload
```

La API estará disponible en: `http://localhost:8000`

### Ejecución del Dashboard de Monitoreo

```bash
streamlit run mlops_pipeline/src/model_monitoring.py
```

## 🔧 Tecnologías Utilizadas

- **Lenguaje**: Python 3.9
- **Análisis de Datos**: pandas, numpy
- **Machine Learning**: scikit-learn, XGBoost, LightGBM
- **Visualización**: matplotlib, seaborn, plotly
- **API**: FastAPI, Uvicorn
- **Monitoreo**: Streamlit, scipy
- **Notebooks**: Jupyter
- **Calidad de Código**: SonarCloud
- **Versionamiento**: Git, GitHub

## 📈 Proceso de Desarrollo

### Fase 1: Exploración de Datos (EDA)
- Análisis univariable, bivariable y multivariable
- Identificación de patrones y correlaciones
- Detección de outliers y valores nulos

### Fase 2: Ingeniería de Características
- Creación de features derivados
- Pipelines de transformación
- Escalado y codificación

### Fase 3: Entrenamiento de Modelos
- Múltiples algoritmos (Logistic Regression, Random Forest, XGBoost, LightGBM)
- Validación cruzada
- Selección del mejor modelo

### Fase 4: Monitoreo
- Detección de Data Drift (KS, PSI, Jensen-Shannon, Chi-cuadrado)
- Dashboard interactivo con Streamlit

### Fase 5: Despliegue
- API REST con FastAPI
- Endpoints para predicción individual y por lotes
- Dockerización

## 📊 Resultados

_Los resultados se actualizarán una vez completado el entrenamiento de modelos._

## 👥 Autor

**Alejandro Pineda Alvarez**
- GitHub: [@AlejandroPinedaAl](https://github.com/AlejandroPinedaAl)

## 📝 Licencia

Este proyecto es parte del curso de Machine Learning y está disponible para fines educativos.

## 🏆 Estado del Proyecto

![Status](https://img.shields.io/badge/Status-En%20Desarrollo-yellow)

---

**Proyecto Final - Machine Learning**  
**Docente**: Juan Sebastián Parra Sánchez  
**Fecha de Entrega**: 10 de noviembre de 2025

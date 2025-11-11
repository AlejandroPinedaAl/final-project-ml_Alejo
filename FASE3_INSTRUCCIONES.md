# 📋 FASE 3: ENTRENAMIENTO Y EVALUACIÓN DE MODELOS - INSTRUCCIONES

## ✅ LIBRERÍAS NECESARIAS

### Librerías ya instaladas (de la Fase 2):
- ✅ pandas
- ✅ numpy
- ✅ matplotlib
- ✅ seaborn
- ✅ scipy
- ✅ scikit-learn
- ✅ joblib

### Librerías instaladas para la Fase 3:

```bash
pip install xgboost lightgbm
```

**Verificar instalación**:
```bash
pip list | findstr "xgboost lightgbm scikit-learn"
```

Debes ver:
- xgboost (versión 3.x o superior)
- lightgbm (versión 4.x o superior)
- scikit-learn (versión 1.7.x o superior)

---

## 📦 DEPENDENCIAS COMPLETAS

### Comandos de instalación:

```bash
# Instalar XGBoost
pip install xgboost

# Instalar LightGBM
pip install lightgbm
```

### Verificar instalación:

```bash
python -c "import xgboost; print('XGBoost:', xgboost.__version__)"
python -c "import lightgbm; print('LightGBM:', lightgbm.__version__)"
python -c "import sklearn; print('scikit-learn:', sklearn.__version__)"
```

---

## 📓 NOTEBOOK CREADO

**Archivo**: `mlops_pipeline/src/model_training_evaluation_fase3.ipynb`

Este notebook contiene todas las celdas necesarias para ejecutar la Fase 3 completa.

---

## 🚀 PASOS PARA EJECUTAR LA FASE 3

### 1. Verificar que la Fase 2 esté completada

Asegúrate de que tengas los siguientes archivos en la raíz del proyecto:
- ✅ `X_train_transformed.csv`
- ✅ `X_test_transformed.csv`
- ✅ `y_train.csv`
- ✅ `y_test.csv`
- ✅ `preprocessor.pkl` (opcional pero recomendado)

### 2. Abrir el notebook

```bash
# Desde Jupyter Notebook o JupyterLab
jupyter notebook mlops_pipeline/src/model_training_evaluation_fase3.ipynb
```

O desde VS Code:
- Abre el archivo `model_training_evaluation_fase3.ipynb`
- Selecciona el kernel de Python

### 3. Ejecutar las celdas en orden

El notebook está organizado en las siguientes secciones:

1. **Importación de Librerías** (Celda 1)
2. **Carga de Datos** (Celdas 2-5)
   - Cargar datos transformados
   - Verificar distribución de clases
   - Cargar preprocessor
3. **Definición de Modelos** (Celdas 6-8)
   - Calcular peso para balanceo
   - Definir 7 modelos
4. **Función de Entrenamiento** (Celda 9)
   - Función `build_model()` completa
5. **Entrenamiento de Modelos** (Celdas 10-11)
   - Entrenar los 7 modelos
   - Validación cruzada (5-fold)
6. **Comparación de Modelos** (Celdas 12-14)
   - Tabla comparativa
   - Identificar mejor modelo
7. **Visualizaciones** (Celdas 15-19)
   - Comparación de métricas
   - Comparación de ROC-AUC
   - Curvas ROC
   - Matrices de confusión
   - Feature importance
8. **Guardar Mejor Modelo** (Celdas 20-21)
   - Guardar modelo con metadata
9. **Resumen Final** (Celda 22)
10. **Verificación** (Celda 23)

### 4. Verificar resultados

Después de ejecutar todas las celdas, deberías tener los siguientes archivos:

- ✅ `best_model.pkl` - Modelo entrenado con metadata
- ✅ `model_comparison_metrics.png` - Gráfico de métricas
- ✅ `model_comparison_roc_auc.png` - Gráfico de ROC-AUC
- ✅ `roc_curves_comparison.png` - Curvas ROC de todos los modelos
- ✅ `confusion_matrices.png` - Matrices de confusión
- ✅ `feature_importance_[modelo].png` - Importancia de features (si aplica)

---

## 🔍 VERIFICACIÓN DE RESULTADOS

### Verificar que los archivos se crearon:

```bash
# Desde la raíz del proyecto
dir best_model.pkl
dir model_comparison_*.png
dir roc_curves_comparison.png
dir confusion_matrices.png
```

### Verificar métricas del mejor modelo:

El notebook imprime un resumen al final con:
- Nombre del mejor modelo
- F1-Score, ROC-AUC, Accuracy
- Tabla comparativa completa
- Gráficos de visualización

---

## ⚠️ POSIBLES PROBLEMAS Y SOLUCIONES

### Problema 1: Error al cargar datos

**Error**: `FileNotFoundError: X_train_transformed.csv`

**Solución**: Asegúrate de haber ejecutado la Fase 2 primero y que los archivos estén en la raíz del proyecto.

### Problema 2: Error con XGBoost o LightGBM

**Error**: `ModuleNotFoundError: No module named 'xgboost'`

**Solución**: Instala las librerías:
```bash
pip install xgboost lightgbm
```

### Problema 3: Error con class_weight='balanced'

**Error**: Algunos modelos pueden no soportar `class_weight='balanced'`

**Solución**: El notebook maneja esto automáticamente. Si hay errores, puedes comentar esos modelos temporalmente.

### Problema 4: Tiempo de entrenamiento muy largo

**Solución**: 
- Reduce `n_estimators` en los modelos de ensemble (de 100 a 50)
- Reduce `cv_folds` de 5 a 3
- Comenta modelos que tarden mucho (SVM puede ser lento)

### Problema 5: Memory Error

**Solución**:
- Reduce el tamaño del dataset si es muy grande
- Usa `n_jobs=1` en lugar de `n_jobs=-1`
- Cierra otras aplicaciones

---

## 📊 MODELOS QUE SE ENTRENAN

El notebook entrena los siguientes 7 modelos:

1. **Logistic Regression** (baseline)
   - Rápido, interpretable
   - Class weight: balanced

2. **Random Forest** (ensemble)
   - Robusto, maneja bien desbalance
   - Class weight: balanced
   - n_estimators: 100

3. **Gradient Boosting** (boosting)
   - Buen rendimiento
   - n_estimators: 100

4. **Extra Trees** (ensemble)
   - Similar a Random Forest
   - Class weight: balanced
   - n_estimators: 100

5. **SVM** (kernel-based)
   - Puede ser lento con datasets grandes
   - Class weight: balanced
   - Kernel: RBF

6. **XGBoost** (boosting avanzado)
   - Excelente rendimiento
   - scale_pos_weight: calculado automáticamente
   - n_estimators: 100

7. **LightGBM** (boosting rápido)
   - Rápido y eficiente
   - Class weight: balanced
   - n_estimators: 100

---

## 📈 MÉTRICAS QUE SE EVALÚAN

Para cada modelo se calculan:

### Métricas Básicas:
- **Accuracy**: Precisión general
- **Precision**: Precisión de predicciones positivas
- **Recall**: Sensibilidad (captura de positivos)
- **F1-Score**: Media armónica de precision y recall

### Métricas Avanzadas:
- **ROC-AUC**: Área bajo la curva ROC
- **Average Precision**: Precisión promedio

### Validación Cruzada:
- **5-fold estratificada**: Media y desviación estándar
- **Métricas**: accuracy, precision, recall, f1, roc_auc

### Overfitting Check:
- **Diferencia Train vs Test**: Accuracy y F1-Score
- **Umbral de alerta**: >0.1 diferencia

---

## 🎯 SELECCIÓN DEL MEJOR MODELO

El mejor modelo se selecciona basado en:

1. **F1-Score en Test**: Métrica principal (balance entre precision y recall)
2. **ROC-AUC**: Capacidad de discriminación
3. **Consistencia**: Bajo overfitting (train vs test)
4. **Validación Cruzada**: Estabilidad en diferentes folds

**Criterios de selección**:
- Mayor F1-Score
- ROC-AUC > 0.7 (bueno)
- Overfitting < 0.1 (bajo)
- CV std < 0.05 (consistente)

---

## 📊 VISUALIZACIONES GENERADAS

### 1. Comparación de Métricas Principales
- Gráfico de barras con Accuracy, Precision, Recall, F1-Score
- Comparación visual de todos los modelos

### 2. Comparación de ROC-AUC
- Gráfico de barras horizontales
- Ordenado por ROC-AUC

### 3. Curvas ROC
- Curvas ROC de todos los modelos en un solo gráfico
- Línea de referencia (random classifier)
- AUC de cada modelo en la leyenda

### 4. Matrices de Confusión
- Grid de matrices de confusión (3 columnas)
- Una matriz por modelo
- Heatmaps con valores anotados

### 5. Feature Importance
- Top 20 features más importantes
- Solo para modelos que lo soporten (tree-based)
- Gráfico de barras horizontales

---

## 💾 ARCHIVOS GENERADOS

### Archivos Principales:
1. **best_model.pkl**: Modelo entrenado con metadata completa
   - Modelo entrenado
   - Métricas de evaluación
   - Preprocessor (opcional)
   - Nombres de features
   - Timestamp y versión

### Archivos de Visualización:
2. **model_comparison_metrics.png**: Comparación de métricas
3. **model_comparison_roc_auc.png**: Comparación de ROC-AUC
4. **roc_curves_comparison.png**: Curvas ROC
5. **confusion_matrices.png**: Matrices de confusión
6. **feature_importance_[modelo].png**: Importancia de features

---

## ✅ CHECKLIST DE VERIFICACIÓN

Antes de pasar a la Fase 4, verifica que:

- [ ] Todas las celdas del notebook se ejecutaron sin errores
- [ ] El archivo `best_model.pkl` se creó correctamente
- [ ] Los gráficos se guardaron correctamente
- [ ] La tabla comparativa muestra todos los modelos
- [ ] El mejor modelo tiene métricas razonables (F1 > 0.5, ROC-AUC > 0.7)
- [ ] No hay overfitting excesivo (diferencia < 0.1)
- [ ] El modelo se puede cargar correctamente

---

## 🎯 PRÓXIMOS PASOS

Una vez completada la Fase 3:

1. ✅ Verifica que todos los archivos se guardaron correctamente
2. ✅ Revisa la tabla comparativa de modelos
3. ✅ Analiza las visualizaciones
4. ✅ Verifica que el mejor modelo tenga buen rendimiento
5. ✅ Avísame cuando estés listo para la Fase 4

**Fase 4**: Monitoreo y Detección de Data Drift
- Necesitarás: streamlit, scipy (ya instalados)
- Archivos de entrada: best_model.pkl, datos históricos

---

## 📝 NOTAS IMPORTANTES

1. **Tiempo de Ejecución**: El entrenamiento puede tardar varios minutos (depende de tu máquina)
   - Logistic Regression: ~1 segundo
   - Random Forest: ~5-10 segundos
   - Gradient Boosting: ~10-20 segundos
   - XGBoost: ~5-10 segundos
   - LightGBM: ~3-5 segundos
   - SVM: ~30-60 segundos (puede ser más lento)

2. **Balanceo de Clases**: Todos los modelos usan técnicas de balanceo:
   - `class_weight='balanced'` en scikit-learn
   - `scale_pos_weight` en XGBoost
   - Esto es crucial para datasets desbalanceados

3. **Validación Cruzada**: Se usa 5-fold estratificada para:
   - Evaluar estabilidad del modelo
   - Detectar overfitting
   - Obtener métricas más confiables

4. **Reproducibilidad**: Se usa `random_state=42` en todos los modelos para resultados reproducibles

5. **Guardado del Modelo**: El modelo se guarda con metadata completa para facilitar el despliegue en la Fase 5

---

## 🔧 CONFIGURACIÓN AVANZADA

### Si quieres ajustar parámetros:

Puedes modificar los modelos en la celda de definición:

```python
# Ejemplo: Aumentar n_estimators para mejor rendimiento (más lento)
'Random Forest': RandomForestClassifier(
    n_estimators=200,  # Aumentado de 100 a 200
    random_state=42,
    class_weight='balanced',
    n_jobs=-1
)

# Ejemplo: Reducir cv_folds para ejecución más rápida
cv_folds = 3  # Reducido de 5 a 3
```

### Si quieres agregar más modelos:

```python
# Agregar al diccionario de modelos
from sklearn.neural_network import MLPClassifier

models['Neural Network'] = MLPClassifier(
    hidden_layer_sizes=(100, 50),
    max_iter=500,
    random_state=42
)
```

---

## 📊 INTERPRETACIÓN DE RESULTADOS

### Métricas Clave:

- **F1-Score > 0.7**: Excelente
- **F1-Score > 0.6**: Bueno
- **F1-Score > 0.5**: Aceptable
- **F1-Score < 0.5**: Necesita mejora

- **ROC-AUC > 0.8**: Excelente
- **ROC-AUC > 0.7**: Bueno
- **ROC-AUC > 0.6**: Aceptable
- **ROC-AUC < 0.6**: Necesita mejora

### Overfitting:

- **Diferencia < 0.05**: Excelente (muy poco overfitting)
- **Diferencia < 0.1**: Bueno (poco overfitting)
- **Diferencia > 0.1**: Advertencia (posible overfitting)
- **Diferencia > 0.2**: Crítico (overfitting significativo)

---

**Fecha de creación**: Noviembre 2025
**Autor**: Alejandro Pineda Alvarez
**Proyecto**: Marketing Campaign Response Prediction


# 📋 FASE 2: INGENIERÍA DE CARACTERÍSTICAS - INSTRUCCIONES

## ✅ LIBRERÍAS NECESARIAS

### Librerías ya instaladas (de la Fase 1):
- ✅ pandas
- ✅ numpy
- ✅ matplotlib
- ✅ seaborn
- ✅ scipy

### Librerías a instalar para la Fase 2:

```bash
pip install scikit-learn joblib
```

**Nota**: Si ya las instalaste, puedes verificar con:
```bash
pip list | findstr "scikit-learn joblib"
```

---

## 📦 DEPENDENCIAS COMPLETAS

### Comandos de instalación:

```bash
# Instalar scikit-learn (incluye todas las herramientas de ML)
pip install scikit-learn

# Instalar joblib (para guardar/cargar modelos)
pip install joblib
```

### Verificar instalación:

```bash
python -c "import sklearn; print('scikit-learn:', sklearn.__version__)"
python -c "import joblib; print('joblib: OK')"
```

---

## 📓 NOTEBOOK CREADO

**Archivo**: `mlops_pipeline/src/feature_engineering_fase2.ipynb`

Este notebook contiene todas las celdas necesarias para ejecutar la Fase 2 completa.

---

## 🚀 PASOS PARA EJECUTAR LA FASE 2

### 1. Verificar que la Fase 1 esté completada

Asegúrate de que tengas el archivo `data_with_features.csv` en la raíz del proyecto:
- Si ejecutaste la Fase 1 manualmente, deberías tener este archivo
- Si no, el notebook cargará desde `Base_de_datos.csv` automáticamente

### 2. Abrir el notebook

```bash
# Desde Jupyter Notebook o JupyterLab
jupyter notebook mlops_pipeline/src/feature_engineering_fase2.ipynb
```

O desde VS Code:
- Abre el archivo `feature_engineering_fase2.ipynb`
- Selecciona el kernel de Python

### 3. Ejecutar las celdas en orden

El notebook está organizado en las siguientes secciones:

1. **Importación de Librerías** (Celda 1)
2. **Carga de Datos** (Celdas 2-3)
3. **Limpieza de Datos** (Celdas 4-8)
   - Eliminar variables irrelevantes
   - Convertir tipos de datos
   - Manejo de valores nulos
   - Unificación de categorías
4. **Creación de Features Derivados** (Celdas 9-11)
   - Features de gastos y compras
   - Features de campañas
   - Features temporales
5. **Preparación para Modelado** (Celdas 12-14)
   - Separar X e y
   - Identificar tipos de variables
6. **Pipeline de Preprocesamiento** (Celdas 15-17)
   - Pipeline numérico
   - Pipeline categórico
   - Combinar pipelines
7. **Split de Datos** (Celdas 18-21)
   - Dividir train/test
   - Transformar datos
   - Obtener nombres de features
8. **Guardar Resultados** (Celdas 22-24)
   - Guardar preprocessor
   - Guardar datos procesados
   - Guardar datos transformados
9. **Resumen Final** (Celda 25)

### 4. Verificar resultados

Después de ejecutar todas las celdas, deberías tener los siguientes archivos:

- ✅ `preprocessor.pkl` - Pipeline de preprocesamiento
- ✅ `data_processed.csv` - Dataset procesado completo
- ✅ `X_train_transformed.csv` - Features de entrenamiento transformadas
- ✅ `X_test_transformed.csv` - Features de prueba transformadas
- ✅ `y_train.csv` - Target de entrenamiento
- ✅ `y_test.csv` - Target de prueba

---

## 🔍 VERIFICACIÓN DE RESULTADOS

### Verificar que los archivos se crearon:

```bash
# Desde la raíz del proyecto
dir preprocessor.pkl
dir data_processed.csv
dir X_train_transformed.csv
dir X_test_transformed.csv
dir y_train.csv
dir y_test.csv
```

### Verificar dimensiones:

El notebook imprime un resumen al final con:
- Dimensiones del dataset procesado
- Número de features derivados
- Número de variables numéricas y categóricas
- Dimensiones de train y test sets

---

## ⚠️ POSIBLES PROBLEMAS Y SOLUCIONES

### Problema 1: Error al cargar datos

**Solución**: Verifica que el archivo `Base_de_datos.csv` o `data_with_features.csv` exista en la raíz del proyecto.

### Problema 2: Error con OneHotEncoder

**Solución**: Asegúrate de tener scikit-learn >= 1.2.0 instalado:
```bash
pip install --upgrade scikit-learn
```

### Problema 3: Error al guardar preprocessor

**Solución**: Verifica que joblib esté instalado:
```bash
pip install joblib
```

### Problema 4: Warning sobre sparse_output

**Solución**: Si usas scikit-learn < 1.2, cambia `sparse_output=False` a `sparse=False` en el notebook.

---

## 📊 ESTRUCTURA DEL NOTEBOOK

El notebook está organizado de la siguiente manera:

```
1. Introducción y Objetivo
2. Importación de Librerías
3. Carga de Datos
4. Limpieza de Datos
   - Eliminar variables irrelevantes
   - Convertir tipos
   - Manejar nulos
   - Unificar categorías
5. Creación de Features Derivados
   - 12 features nuevos
6. Preparación para Modelado
   - Separar X e y
   - Identificar tipos
7. Pipeline de Preprocesamiento
   - Pipelines numérico y categórico
   - ColumnTransformer
8. Split de Datos
   - Train/Test estratificado
   - Transformación
9. Guardar Resultados
   - Preprocessor
   - Datos procesados
10. Resumen Final
```

---

## ✅ CHECKLIST DE VERIFICACIÓN

Antes de pasar a la Fase 3, verifica que:

- [ ] Todas las celdas del notebook se ejecutaron sin errores
- [ ] El archivo `preprocessor.pkl` se creó correctamente
- [ ] Los archivos CSV se guardaron correctamente
- [ ] Las dimensiones de train y test son correctas
- [ ] La distribución de clases se mantiene en train y test
- [ ] No hay valores NaN en los datos transformados

---

## 🎯 PRÓXIMOS PASOS

Una vez completada la Fase 2:

1. ✅ Verifica que todos los archivos se guardaron correctamente
2. ✅ Revisa el resumen final del notebook
3. ✅ Avísame cuando estés listo para la Fase 3

**Fase 3**: Entrenamiento y Evaluación de Modelos
- Necesitarás: scikit-learn, xgboost, lightgbm (opcional)
- Archivos de entrada: X_train_transformed.csv, y_train.csv, etc.

---

## 📝 NOTAS IMPORTANTES

1. **Estratificación**: El split de datos usa estratificación para mantener la proporción de clases (importante para datasets desbalanceados)

2. **RobustScaler**: Se usa RobustScaler en lugar de StandardScaler porque es más robusto a outliers (recomendado para este dataset)

3. **OneHotEncoder**: Se usa `drop='first'` para evitar multicolinealidad

4. **Data Leakage**: El split se hace ANTES de transformar para evitar data leakage

5. **Reproducibilidad**: Se usa `random_state=42` para garantizar resultados reproducibles

---

**Fecha de creación**: Noviembre 2025
**Autor**: Alejandro Pineda Alvarez
**Proyecto**: Marketing Campaign Response Prediction


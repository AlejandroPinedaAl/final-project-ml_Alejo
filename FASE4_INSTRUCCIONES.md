# FASE 4: Monitoreo y Detección de Data Drift

## 📋 Resumen

Esta fase implementa el monitoreo del modelo y la detección de data drift, comparando datos históricos (baseline) con datos actuales para detectar cambios en las distribuciones.

---

## 📦 Librerías Necesarias

### Instalación

```bash
pip install streamlit plotly scipy
```

O instalar todas las dependencias:

```bash
pip install -r requirements.txt
```

### Librerías Principales

- **streamlit**: Aplicación web interactiva
- **plotly**: Gráficos interactivos
- **scipy**: Métricas estadísticas (KS test, Chi-cuadrado, Jensen-Shannon)
- **pandas**: Manipulación de datos
- **numpy**: Operaciones numéricas
- **matplotlib**: Gráficos estáticos
- **seaborn**: Visualizaciones estadísticas

---

## 🚀 Ejecución

### Opción 1: Notebook Jupyter (Recomendado para análisis)

```bash
jupyter notebook mlops_pipeline/src/model_monitoring_fase4.ipynb
```

O en VS Code:
1. Abre `model_monitoring_fase4.ipynb`
2. Selecciona el kernel de Python
3. Ejecuta todas las celdas (Run All)

### Opción 2: Aplicación Streamlit (Recomendado para monitoreo continuo)

```bash
streamlit run mlops_pipeline/src/streamlit_monitoring_app.py
```

La aplicación se abrirá automáticamente en tu navegador en `http://localhost:8501`

---

## 📊 Métricas de Drift Implementadas

### 1. PSI (Population Stability Index)

- **PSI < 0.1**: Sin cambio significativo ✅
- **0.1 ≤ PSI < 0.2**: Cambio moderado ⚠️
- **PSI ≥ 0.2**: Cambio significativo 🚨

### 2. KS Test (Kolmogorov-Smirnov)

- Compara distribuciones numéricas
- Estadístico: Distancia máxima entre distribuciones
- p-value: Significancia estadística

### 3. Jensen-Shannon Divergence

- **0**: Distribuciones idénticas
- **1**: Distribuciones completamente diferentes

### 4. Chi-cuadrado

- Para variables categóricas
- Test de independencia
- p-value: Significancia del cambio

---

## 📁 Archivos de Entrada

### Datos Baseline (Requeridos)

- `data_processed.csv` (de la Fase 2)
- O `X_train_transformed.csv` (de la Fase 2)

### Datos Actuales

- En producción: Datos nuevos que llegan continuamente
- En el notebook: Simulación con muestra de datos baseline

---

## 📤 Archivos de Salida

Después de ejecutar el notebook, se generarán:

1. **drift_results.csv**: Métricas de drift por variable
2. **drift_summary.json**: Resumen de resultados
3. **Gráficos**: Visualizaciones de distribuciones y métricas

---

## 🔍 Pasos de Ejecución

### 1. Preparación

```bash
# Verificar que las librerías están instaladas
python -c "import streamlit; import plotly; import scipy; print('OK')"
```

### 2. Ejecutar Notebook

1. Abre `model_monitoring_fase4.ipynb`
2. Ejecuta todas las celdas en orden
3. Revisa los resultados:
   - Resumen de drift
   - Alertas
   - Visualizaciones
   - Recomendaciones

### 3. Ejecutar Streamlit App

```bash
streamlit run mlops_pipeline/src/streamlit_monitoring_app.py
```

1. En el sidebar, activa "Usar datos guardados"
2. Ajusta los umbrales si es necesario
3. Revisa el dashboard:
   - Resumen general
   - Alertas
   - Métricas por variable
   - Visualizaciones
   - Recomendaciones

---

## 📈 Interpretación de Resultados

### Estado de Drift

- **no_drift**: No se detectaron cambios significativos
- **moderate_drift**: Cambios moderados detectados
- **significant_drift**: Cambios significativos detectados

### Acciones Recomendadas

#### Si hay drift significativo:

1. 🚨 **Revisar variables críticas**
2. 🔍 **Investigar causas del cambio**
3. 🔄 **Considerar retraining del modelo**
4. 📊 **Actualizar dataset baseline si el cambio es válido**

#### Si hay drift moderado:

1. ⚠️ **Monitorear variables**
2. 📈 **Revisar tendencias temporales**
3. 🔧 **Considerar ajustes menores**

#### Si no hay drift:

1. ✅ **Continuar con monitoreo regular**
2. 📊 **Mantener modelo actual**
3. 🔄 **Actualizar baseline periódicamente**

---

## 🔧 Configuración de Umbrales

### Umbrales por Defecto

```python
THRESHOLD_PSI = 0.2      # PSI
THRESHOLD_KS = 0.2       # KS Test
THRESHOLD_JS = 0.2       # Jensen-Shannon
THRESHOLD_CHI2 = 0.05    # Chi-cuadrado (p-value)
```

### Ajustar Umbrales

En el notebook o en Streamlit, puedes ajustar los umbrales según tus necesidades:

- **Umbrales más estrictos**: Detectan cambios más pequeños
- **Umbrales más laxos**: Solo detectan cambios grandes

---

## 📊 Visualizaciones

### 1. Distribución de Estados de Drift

- Gráfico de barras con estado por variable
- Colores: Verde (sin drift), Naranja (moderado), Rojo (significativo)

### 2. PSI por Variable

- Gráfico de barras horizontales
- Línea de umbral
- Colores según estado

### 3. Distribuciones Baseline vs Actual

- Histogramas superpuestos (variables numéricas)
- Gráficos de barras (variables categóricas)
- Comparación visual de distribuciones

---

## ⚠️ Solución de Problemas

### Error: "No se encontraron datos baseline"

**Solución**: Ejecuta primero la Fase 2 para generar los datos procesados.

### Error: "ModuleNotFoundError: No module named 'streamlit'"

**Solución**: 
```bash
pip install streamlit plotly
```

### Error: "ModuleNotFoundError: No module named 'model_monitoring'"

**Solución**: Asegúrate de que `model_monitoring.py` esté en el mismo directorio.

### La aplicación Streamlit no se abre

**Solución**: 
1. Verifica que el puerto 8501 no esté en uso
2. Ejecuta: `streamlit run streamlit_monitoring_app.py --server.port 8502`

---

## ✅ Checklist de Verificación

- [ ] Librerías instaladas (streamlit, plotly, scipy)
- [ ] Datos baseline cargados correctamente
- [ ] Datos actuales cargados (o simulados)
- [ ] Métricas de drift calculadas
- [ ] Resultados guardados (drift_results.csv, drift_summary.json)
- [ ] Visualizaciones generadas
- [ ] Alertas revisadas
- [ ] Recomendaciones analizadas
- [ ] Aplicación Streamlit funcionando (opcional)

---

## 📝 Notas Importantes

1. **Datos de Producción**: En producción, los datos actuales deben ser datos reales nuevos, no una muestra de los datos baseline.

2. **Frecuencia de Monitoreo**: Se recomienda ejecutar el monitoreo regularmente (diario, semanal, mensual) según el contexto del negocio.

3. **Umbrales**: Los umbrales por defecto son sugerencias. Ajusta según tu dominio y tolerancia al riesgo.

4. **Retraining**: Si se detecta drift significativo, considera retraining del modelo con datos actualizados.

5. **Baseline**: El dataset baseline debe representar los datos con los que se entrenó el modelo originalmente.

---

## 🎯 Próximos Pasos

Una vez completada la Fase 4:

1. ✅ Revisar resultados de drift
2. ✅ Analizar variables con drift significativo
3. ✅ Implementar monitoreo continuo (opcional)
4. ✅ Continuar con la Fase 5 (Despliegue del Modelo)

---

## 📚 Referencias

- **PSI**: Population Stability Index
- **KS Test**: Kolmogorov-Smirnov Test
- **JS Divergence**: Jensen-Shannon Divergence
- **Chi-cuadrado**: Chi-square Test

---

## 💡 Tips

1. **Monitoreo Automatizado**: Considera automatizar el monitoreo con cron jobs o schedulers.

2. **Alertas**: Configura alertas automáticas cuando se detecte drift significativo.

3. **Histórico**: Guarda un histórico de métricas de drift para análisis temporal.

4. **Dashboard**: Usa la aplicación Streamlit para crear un dashboard de monitoreo en tiempo real.

5. **Documentación**: Documenta los umbrales y decisiones de monitoreo.

---

**Autor**: Alejandro Pineda Alvarez  
**Proyecto**: Marketing Campaign Response Prediction  
**Fecha**: 2025


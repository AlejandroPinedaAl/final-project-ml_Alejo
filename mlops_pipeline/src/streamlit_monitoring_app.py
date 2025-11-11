"""
streamlit_monitoring_app.py
Aplicación Streamlit para monitoreo de data drift.

Ejecutar con:
    streamlit run streamlit_monitoring_app.py

Autor: Alejandro Pineda Alvarez
Proyecto: Marketing Campaign Response Prediction
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import sys
import os

# Agregar el directorio actual al path para importar model_monitoring
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model_monitoring import detect_drift, get_drift_status

# Configuración de la página
st.set_page_config(
    page_title="Monitoreo de Data Drift - Marketing Campaign",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Monitoreo de Data Drift - Marketing Campaign")
st.markdown("---")

# Sidebar
st.sidebar.title("Configuración")
st.sidebar.markdown("### Parámetros de Drift")

threshold_psi = st.sidebar.slider("Umbral PSI", 0.0, 1.0, 0.2, 0.05)
threshold_ks = st.sidebar.slider("Umbral KS", 0.0, 1.0, 0.2, 0.05)
threshold_js = st.sidebar.slider("Umbral JS", 0.0, 1.0, 0.2, 0.05)
threshold_chi2 = st.sidebar.slider("Umbral Chi2 (p-value)", 0.0, 0.1, 0.05, 0.01)

st.sidebar.markdown("---")
st.sidebar.markdown("### Carga de Datos")

# Cargar datos baseline
baseline_file = st.sidebar.file_uploader(
    "Cargar datos Baseline (CSV)", 
    type=['csv'],
    help="Dataset histórico de referencia"
)

current_file = st.sidebar.file_uploader(
    "Cargar datos Actuales (CSV)",
    type=['csv'],
    help="Dataset actual para comparar"
)

# Opción alternativa: usar datos guardados
use_saved_data = st.sidebar.checkbox("Usar datos guardados", value=True)

baseline_data = None
current_data = None

if use_saved_data:
    try:
        # Intentar cargar datos de la Fase 2
        # Buscar en diferentes ubicaciones
        possible_paths = [
            '../../data_processed.csv',
            '../../../data_processed.csv',
            'data_processed.csv',
            '../../X_train_transformed.csv',
            '../../../X_train_transformed.csv'
        ]
        
        baseline_path = None
        for path in possible_paths:
            if os.path.exists(path):
                baseline_path = path
                break
        
        if baseline_path:
            baseline_data = pd.read_csv(baseline_path)
            st.sidebar.success("✅ Datos baseline cargados")
            
            # Para datos actuales, usar una muestra o los mismos datos (simulación)
            # En producción, aquí cargarías datos reales actuales
            current_data = baseline_data.sample(frac=0.5, random_state=42)
            st.sidebar.info("ℹ️ Usando muestra de datos baseline como datos actuales (simulación)")
        else:
            st.sidebar.warning("⚠️ No se encontraron datos guardados. Carga archivos manualmente.")
            
    except Exception as e:
        st.sidebar.error(f"❌ Error al cargar datos: {str(e)}")
        baseline_data = None
        current_data = None
else:
    if baseline_file is not None and current_file is not None:
        try:
            baseline_data = pd.read_csv(baseline_file)
            current_data = pd.read_csv(current_file)
            st.sidebar.success("✅ Datos cargados")
        except Exception as e:
            st.sidebar.error(f"❌ Error al leer archivos: {str(e)}")
            baseline_data = None
            current_data = None
    elif baseline_file is not None:
        st.sidebar.warning("⚠️ Falta cargar el archivo de datos actuales")
    elif current_file is not None:
        st.sidebar.warning("⚠️ Falta cargar el archivo de datos baseline")

# Main content
if baseline_data is not None and current_data is not None:
    # Calcular drift
    with st.spinner("Calculando métricas de drift..."):
        try:
            drift_results = detect_drift(
                baseline_data, current_data,
                threshold_psi=threshold_psi,
                threshold_ks=threshold_ks,
                threshold_js=threshold_js,
                threshold_chi2=threshold_chi2
            )
        except Exception as e:
            st.error(f"Error al calcular drift: {str(e)}")
            st.stop()
    
    # Resumen general
    st.header("📊 Resumen General")
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_vars = len(drift_results)
    no_drift = len(drift_results[drift_results['overall_status'] == 'no_drift'])
    moderate_drift = len(drift_results[drift_results['overall_status'] == 'moderate_drift'])
    significant_drift = len(drift_results[drift_results['overall_status'] == 'significant_drift'])
    
    with col1:
        st.metric("Total Variables", total_vars)
    with col2:
        st.metric("Sin Drift", no_drift, delta=f"{(no_drift/total_vars*100):.1f}%")
    with col3:
        st.metric("Drift Moderado", moderate_drift, 
                 delta=f"{(moderate_drift/total_vars*100):.1f}%", delta_color="off")
    with col4:
        st.metric("Drift Significativo", significant_drift,
                 delta=f"{(significant_drift/total_vars*100):.1f}%", delta_color="inverse")
    
    # Alertas
    st.header("⚠️ Alertas")
    
    significant_vars = drift_results[drift_results['overall_status'] == 'significant_drift']
    
    if len(significant_vars) > 0:
        st.error(f"🚨 **ALERTA**: Se detectaron {len(significant_vars)} variables con drift significativo")
        st.dataframe(significant_vars[['variable', 'type', 'psi', 'ks_statistic', 'js_divergence', 'chi2_pvalue']])
        
        st.warning("⚠️ **Recomendación**: Considerar retraining del modelo")
    else:
        st.success("✅ No se detectaron drift significativos")
    
    # Tabla de resultados
    st.header("📋 Métricas de Drift por Variable")
    
    # Filtrar por tipo
    metric_type_filter = st.selectbox(
        "Filtrar por tipo de variable",
        ['Todas', 'Numéricas', 'Categóricas']
    )
    
    if metric_type_filter == 'Numéricas':
        display_results = drift_results[drift_results['type'] == 'numeric']
    elif metric_type_filter == 'Categóricas':
        display_results = drift_results[drift_results['type'] == 'categorical']
    else:
        display_results = drift_results
    
    # Mostrar tabla
    st.dataframe(display_results, use_container_width=True)
    
    # Visualizaciones
    st.header("📈 Visualizaciones")
    
    # Gráfico de estado por variable
    fig_status = px.bar(
        drift_results,
        x='variable',
        y='overall_status',
        color='overall_status',
        color_discrete_map={
            'no_drift': 'green',
            'moderate_drift': 'orange',
            'significant_drift': 'red'
        },
        title='Estado de Drift por Variable'
    )
    fig_status.update_xaxes(tickangle=45)
    st.plotly_chart(fig_status, use_container_width=True)
    
    # Gráficos de distribución para variables con drift
    st.subheader("Distribuciones: Baseline vs Actual")
    
    vars_with_drift = drift_results[
        drift_results['overall_status'].isin(['moderate_drift', 'significant_drift'])
    ]['variable'].head(6)
    
    if len(vars_with_drift) > 0:
        for var in vars_with_drift:
            if var in baseline_data.columns and var in current_data.columns:
                st.markdown(f"### {var}")
                
                baseline_var = baseline_data[var].dropna()
                current_var = current_data[var].dropna()
                
                if pd.api.types.is_numeric_dtype(baseline_data[var]):
                    # Histograma para variables numéricas
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(
                        x=baseline_var,
                        name='Baseline',
                        opacity=0.7,
                        nbinsx=30
                    ))
                    fig.add_trace(go.Histogram(
                        x=current_var,
                        name='Actual',
                        opacity=0.7,
                        nbinsx=30
                    ))
                    fig.update_layout(
                        title=f'Distribución de {var}',
                        xaxis_title=var,
                        yaxis_title='Frecuencia',
                        barmode='overlay'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    # Gráfico de barras para categóricas
                    baseline_counts = pd.Series(baseline_var).value_counts().head(10)
                    current_counts = pd.Series(current_var).value_counts().head(10)
                    
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=baseline_counts.index.astype(str),
                        y=baseline_counts.values,
                        name='Baseline',
                        opacity=0.7
                    ))
                    fig.add_trace(go.Bar(
                        x=current_counts.index.astype(str),
                        y=current_counts.values,
                        name='Actual',
                        opacity=0.7
                    ))
                    fig.update_layout(
                        title=f'Distribución de {var}',
                        xaxis_title=var,
                        yaxis_title='Frecuencia',
                        barmode='group'
                    )
                    st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No hay variables con drift para visualizar")
    
    # Recomendaciones
    st.header("💡 Recomendaciones")
    
    if len(significant_vars) > 0:
        st.warning("""
        **Acciones Recomendadas:**
        1. Revisar las variables con drift significativo
        2. Investigar causas del cambio en la distribución
        3. Considerar retraining del modelo
        4. Actualizar el dataset baseline si el cambio es válido
        """)
    elif len(drift_results[drift_results['overall_status'] == 'moderate_drift']) > 0:
        st.info("""
        **Acciones Recomendadas:**
        1. Monitorear las variables con drift moderado
        2. Revisar tendencias a lo largo del tiempo
        3. Considerar ajustes menores en el modelo
        """)
    else:
        st.success("""
        **Estado Actual:**
        - No se detectaron problemas significativos de drift
        - El modelo está funcionando con datos consistentes
        - Continuar con monitoreo regular
        """)
    
    # Footer
    st.markdown("---")
    st.markdown(f"**Última actualización**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
else:
    st.info("""
    **Instrucciones:**
    1. Activa la opción "Usar datos guardados" en el sidebar, o
    2. Carga archivos CSV manualmente:
       - Baseline: Dataset histórico de referencia
       - Actual: Dataset actual para comparar
    """)


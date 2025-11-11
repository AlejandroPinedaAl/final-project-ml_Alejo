"""
model_monitoring.py
Monitoreo del modelo y detección de data drift.

Este módulo contiene todas las funciones necesarias para:
- Calcular métricas de drift (KS, PSI, Jensen-Shannon, Chi-cuadrado)
- Detectar cambios en la distribución de datos
- Generar alertas por desviaciones significativas
- Aplicación Streamlit para visualización

Autor: Alejandro Pineda Alvarez
Proyecto: Marketing Campaign Response Prediction
"""

import pandas as pd
import numpy as np
from scipy.stats import ks_2samp, chi2_contingency
from scipy.spatial.distance import jensenshannon
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# 1. CÁLCULO DE MÉTRICAS DE DRIFT
# ============================================================================

def calculate_psi(expected, actual, bins=10):
    """
    Calcula el Population Stability Index (PSI).
    
    Args:
        expected: Distribución histórica (baseline)
        actual: Distribución actual
        bins: Número de bins para discretización
        
    Returns:
        Valor PSI
    """
    # Eliminar valores nulos
    expected = np.array(expected).flatten()
    actual = np.array(actual).flatten()
    
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    
    if len(expected) == 0 or len(actual) == 0:
        return np.nan
    
    # Crear bins basados en la distribución esperada
    min_val = min(np.min(expected), np.min(actual))
    max_val = max(np.max(expected), np.max(actual))
    
    # Crear bins uniformes
    bin_edges = np.linspace(min_val, max_val, bins + 1)
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf
    
    # Calcular frecuencias
    expected_hist, _ = np.histogram(expected, bins=bin_edges)
    actual_hist, _ = np.histogram(actual, bins=bin_edges)
    
    # Normalizar a probabilidades
    expected_pct = expected_hist / len(expected)
    actual_pct = actual_hist / len(actual)
    
    # Calcular PSI
    psi = 0
    for i in range(len(expected_pct)):
        if expected_pct[i] == 0:
            expected_pct[i] = 1e-10  # Evitar división por cero
        if actual_pct[i] == 0:
            actual_pct[i] = 1e-10
        
        psi += (actual_pct[i] - expected_pct[i]) * np.log(actual_pct[i] / expected_pct[i])
    
    return psi


def calculate_ks_test(expected, actual):
    """
    Calcula el test de Kolmogorov-Smirnov.
    
    Args:
        expected: Distribución histórica
        actual: Distribución actual
        
    Returns:
        Estadístico KS y p-value
    """
    # Eliminar valores nulos
    expected = np.array(expected).flatten()
    actual = np.array(actual).flatten()
    
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    
    if len(expected) == 0 or len(actual) == 0:
        return np.nan, np.nan
    
    statistic, p_value = ks_2samp(expected, actual)
    return statistic, p_value


def calculate_js_divergence(expected, actual, bins=50):
    """
    Calcula la divergencia de Jensen-Shannon.
    
    Args:
        expected: Distribución histórica
        actual: Distribución actual
        bins: Número de bins para discretización
        
    Returns:
        Divergencia JS (0 = iguales, 1 = completamente diferentes)
    """
    # Eliminar valores nulos
    expected = np.array(expected).flatten()
    actual = np.array(actual).flatten()
    
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    
    if len(expected) == 0 or len(actual) == 0:
        return np.nan
    
    # Crear bins
    min_val = min(np.min(expected), np.min(actual))
    max_val = max(np.max(expected), np.max(actual))
    
    bin_edges = np.linspace(min_val, max_val, bins + 1)
    
    # Calcular distribuciones
    expected_hist, _ = np.histogram(expected, bins=bin_edges, density=True)
    actual_hist, _ = np.histogram(actual, bins=bin_edges, density=True)
    
    # Normalizar
    expected_hist = expected_hist / np.sum(expected_hist)
    actual_hist = actual_hist / np.sum(actual_hist)
    
    # Calcular JS divergence
    js_div = jensenshannon(expected_hist, actual_hist)
    
    return js_div


def calculate_chi_square(expected, actual):
    """
    Calcula el test Chi-cuadrado para variables categóricas.
    
    Args:
        expected: Distribución histórica (serie o array)
        actual: Distribución actual (serie o array)
        
    Returns:
        Estadístico Chi-cuadrado, p-value, y grados de libertad
    """
    # Convertir a arrays
    expected = np.array(expected).flatten()
    actual = np.array(actual).flatten()
    
    # Eliminar nulos
    mask = ~(pd.isna(expected) | pd.isna(actual))
    expected = expected[mask]
    actual = actual[mask]
    
    if len(expected) == 0 or len(actual) == 0:
        return np.nan, np.nan, np.nan
    
    # Obtener todas las categorías únicas
    all_categories = np.unique(np.concatenate([expected, actual]))
    
    # Crear tabla de contingencia
    expected_counts = pd.Series(expected).value_counts().reindex(all_categories, fill_value=0)
    actual_counts = pd.Series(actual).value_counts().reindex(all_categories, fill_value=0)
    
    # Crear matriz de contingencia
    contingency_table = np.array([expected_counts.values, actual_counts.values])
    
    # Calcular Chi-cuadrado
    chi2, p_value, dof, expected_freq = chi2_contingency(contingency_table)
    
    return chi2, p_value, dof


def get_drift_status(metric_value, metric_type='psi', thresholds=None):
    """
    Determina el estado de drift basado en umbrales.
    
    Args:
        metric_value: Valor de la métrica
        metric_type: Tipo de métrica ('psi', 'ks', 'js', 'chi2')
        thresholds: Diccionario con umbrales personalizados
        
    Returns:
        Estado: 'no_drift', 'moderate_drift', 'significant_drift'
    """
    if thresholds is None:
        # Umbrales por defecto
        if metric_type == 'psi':
            thresholds = {'moderate': 0.1, 'significant': 0.2}
        elif metric_type == 'ks':
            thresholds = {'moderate': 0.1, 'significant': 0.2}
        elif metric_type == 'js':
            thresholds = {'moderate': 0.1, 'significant': 0.2}
        elif metric_type == 'chi2':
            thresholds = {'moderate': 0.05, 'significant': 0.01}  # p-value
        else:
            thresholds = {'moderate': 0.1, 'significant': 0.2}
    
    if pd.isna(metric_value):
        return 'unknown'
    
    if metric_type == 'chi2':
        # Para chi2, usamos p-value (menor = más significativo)
        if metric_value >= thresholds['moderate']:
            return 'no_drift'
        elif metric_value >= thresholds['significant']:
            return 'moderate_drift'
        else:
            return 'significant_drift'
    else:
        # Para otras métricas, mayor = más drift
        if metric_value < thresholds['moderate']:
            return 'no_drift'
        elif metric_value < thresholds['significant']:
            return 'moderate_drift'
        else:
            return 'significant_drift'


# ============================================================================
# 2. DETECCIÓN DE DRIFT EN DATASET COMPLETO
# ============================================================================

def detect_drift(baseline_data, current_data, threshold_psi=0.2, 
                threshold_ks=0.2, threshold_js=0.2, threshold_chi2=0.05):
    """
    Detecta drift en los datos comparando baseline vs actual.
    
    Args:
        baseline_data: DataFrame con datos históricos
        current_data: DataFrame con datos actuales
        threshold_psi: Umbral de PSI para alertas
        threshold_ks: Umbral de KS para alertas
        threshold_js: Umbral de JS para alertas
        threshold_chi2: Umbral de p-value Chi2 para alertas
        
    Returns:
        DataFrame con métricas de drift por variable
    """
    drift_results = []
    
    # Obtener columnas comunes
    common_cols = set(baseline_data.columns) & set(current_data.columns)
    
    for col in common_cols:
        baseline_col = baseline_data[col].dropna()
        current_col = current_data[col].dropna()
        
        if len(baseline_col) == 0 or len(current_col) == 0:
            continue
        
        # Determinar tipo de variable
        is_numeric = pd.api.types.is_numeric_dtype(baseline_data[col])
        is_categorical = pd.api.types.is_categorical_dtype(baseline_data[col]) or \
                        baseline_data[col].dtype == 'object'
        
        result = {
            'variable': col,
            'type': 'numeric' if is_numeric else 'categorical',
            'baseline_size': len(baseline_col),
            'current_size': len(current_col)
        }
        
        # Calcular métricas según tipo
        if is_numeric:
            # PSI
            psi = calculate_psi(baseline_col, current_col)
            result['psi'] = psi
            result['psi_status'] = get_drift_status(psi, 'psi', 
                                                    {'moderate': 0.1, 'significant': threshold_psi})
            
            # KS Test
            ks_stat, ks_pvalue = calculate_ks_test(baseline_col, current_col)
            result['ks_statistic'] = ks_stat
            result['ks_pvalue'] = ks_pvalue
            result['ks_status'] = get_drift_status(ks_stat, 'ks',
                                                   {'moderate': 0.1, 'significant': threshold_ks})
            
            # JS Divergence
            js_div = calculate_js_divergence(baseline_col, current_col)
            result['js_divergence'] = js_div
            result['js_status'] = get_drift_status(js_div, 'js',
                                                   {'moderate': 0.1, 'significant': threshold_js})
            
            # Chi2 no aplica para numéricas
            result['chi2_statistic'] = np.nan
            result['chi2_pvalue'] = np.nan
            result['chi2_status'] = 'N/A'
        
        elif is_categorical:
            # Chi2 Test
            chi2_stat, chi2_pvalue, dof = calculate_chi_square(baseline_col, current_col)
            result['chi2_statistic'] = chi2_stat
            result['chi2_pvalue'] = chi2_pvalue
            result['chi2_dof'] = dof
            result['chi2_status'] = get_drift_status(chi2_pvalue, 'chi2',
                                                     {'moderate': threshold_chi2, 'significant': 0.01})
            
            # Para categóricas, podemos convertir a numérico y calcular PSI
            try:
                baseline_encoded = pd.Categorical(baseline_col).codes
                current_encoded = pd.Categorical(current_col).codes
                psi = calculate_psi(baseline_encoded, current_encoded)
                result['psi'] = psi
                result['psi_status'] = get_drift_status(psi, 'psi',
                                                        {'moderate': 0.1, 'significant': threshold_psi})
            except:
                result['psi'] = np.nan
                result['psi_status'] = 'N/A'
            
            # KS y JS no aplican directamente a categóricas
            result['ks_statistic'] = np.nan
            result['ks_pvalue'] = np.nan
            result['ks_status'] = 'N/A'
            result['js_divergence'] = np.nan
            result['js_status'] = 'N/A'
        
        drift_results.append(result)
    
    df_drift = pd.DataFrame(drift_results)
    
    # Calcular estado general por variable
    df_drift['overall_status'] = df_drift.apply(
        lambda row: determine_overall_status(row), axis=1
    )
    
    return df_drift


def determine_overall_status(row):
    """
    Determina el estado general de drift para una variable.
    
    Args:
        row: Fila del DataFrame de drift results
        
    Returns:
        Estado general: 'no_drift', 'moderate_drift', 'significant_drift'
    """
    statuses = []
    
    if pd.notna(row.get('psi_status')) and row.get('psi_status') != 'N/A':
        statuses.append(row['psi_status'])
    if pd.notna(row.get('ks_status')) and row.get('ks_status') != 'N/A':
        statuses.append(row['ks_status'])
    if pd.notna(row.get('js_status')) and row.get('js_status') != 'N/A':
        statuses.append(row['js_status'])
    if pd.notna(row.get('chi2_status')) and row.get('chi2_status') != 'N/A':
        statuses.append(row['chi2_status'])
    
    if not statuses:
        return 'unknown'
    
    # Si cualquier métrica indica drift significativo, el estado es significativo
    if 'significant_drift' in statuses:
        return 'significant_drift'
    elif 'moderate_drift' in statuses:
        return 'moderate_drift'
    else:
        return 'no_drift'


# ============================================================================
# 3. APLICACIÓN STREAMLIT
# ============================================================================

def create_streamlit_app():
    """
    Crea la aplicación Streamlit para monitoreo de drift.
    Esta función debe ser llamada desde un archivo .py separado para Streamlit.
    """
    import streamlit as st
    import plotly.express as px
    import plotly.graph_objects as go
    from datetime import datetime
    
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
    
    if use_saved_data:
        try:
            # Intentar cargar datos de la Fase 2
            baseline_data = pd.read_csv('../../data_processed.csv')
            st.sidebar.success("✅ Datos baseline cargados")
            
            # Para datos actuales, usar una muestra o los mismos datos (simulación)
            current_data = baseline_data.sample(frac=0.5, random_state=42)
            st.sidebar.info("ℹ️ Usando muestra de datos baseline como datos actuales (simulación)")
            
        except FileNotFoundError:
            st.sidebar.error("❌ No se encontraron datos guardados. Carga archivos manualmente.")
            baseline_data = None
            current_data = None
    else:
        if baseline_file is not None and current_file is not None:
            baseline_data = pd.read_csv(baseline_file)
            current_data = pd.read_csv(current_file)
            st.sidebar.success("✅ Datos cargados")
        else:
            baseline_data = None
            current_data = None
    
    # Main content
    if baseline_data is not None and current_data is not None:
        # Calcular drift
        with st.spinner("Calculando métricas de drift..."):
            drift_results = detect_drift(
                baseline_data, current_data,
                threshold_psi=threshold_psi,
                threshold_ks=threshold_ks,
                threshold_js=threshold_js,
                threshold_chi2=threshold_chi2
            )
        
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
                            x=baseline_counts.index,
                            y=baseline_counts.values,
                            name='Baseline',
                            opacity=0.7
                        ))
                        fig.add_trace(go.Bar(
                            x=current_counts.index,
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


# ============================================================================
# EJEMPLO DE USO
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("MODEL MONITORING - DATA DRIFT DETECTION")
    print("="*80)
    
    print("\nPara ejecutar la aplicación Streamlit:")
    print("  streamlit run model_monitoring.py")
    
    print("\nPara usar las funciones directamente:")
    print("  from model_monitoring import detect_drift, calculate_psi, calculate_ks_test")


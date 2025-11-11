"""
ft_engineering.py
Ingeniería de Características y Preparación de Datos para Modelado.

Este módulo contiene todas las funciones necesarias para:
- Limpieza de datos (nulos, outliers, inconsistencias)
- Creación de features derivados
- Transformación de variables (escalado, encoding)
- Split de datos (train/test estratificado)

Autor: Alejandro Pineda Alvarez
Proyecto: Marketing Campaign Response Prediction
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import json
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# 1. LIMPIEZA DE DATOS
# ============================================================================

def clean_data(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Limpia el dataset: maneja nulos, outliers e inconsistencias.
    
    Args:
        df: DataFrame de entrada
        verbose: Si True, imprime información del proceso
        
    Returns:
        DataFrame limpio
    """
    df_clean = df.copy()
    
    if verbose:
        print("\n" + "="*80)
        print("🧹 LIMPIEZA DE DATOS")
        print("="*80)
        print(f"Dimensiones iniciales: {df_clean.shape}")
    
    # 1. Eliminar variables irrelevantes
    cols_to_drop = ['ID', 'Z_CostContact', 'Z_Revenue']
    cols_to_drop = [col for col in cols_to_drop if col in df_clean.columns]
    if cols_to_drop:
        df_clean = df_clean.drop(columns=cols_to_drop)
        if verbose:
            print(f"\n✅ Variables eliminadas: {cols_to_drop}")
    
    # 2. Convertir tipos de datos
    if 'Dt_Customer' in df_clean.columns:
        df_clean['Dt_Customer'] = pd.to_datetime(df_clean['Dt_Customer'], format='%Y-%m-%d', errors='coerce')
    
    # 3. Manejo de valores nulos en Income
    if 'Income' in df_clean.columns:
        nulos_income = df_clean['Income'].isnull().sum()
        if nulos_income > 0:
            if verbose:
                print(f"\n⚠️ Valores nulos en Income: {nulos_income} ({nulos_income/len(df_clean)*100:.2f}%)")
            
            # Imputar con mediana (robusto a outliers)
            median_income = df_clean['Income'].median()
            df_clean['Income'] = df_clean['Income'].fillna(median_income)
            
            if verbose:
                print(f"✅ Imputados con mediana: {median_income:.2f}")
    
    # 4. Manejo de outliers extremos en Income (opcional: usar IQR)
    if 'Income' in df_clean.columns:
        Q1 = df_clean['Income'].quantile(0.25)
        Q3 = df_clean['Income'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 3 * IQR  # 3 IQR para outliers extremos
        upper_bound = Q3 + 3 * IQR
        
        outliers_count = ((df_clean['Income'] < lower_bound) | (df_clean['Income'] > upper_bound)).sum()
        
        if outliers_count > 0 and verbose:
            print(f"\n⚠️ Outliers extremos en Income: {outliers_count}")
            print(f"   Rango normal: [{lower_bound:.0f}, {upper_bound:.0f}]")
            # Nota: No eliminamos outliers, solo los reportamos
            # El RobustScaler manejará esto en la transformación
    
    # 5. Unificar categorías en Education
    if 'Education' in df_clean.columns:
        # Mapeo de categorías similares
        education_mapping = {
            '2n Cycle': 'Undergraduate',
            'Basic': 'Basic',
            'Graduation': 'Graduate',
            'Master': 'Postgraduate',
            'PhD': 'Postgraduate'
        }
        df_clean['Education'] = df_clean['Education'].map(education_mapping)
        
        if verbose:
            print(f"\n✅ Education unificado: {df_clean['Education'].value_counts().to_dict()}")
    
    # 6. Unificar categorías en Marital_Status
    if 'Marital_Status' in df_clean.columns:
        # Agrupar categorías poco frecuentes
        marital_mapping = {
            'Single': 'Single',
            'Together': 'Relationship',
            'Married': 'Relationship',
            'Divorced': 'Single',
            'Widow': 'Single',
            'Alone': 'Single',
            'Absurd': 'Other',
            'YOLO': 'Other'
        }
        df_clean['Marital_Status'] = df_clean['Marital_Status'].map(
            lambda x: marital_mapping.get(x, 'Other')
        )
        
        if verbose:
            print(f"✅ Marital_Status unificado: {df_clean['Marital_Status'].value_counts().to_dict()}")
    
    if verbose:
        print(f"\nDimensiones finales: {df_clean.shape}")
        print("="*80)
    
    return df_clean


# ============================================================================
# 2. CREACIÓN DE FEATURES DERIVADOS
# ============================================================================

def create_derived_features(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Crea features derivados a partir de variables existentes.
    
    Args:
        df: DataFrame de entrada
        verbose: Si True, imprime información del proceso
        
    Returns:
        DataFrame con features derivados
    """
    df_features = df.copy()
    
    if verbose:
        print("\n" + "="*80)
        print("🔧 CREACIÓN DE FEATURES DERIVADOS")
        print("="*80)
    
    # 1. Total Spent (suma de todos los gastos)
    gastos_cols = ['MntWines', 'MntFruits', 'MntMeatProducts', 
                   'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
    gastos_cols = [col for col in gastos_cols if col in df_features.columns]
    
    if gastos_cols:
        df_features['TotalSpent'] = df_features[gastos_cols].sum(axis=1)
        if verbose:
            print(f"✅ TotalSpent creado (suma de {len(gastos_cols)} variables de gasto)")
    
    # 2. Total Purchases (suma de todas las compras)
    purchases_cols = ['NumDealsPurchases', 'NumWebPurchases', 
                     'NumCatalogPurchases', 'NumStorePurchases']
    purchases_cols = [col for col in purchases_cols if col in df_features.columns]
    
    if purchases_cols:
        df_features['TotalPurchases'] = df_features[purchases_cols].sum(axis=1)
        if verbose:
            print(f"✅ TotalPurchases creado (suma de {len(purchases_cols)} canales)")
    
    # 3. Average Purchase Value
    if 'TotalSpent' in df_features.columns and 'TotalPurchases' in df_features.columns:
        df_features['AvgPurchaseValue'] = df_features['TotalSpent'] / (df_features['TotalPurchases'] + 1)
        if verbose:
            print("✅ AvgPurchaseValue creado (TotalSpent / TotalPurchases)")
    
    # 4. Total Accepted Campaigns
    campaigns_cols = ['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 
                     'AcceptedCmp4', 'AcceptedCmp5']
    campaigns_cols = [col for col in campaigns_cols if col in df_features.columns]
    
    if campaigns_cols:
        df_features['TotalCampaignsAccepted'] = df_features[campaigns_cols].sum(axis=1)
        if verbose:
            print(f"✅ TotalCampaignsAccepted creado (suma de {len(campaigns_cols)} campañas)")
    
    # 5. Has Children
    if 'Kidhome' in df_features.columns and 'Teenhome' in df_features.columns:
        df_features['HasChildren'] = ((df_features['Kidhome'] + df_features['Teenhome']) > 0).astype(int)
        if verbose:
            print("✅ HasChildren creado (indicador binario)")
    
    # 6. Total Children
    if 'Kidhome' in df_features.columns and 'Teenhome' in df_features.columns:
        df_features['TotalChildren'] = df_features['Kidhome'] + df_features['Teenhome']
        if verbose:
            print("✅ TotalChildren creado (Kidhome + Teenhome)")
    
    # 7. Age (desde Year_Birth)
    if 'Year_Birth' in df_features.columns:
        current_year = 2014  # Año de referencia del dataset
        df_features['Age'] = current_year - df_features['Year_Birth']
        if verbose:
            print(f"✅ Age creado (edad calculada desde Year_Birth)")
    
    # 8. Customer Tenure (días desde inscripción)
    if 'Dt_Customer' in df_features.columns:
        reference_date = df_features['Dt_Customer'].max()
        df_features['CustomerTenure'] = (reference_date - df_features['Dt_Customer']).dt.days
        if verbose:
            print("✅ CustomerTenure creado (días desde inscripción)")
    
    # 9. Web Engagement (tasa de conversión web)
    if 'NumWebPurchases' in df_features.columns and 'NumWebVisitsMonth' in df_features.columns:
        df_features['WebEngagement'] = df_features['NumWebPurchases'] / (df_features['NumWebVisitsMonth'] + 1)
        if verbose:
            print("✅ WebEngagement creado (NumWebPurchases / NumWebVisitsMonth)")
    
    # 10. Income per Person
    if 'Income' in df_features.columns and 'TotalChildren' in df_features.columns:
        df_features['IncomePerPerson'] = df_features['Income'] / (1 + df_features['TotalChildren'])
        if verbose:
            print("✅ IncomePerPerson creado (Income / (1 + TotalChildren))")
    
    # 11. Spending Ratio (proporción de ingreso gastado)
    if 'TotalSpent' in df_features.columns and 'Income' in df_features.columns:
        df_features['SpendingRatio'] = df_features['TotalSpent'] / (df_features['Income'] + 1)
        if verbose:
            print("✅ SpendingRatio creado (TotalSpent / Income)")
    
    # 12. Days Since Last Purchase (inverso de Recency para mejor interpretación)
    if 'Recency' in df_features.columns:
        df_features['DaysSinceLastPurchase'] = df_features['Recency']
        if verbose:
            print("✅ DaysSinceLastPurchase creado (alias de Recency)")
    
    if verbose:
        print(f"\nTotal de features derivados creados: 12")
        print("="*80)
    
    return df_features


# ============================================================================
# 3. PREPARACIÓN DE VARIABLES PARA MODELADO
# ============================================================================

def prepare_features_for_modeling(df: pd.DataFrame, target_col: str = 'Response', 
                                  verbose: bool = True) -> tuple:
    """
    Prepara las features para modelado: separa X e y, identifica tipos de variables.
    
    Args:
        df: DataFrame con features
        target_col: Nombre de la columna objetivo
        verbose: Si True, imprime información
        
    Returns:
        tuple: (X, y, numeric_features, categorical_features)
    """
    df_model = df.copy()
    
    if verbose:
        print("\n" + "="*80)
        print("📊 PREPARACIÓN DE FEATURES PARA MODELADO")
        print("="*80)
    
    # Separar X e y
    if target_col not in df_model.columns:
        raise ValueError(f"Columna objetivo '{target_col}' no encontrada en el dataset")
    
    y = df_model[target_col]
    X = df_model.drop(columns=[target_col])
    
    # Eliminar columnas que no se usarán en el modelo
    cols_to_drop = ['Dt_Customer', 'Year_Birth']  # Ya creamos features derivados
    cols_to_drop = [col for col in cols_to_drop if col in X.columns]
    if cols_to_drop:
        X = X.drop(columns=cols_to_drop)
    
    # Identificar tipos de variables
    numeric_features = X.select_dtypes(include=['int64', 'float64', 'int8']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if verbose:
        print(f"\nDimensiones de X: {X.shape}")
        print(f"Dimensiones de y: {y.shape}")
        print(f"\nVariables numéricas ({len(numeric_features)}): {numeric_features[:5]}...")
        print(f"Variables categóricas ({len(categorical_features)}): {categorical_features}")
        print(f"\nBalance de clases en y:")
        print(y.value_counts())
        print(f"Proporción: {y.value_counts(normalize=True).to_dict()}")
        print("="*80)
    
    return X, y, numeric_features, categorical_features


# ============================================================================
# 4. CREACIÓN DE PIPELINES DE TRANSFORMACIÓN
# ============================================================================

def create_preprocessing_pipeline(numeric_features: list, categorical_features: list,
                                 use_robust_scaler: bool = True) -> ColumnTransformer:
    """
    Crea un pipeline de preprocesamiento con transformaciones para variables numéricas y categóricas.
    
    Args:
        numeric_features: Lista de nombres de features numéricas
        categorical_features: Lista de nombres de features categóricas
        use_robust_scaler: Si True, usa RobustScaler (mejor para outliers), si no StandardScaler
        
    Returns:
        ColumnTransformer configurado
    """
    
    # Pipeline para variables numéricas
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),  # Imputar nulos con mediana
        ('scaler', RobustScaler() if use_robust_scaler else StandardScaler())  # Escalado
    ])
    
    # Pipeline para variables categóricas
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),  # Imputar nulos
        ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))  # One-hot encoding
    ])
    
    # Combinar pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'  # Eliminar columnas no especificadas
    )
    
    return preprocessor


# ============================================================================
# 5. SPLIT DE DATOS
# ============================================================================

def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, 
               random_state: int = 42, stratify: bool = True, 
               verbose: bool = True) -> tuple:
    """
    Divide los datos en conjuntos de entrenamiento y prueba.
    
    Args:
        X: Features
        y: Variable objetivo
        test_size: Proporción del conjunto de prueba (default: 0.2)
        random_state: Semilla aleatoria para reproducibilidad
        stratify: Si True, mantiene la proporción de clases (recomendado para datasets desbalanceados)
        verbose: Si True, imprime información
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    
    if verbose:
        print("\n" + "="*80)
        print("✂️ SPLIT DE DATOS")
        print("="*80)
    
    stratify_param = y if stratify else None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=stratify_param
    )
    
    if verbose:
        print(f"\nTest size: {test_size*100:.0f}%")
        print(f"Random state: {random_state}")
        print(f"Estratificación: {'Sí' if stratify else 'No'}")
        print(f"\nDimensiones:")
        print(f"  X_train: {X_train.shape}")
        print(f"  X_test:  {X_test.shape}")
        print(f"  y_train: {y_train.shape}")
        print(f"  y_test:  {y_test.shape}")
        
        print(f"\nDistribución de clases:")
        print(f"  Train: {y_train.value_counts().to_dict()}")
        print(f"  Test:  {y_test.value_counts().to_dict()}")
        print(f"\n  Proporción train: {y_train.value_counts(normalize=True).to_dict()}")
        print(f"  Proporción test:  {y_test.value_counts(normalize=True).to_dict()}")
        print("="*80)
    
    return X_train, X_test, y_train, y_test


# ============================================================================
# 6. FUNCIÓN PRINCIPAL - PIPELINE COMPLETO
# ============================================================================

def run_feature_engineering_pipeline(data_path: str = None, df: pd.DataFrame = None,
                                    test_size: float = 0.2, random_state: int = 42,
                                    use_robust_scaler: bool = True,
                                    save_preprocessor: bool = True,
                                    verbose: bool = True) -> dict:
    """
    Ejecuta el pipeline completo de ingeniería de características.
    
    Args:
        data_path: Ruta al archivo CSV (si df es None)
        df: DataFrame de entrada (si data_path es None)
        test_size: Proporción del conjunto de prueba
        random_state: Semilla aleatoria
        use_robust_scaler: Si True, usa RobustScaler
        save_preprocessor: Si True, guarda el preprocessor
        verbose: Si True, imprime información detallada
        
    Returns:
        dict con todos los componentes:
            - X_train, X_test, y_train, y_test: Datos divididos
            - preprocessor: Pipeline de transformación
            - numeric_features, categorical_features: Listas de features
            - df_processed: DataFrame procesado completo
    """
    
    if verbose:
        print("\n" + "="*80)
        print("🚀 PIPELINE DE INGENIERÍA DE CARACTERÍSTICAS")
        print("="*80)
    
    # 1. Cargar datos
    if df is None:
        if data_path is None:
            raise ValueError("Debe proporcionar data_path o df")
        df = pd.read_csv(data_path, sep=';')
        if verbose:
            print(f"\n✅ Datos cargados desde: {data_path}")
            print(f"   Dimensiones: {df.shape}")
    
    # 2. Limpieza de datos
    df_clean = clean_data(df, verbose=verbose)
    
    # 3. Creación de features derivados
    df_features = create_derived_features(df_clean, verbose=verbose)
    
    # 4. Preparación para modelado
    X, y, numeric_features, categorical_features = prepare_features_for_modeling(
        df_features, target_col='Response', verbose=verbose
    )
    
    # 5. Split de datos
    X_train, X_test, y_train, y_test = split_data(
        X, y, test_size=test_size, random_state=random_state, 
        stratify=True, verbose=verbose
    )
    
    # 6. Crear preprocessor
    if verbose:
        print("\n" + "="*80)
        print("🔧 CREACIÓN DE PREPROCESSOR")
        print("="*80)
        print(f"\nScaler: {'RobustScaler' if use_robust_scaler else 'StandardScaler'}")
        print(f"Encoding: OneHotEncoder (drop='first')")
    
    preprocessor = create_preprocessing_pipeline(
        numeric_features, categorical_features, use_robust_scaler
    )
    
    # 7. Guardar preprocessor (opcional)
    if save_preprocessor:
        import joblib
        joblib.dump(preprocessor, 'preprocessor.pkl')
        if verbose:
            print(f"\n✅ Preprocessor guardado en: preprocessor.pkl")
    
    if verbose:
        print("\n" + "="*80)
        print("✅ PIPELINE COMPLETADO EXITOSAMENTE")
        print("="*80)
    
    # Retornar todos los componentes
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'preprocessor': preprocessor,
        'numeric_features': numeric_features,
        'categorical_features': categorical_features,
        'df_processed': df_features,
        'feature_names': X.columns.tolist()
    }


# ============================================================================
# 7. FUNCIÓN DE UTILIDAD - OBTENER NOMBRES DE FEATURES TRANSFORMADOS
# ============================================================================

def get_feature_names_after_preprocessing(preprocessor, numeric_features, categorical_features):
    """
    Obtiene los nombres de las features después de aplicar el preprocessor.
    
    Args:
        preprocessor: ColumnTransformer fitted
        numeric_features: Lista de features numéricas originales
        categorical_features: Lista de features categóricas originales
        
    Returns:
        Lista con nombres de todas las features transformadas
    """
    feature_names = []
    
    # Features numéricas (mantienen su nombre)
    feature_names.extend(numeric_features)
    
    # Features categóricas (one-hot encoded)
    if len(categorical_features) > 0:
        cat_encoder = preprocessor.named_transformers_['cat']['onehot']
        cat_feature_names = cat_encoder.get_feature_names_out(categorical_features)
        feature_names.extend(cat_feature_names)
    
    return feature_names


# ============================================================================
# EJEMPLO DE USO
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("EJEMPLO DE USO - FEATURE ENGINEERING PIPELINE")
    print("="*80)
    
    # Cargar configuración
    with open('../../config.json', 'r') as f:
        config = json.load(f)
    
    data_path = f'../../{config["data_path"]}'
    
    # Ejecutar pipeline completo
    results = run_feature_engineering_pipeline(
        data_path=data_path,
        test_size=0.2,
        random_state=42,
        use_robust_scaler=True,
        save_preprocessor=True,
        verbose=True
    )
    
    print("\n" + "="*80)
    print("📦 COMPONENTES DISPONIBLES:")
    print("="*80)
    print(f"  - X_train: {results['X_train'].shape}")
    print(f"  - X_test: {results['X_test'].shape}")
    print(f"  - y_train: {results['y_train'].shape}")
    print(f"  - y_test: {results['y_test'].shape}")
    print(f"  - preprocessor: {type(results['preprocessor'])}")
    print(f"  - numeric_features: {len(results['numeric_features'])} features")
    print(f"  - categorical_features: {len(results['categorical_features'])} features")
    print(f"  - df_processed: {results['df_processed'].shape}")
    print(f"  - feature_names: {len(results['feature_names'])} features")
    
    print("\n✅ Pipeline ejecutado exitosamente!")
    print("="*80)

"""
ft_engineering.py
Generación de features y creación de datasets.
Contenido: funciones para ingeniería de features y pipelines de transformación.
"""

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


def create_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Crea features derivados desde el dataset base.

    Args:
        df: dataframe de entrada

    Returns:
        dataframe con nuevas features derivados
    """
    df = df.copy()
    
    # TODO: Implementar features derivados
    # Ejemplos:
    # - TotalSpent: suma de todos los Mnt*
    # - TotalPurchases: suma de todos los Num*Purchases
    # - TotalAcceptedCampaigns: suma de AcceptedCmp1-5
    # - HasChildren: Kidhome + Teenhome > 0
    # - CustomerAge: días desde DtCustomer
    
    return df


def create_preprocessing_pipeline():
    """Crea el pipeline de preprocesamiento de datos.
    
    Returns:
        Pipeline de sklearn configurado
    """
    # TODO: Implementar pipeline de preprocesamiento
    # - Imputación de nulos
    # - Escalado de variables numéricas
    # - Codificación de variables categóricas
    
    pass


def split_data(X, y, test_size=0.2, random_state=42):
    """Separa los datos en conjuntos de entrenamiento y prueba.
    
    Args:
        X: Features
        y: Variable objetivo
        test_size: Proporción del conjunto de prueba
        random_state: Semilla aleatoria
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    return train_test_split(X, y, test_size=test_size, 
                          random_state=random_state, 
                          stratify=y)

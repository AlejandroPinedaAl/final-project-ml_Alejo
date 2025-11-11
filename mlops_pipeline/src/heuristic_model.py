"""
heuristic_model.py
Modelo heurístico base: ejemplo de una predicción simple sin entrenamiento.
"""

import numpy as np


def predict_heuristic(X):
    """Devuelve predicciones heurísticas (placeholder).

    Args:
        X: array-like o dataframe de features
    Returns:
        numpy array de predicciones
    """
    n = len(X)
    # Heurística: predecir 0 para todos (placeholder)
    return np.zeros(n, dtype=int)

"""
model_deploy.py
Despliegue del modelo como API REST usando FastAPI.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from typing import List, Dict

app = FastAPI(
    title="Marketing Campaign Prediction API",
    description="API para predecir respuesta de clientes a campañas de marketing",
    version="1.0.0"
)

# TODO: Cargar modelo entrenado
# model = joblib.load('best_model.pkl')


class CustomerData(BaseModel):
    """Modelo de datos para un cliente individual."""
    # TODO: Definir campos según features del modelo
    Income: float
    Recency: int
    # ... agregar todos los campos necesarios


class PredictionResponse(BaseModel):
    """Modelo de respuesta de predicción."""
    prediction: int
    probability: float
    

@app.get("/")
def root():
    """Endpoint raíz con información de la API."""
    return {
        "message": "Marketing Campaign Prediction API",
        "version": "1.0.0",
        "endpoints": ["/health", "/predict", "/predict_batch"]
    }


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": False}  # TODO: verificar modelo


@app.post("/predict", response_model=PredictionResponse)
def predict_single(data: CustomerData):
    """Predice la respuesta para un cliente individual.
    
    Args:
        data: Datos del cliente
        
    Returns:
        Predicción y probabilidad
    """
    # TODO: Implementar predicción individual
    pass


@app.post("/predict_batch")
def predict_batch(file: UploadFile = File(...)):
    """Predice la respuesta para múltiples clientes desde CSV.
    
    Args:
        file: Archivo CSV con datos de clientes
        
    Returns:
        Lista de predicciones
    """
    # TODO: Implementar predicción por lotes
    pass

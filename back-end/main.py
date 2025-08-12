import os
import uvicorn
import json
import threading
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from services.ensemble import predict_ensemble
from core.models import carregar_modelos

# Variável global para armazenar as métricas de avaliação (serão populadas pelo script de avaliação)
evaluation_metrics = {}

# Classe Pydantic para validar a entrada da API de previsão
class Municipio(BaseModel):
    municipio: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Lifespan: Iniciando aplicação...")
    
    # Carrega os modelos treinados na inicialização
    carregar_modelos()
    
    yield
    print("Lifespan: Desligando aplicação...")

app = FastAPI(lifespan=lifespan)

# Endpoint raiz para verificar se a API está online
@app.get("/")
def read_root():
    return {"status": "API de Previsão de Enchente está online!"}

# Endpoint para fazer a previsão de enchente para um município
@app.post("/predict/")
def predict_flood(municipio: Municipio):
    prediction = predict_ensemble(municipio.municipio)
    if "error" in prediction:
        raise HTTPException(status_code=400, detail=prediction["error"])
    return prediction

# Endpoint para obter as métricas de avaliação do último ciclo
@app.get("/evaluate/")
def get_evaluation():
    global evaluation_metrics
    if evaluation_metrics:
        return evaluation_metrics
    raise HTTPException(status_code=404, detail="Métricas de avaliação não disponíveis ainda.")

if __name__ == "__main__":
    # --- Execute o treinamento acelerado UMA ÚNICA VEZ antes de rodar a API ---
    # Para rodar o treinamento, você deve executar `python3 treinamento_acelerado.py` separadamente.
    # Depois, para rodar a API, use o comando abaixo:
    uvicorn.run(app, host="0.0.0.0", port=8000)
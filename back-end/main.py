import os
import uvicorn
import json
import threading
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from services.ensemble import predict_ensemble
from core.models import carregar_modelos
from fastapi.middleware.cors import CORSMiddleware # Adiciona esta importação

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

# === Configuração do CORS Middleware ===
# Isso permite que seu front-end (em localhost:5173) acesse a API.
origins = [
    "http://localhost",
    "http://localhost:5173", # Seu front-end React
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ========================================

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

@app.get("/estacoes/")
def get_estacoes():
    try:
        caminho_catalogo = os.path.join(os.path.dirname(__file__), '..', 'dados', 'catalogoestacoesautomaticas.csv')
        df = pd.read_csv(caminho_catalogo, sep=';')
        
        # Mapeamento e normalização das colunas
        df.columns = df.columns.str.strip().str.upper().str.replace(' ', '_').str.replace('.', '')
        
        # Seleciona as colunas relevantes e renomeia para o front-end
        df_json = df[['DC_NOME', 'VL_LATITUDE', 'VL_LONGITUDE']].rename(columns={
            'DC_NOME': 'nome',
            'VL_LATITUDE': 'lat',
            'VL_LONGITUDE': 'lon'
        })
        
        # Converte para um formato de lista de dicionários para o React
        estacoes = df_json.to_dict('records')
        return {"estacoes": estacoes}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Arquivo de estações não encontrado.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao carregar estações: {e}")

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
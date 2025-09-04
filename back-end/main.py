import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from services.ensemble import predict_ensemble, predict_historical
from services.weather import get_weather_data
from core.models import carregar_modelos

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

df_estacoes = None

@app.on_event("startup")
async def load_data():
    global df_estacoes
    try:
        df_estacoes = pd.read_csv("dados/catalogoestacoesautomaticas.csv", sep=';', decimal=',', dtype={'VL_LATITUDE': float, 'VL_LONGITUDE': float})
        print("Dados de estações carregados com sucesso!")
        carregar_modelos()
    except FileNotFoundError as e:
        print(f"Erro: Arquivo não encontrado. Verifique os caminhos dos arquivos: {e}")
    except Exception as e:
        print(f"Erro ao carregar os dados: {e}")

@app.get("/estacoes/")
async def get_estacoes():
    if df_estacoes is not None:
        df_estacoes.rename(columns={'DC_NOME': 'municipio', 'VL_LATITUDE': 'lat', 'VL_LONGITUDE': 'lon'}, inplace=True)
        df_estacoes = df_estacoes.dropna(subset=['lat', 'lon'])
        return {"estacoes": df_estacoes.to_dict(orient='records')}
    return {"estacoes": []}

@app.get("/predict/{municipio}")
async def get_prediction(municipio: str):
    return predict_ensemble(municipio)

@app.get("/predict/history/{municipio}")
async def get_history(municipio: str):
    return predict_historical(municipio)

@app.get("/evaluate/")
async def get_evaluation():
    return {"status": "ok", "message": "Avaliacao de modelo nao carregada de arquivo."}
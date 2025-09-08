# --- START OF FILE main.py ---
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from services.ensemble import predict_ensemble, predict_historical
from core.models import carregar_modelos
import json # Importar json
import os # Importar os para checar arquivo

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

df_estacoes = None
evaluation_metrics_data = None # Variável para armazenar as métricas de avaliação

@app.on_event("startup")
async def load_data_and_models():
    global df_estacoes, evaluation_metrics_data
    try:
        # Carrega dados das estações
        # O arquivo catalogoestacoesautomaticas.csv usa ; como separador e , como decimal
        df_estacoes = pd.read_csv(
            "dados/catalogoestacoesautomaticas.csv",
            sep=';',
            decimal=',',
            dtype={'VL_LATITUDE': float, 'VL_LONGITUDE': float}
        )
        print("Dados de estações carregados com sucesso!")
        
        # Carrega os modelos de Machine Learning
        carregar_modelos()

        # Carrega as métricas de avaliação se o arquivo existir
        evaluation_file = 'evaluation_metrics.json'
        if os.path.exists(evaluation_file):
            with open(evaluation_file, 'r') as f:
                evaluation_metrics_data = json.load(f)
            print(f"Métricas de avaliação carregadas de {evaluation_file}")
        else:
            print(f"AVISO: Arquivo de métricas de avaliação '{evaluation_file}' não encontrado.")

    except FileNotFoundError as e:
        print(f"ERRO: Arquivo não encontrado. Verifique os caminhos dos arquivos: {e}")
    except Exception as e:
        print(f"ERRO ao carregar dados ou modelos: {e}")

@app.get("/estacoes/")
async def get_estacoes():
    if df_estacoes is not None:
        # Filtra estações que têm um nome e coordenadas válidas
        valid_estacoes = df_estacoes.dropna(subset=['DC_NOME', 'VL_LATITUDE', 'VL_LONGITUDE'])
        
        # Renomeia as colunas para o formato esperado pelo frontend
        # Certifique-se de que a coluna 'municipio' usada no frontend é 'DC_NOME' aqui.
        estacoes_list = valid_estacoes.rename(columns={
            'DC_NOME': 'nome', # Usar 'nome' para ser consistente com o `municipioSelecionado.nome` no frontend
            'VL_LATITUDE': 'lat',
            'VL_LONGITUDE': 'lon'
        })[['nome', 'lat', 'lon']].to_dict(orient='records')
        
        return {"estacoes": estacoes_list}
    
    # Se df_estacoes for None, retorna um erro ou lista vazia
    raise HTTPException(status_code=500, detail="Dados de estações não carregados.")


@app.get("/predict/") # Ajustar o endpoint para receber lat/lon diretamente
async def get_prediction(lat: float, lon: float):
    # A função predict_ensemble no ensemble.py espera lat/lon, não um nome de município.
    # Vamos adaptar aqui.
    try:
        prediction = predict_ensemble(lat, lon) # Passar lat e lon
        return prediction
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predict/history/") # Ajustar para lat/lon
async def get_history(lat: float, lon: float, limit: int = 30):
    try:
        history = predict_historical(lat, lon, limit)
        return history
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/evaluate/")
async def get_evaluation():
    if evaluation_metrics_data:
        return evaluation_metrics_data
    raise HTTPException(status_code=503, detail="Métricas de avaliação não disponíveis ou não carregadas.")
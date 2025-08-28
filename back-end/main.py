import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Permite todas as origens para o desenvolvimento
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Adicione on_startup para carregar os dados
df_estacoes = None
df_avaliacoes = None

@app.on_startup
async def load_data():
    global df_estacoes, df_avaliacoes
    try:
        # Correção principal aqui: garante que lat e lon são lidas como float
        df_estacoes = pd.read_csv("dados/catalogoestacoesautomaticas.csv", sep=';', dtype={'lat': float, 'lon': float})
        df_avaliacoes = pd.read_csv("dados/avaliacoes.csv", sep=';')
        print("Dados carregados com sucesso!")
    except FileNotFoundError:
        print("Erro: Arquivo não encontrado. Verifique os caminhos dos arquivos.")
    except Exception as e:
        print(f"Erro ao carregar os dados: {e}")

@app.get("/estacoes/")
async def get_estacoes():
    if df_estacoes is not None:
        # Converte para uma lista de dicionários para enviar ao front-end
        return {"estacoes": df_estacoes.to_dict(orient='records')}
    return {"estacoes": []}

@app.get("/evaluate/")
async def get_evaluate_metrics():
    if df_avaliacoes is not None:
        # Retorna as avaliações no formato correto para o front-end
        return df_avaliacoes.to_dict(orient='records')[0] # Assumindo que a primeira linha contém as métricas
    return {"erro": "Métricas de avaliação não carregadas."}
import os
import uvicorn
import json
import threading
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from apscheduler.schedulers.background import BackgroundScheduler
from pydantic import BaseModel
from services.ensemble import predict_ensemble
from core.models import carregar_modelos
from core.evaluation import run_ensemble_evaluation
from core.treino_geral import iniciar_ciclo_treinamento

# Variável global para armazenar as métricas de avaliação
evaluation_metrics = {}

# Classe Pydantic para validar a entrada da API de previsão
class Municipio(BaseModel):
    municipio: str

# Scheduler para agendar a avaliação
scheduler = BackgroundScheduler()

# Função para agendar a tarefa de avaliação
def scheduled_evaluation_job():
    print("Scheduler: Iniciando a tarefa de avaliação do ensemble.")
    global evaluation_metrics
    new_metrics = run_ensemble_evaluation()
    if new_metrics:
        evaluation_metrics.update(new_metrics)
    print("Scheduler: Avaliação do ensemble concluída. Métricas:", evaluation_metrics)

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Lifespan: Iniciando aplicação...")
    
    # 1. Carrega os modelos na inicialização
    carregar_modelos()

    # 2. Inicia a thread de treinamento com a nova função
    thread_coleta_treinamento = threading.Thread(target=iniciar_ciclo_treinamento, daemon=True)
    thread_coleta_treinamento.start()
    
    # 3. Agenda a avaliação do ensemble
    scheduler.add_job(scheduled_evaluation_job, 'interval', minutes=60)
    scheduler.start()
    print("Lifespan: Scheduler de avaliação do ensemble configurado e iniciado.")

    # 4. Executa a primeira avaliação no startup
    print("Lifespan: Executando a primeira avaliação do ensemble no startup...")
    scheduled_evaluation_job()
    
    yield
    print("Lifespan: Desligando aplicação...")
    scheduler.shutdown()

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

# Main para rodar o servidor
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
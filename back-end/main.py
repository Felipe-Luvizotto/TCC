# main.py
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from apscheduler.schedulers.background import BackgroundScheduler
from sklearn.exceptions import NotFittedError
import sqlite3

from core.database import criar_tabela_clima
from core.training import iniciar_threads_de_treinamento
from services.ensemble import predict_ensemble
from core.evaluation import run_ensemble_evaluation
from core.models import carregar_modelos

# Função para o job do scheduler
def scheduled_evaluation_job():
    print("Scheduler: Iniciando a tarefa de avaliação do ensemble.")
    try:
        carregar_modelos() 
        metrics = run_ensemble_evaluation()
        print("Scheduler: Avaliação do ensemble concluída. Métricas:", metrics)
    except NotFittedError as nfe:
        print(f"INFO [Scheduler]: Avaliação adiada. Um ou mais modelos ainda não estão treinados/prontos: {nfe}")
    except Exception as e:
        print(f"ERRO [Scheduler]: Erro durante a avaliação agendada do ensemble: {e}")

# Variável global para o scheduler
scheduler = BackgroundScheduler(timezone="America/Sao_Paulo") 

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Código de Startup
    print("Lifespan: Iniciando aplicação...")
    criar_tabela_clima()
    carregar_modelos() 
    iniciar_threads_de_treinamento()
    
    global scheduler
    scheduler.add_job(scheduled_evaluation_job, 'interval', minutes=60, misfire_grace_time=300)
    if not scheduler.running:
        scheduler.start()
        print("Lifespan: Scheduler de avaliação do ensemble configurado e iniciado.")
    
    print("Lifespan: Executando a primeira avaliação do ensemble no startup...")
    try:
        scheduled_evaluation_job()
    except Exception as e:
        print(f"Lifespan: Erro ao executar a primeira avaliação do ensemble no startup: {e}")
        
    yield 

    # Código de Shutdown
    print("Lifespan: Encerrando aplicação...")
    if scheduler.running:
        scheduler.shutdown()
        print("Lifespan: Scheduler de avaliação encerrado.")

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"mensagem": "API de previsão de enchentes ativa!"}

@app.get("/evaluate/")
def get_ensemble_evaluation():
    try:
        metrics = run_ensemble_evaluation()
        return metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao executar avaliação: {str(e)}")
    
@app.get("/predict/")
def predict_flood(municipio: str):
    try:
        return predict_ensemble(municipio)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na previsão: {str(e)}")
    
@app.get("/predict/history/")
def get_prediction_history(lat: float, lon: float, limit: int = 30):
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()

    try:
        cursor.execute("""
            SELECT data_hora, enchente
            FROM clima
            WHERE latitude = ? AND longitude = ?
            ORDER BY data_hora DESC
            LIMIT ?
        """, (lat, lon, limit))
        history = cursor.fetchall()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao buscar histórico: {str(e)}")
    finally:
        conn.close() 
        
    return [{"data_hora": row[0], "probabilidade_enchente": row[1]} for row in reversed(history)]

if __name__ == "__main__":
    # CORREÇÃO CRUCIAL: Adicionar a linha para iniciar o servidor.
    uvicorn.run(app, host="0.0.0.0", port=8000)
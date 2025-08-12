import numpy as np
import time
import sqlite3
import joblib
import pandas as pd
from core.models import rf_model, xgb_model, lstm_model
from core.treino_lstm import treinar_modelo_lstm # Usaremos esta função
from core.treino_xgb import treinar_modelo_xgb  # Usaremos esta função
from core.evaluation import run_ensemble_evaluation
from sklearn.model_selection import train_test_split

def iniciar_treinamento_acelerado(num_ciclos=5):
    """
    Roda um ciclo de treinamento e avaliação com dados históricos.
    """
    print("Iniciando o ciclo de treinamento acelerado...")
    conn = sqlite3.connect("database.db")
    
    # Carrega todos os dados históricos de uma vez
    df = pd.read_sql_query("SELECT * FROM clima_historico", conn)
    conn.close()
    
    if df.empty or len(df) < 50:
        print("ERRO: Dados insuficientes para treinamento. Verifique a tabela 'clima_historico'.")
        return
        
    for ciclo in range(num_ciclos):
        print(f"\n--- Iniciando Ciclo de Treinamento {ciclo + 1}/{num_ciclos} ---")
        
        # === 1. Prepara os dados para treinamento e teste ===
        # As colunas precisam ser ajustadas para o seu dataset consolidado
        try:
            X = df[['Temperatura', 'Umidade', 'Vento', 'Precipitacao']].values
            y = df['Enchente'].values
        except KeyError as e:
            print(f"ERRO: Coluna {e} não encontrada no seu dataset. Verifique os nomes das colunas.")
            return

        X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # === 2. Treina os modelos ===
        print("Treinando o modelo de Random Forest...")
        rf_model.fit(X_treino, y_treino)
        joblib.dump(rf_model, "modelo_rf.pkl")
        
        print("Treinando o modelo XGBoost...")
        treinar_modelo_xgb(X_treino, y_treino)
        
        print("Treinando o modelo LSTM...")
        treinar_modelo_lstm(X_treino, y_treino)
        
        print("--- Treinamento concluído. ---")
        
        # === 3. Avalia os modelos ===
        print("Iniciando a avaliação do ensemble...")
        metrics = run_ensemble_evaluation(X_teste, y_teste)
        
        if metrics:
            print("Avaliação completa:")
            for model_name, model_metrics in metrics.items():
                print(f"  Métricas para {model_name}:")
                for key, value in model_metrics.items():
                    print(f"    - {key}: {value:.4f}")
        else:
            print("Falha ao executar a avaliação.")

    print("\nTreinamento acelerado finalizado.")

if __name__ == '__main__':
    iniciar_treinamento_acelerado(num_ciclos=10) # Altere o número de ciclos aqui
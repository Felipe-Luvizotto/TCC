import numpy as np
import time
import sqlite3
import joblib
import pandas as pd
from core.models import rf_model, xgb_model, lstm_model
from core.treino_lstm import treinar_modelo_lstm
from core.treino_xgb import treinar_modelo_xgb
from core.evaluation import run_ensemble_evaluation
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def iniciar_treinamento_acelerado(num_ciclos=1, data_limit=5000):
    """
    Roda um ciclo de treinamento e avaliação com dados históricos.
    Args:
        num_ciclos (int): Número de vezes para rodar o treinamento.
        data_limit (int): Limite de linhas a serem lidas do banco de dados para um teste rápido.
                          Use None para ler todos os dados.
    """
    print("Iniciando o ciclo de treinamento acelerado...")
    conn = sqlite3.connect("database.db")

    try:
        # AQUI ESTÁ A OTIMIZAÇÃO: Adiciona a cláusula LIMIT para ler apenas uma amostra dos dados
        query = "SELECT * FROM clima_historico"
        if data_limit is not None and isinstance(data_limit, int) and data_limit > 0:
            query += f" LIMIT {data_limit}"
        
        df = pd.read_sql_query(query, conn)
        
    except pd.io.sql.DatabaseError as e:
        print(f"ERRO: Não foi possível carregar os dados do banco de dados. Verifique se o arquivo 'database.db' existe e a tabela 'clima_historico' está correta. Detalhes: {e}")
        return
    finally:
        conn.close()

    if df.empty or len(df) < 50:
        print("ERRO: Dados insuficientes para treinamento. Verifique a tabela 'clima_historico'.")
        return

    print(f"Dataset carregado com sucesso. Total de {df.shape[0]} linhas e {df.shape[1]} colunas.")
    print("AVISO: Usando um subconjunto de dados para treinamento rápido. A precisão do modelo será reduzida.")

    # === 1. Prepara os dados para treinamento e teste ===
    try:
        X = df[['TEMPERATURA', 'UMIDADE', 'VENTO', 'PRECIPITACAO']].values
        y = df['Enchente'].values
    except KeyError as e:
        print(f"ERRO: Coluna {e} não encontrada no seu dataset. Verifique os nomes das colunas.")
        return

    X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Dados divididos: {len(X_treino)} amostras para treino e {len(X_teste)} para teste.")

    for ciclo in range(num_ciclos):
        print(f"\n--- Iniciando Ciclo de Treinamento {ciclo + 1}/{num_ciclos} ---")

        # === 2. Treina os modelos ===
        print("Treinando o modelo de Random Forest...")
        rf_model_acelerado = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        rf_model_acelerado.fit(X_treino, y_treino)
        joblib.dump(rf_model_acelerado, "modelo_rf.pkl")
        print("Treinamento do Random Forest concluído.")

        print("Treinando o modelo XGBoost...")
        treinar_modelo_xgb(X_treino, y_treino)
        print("Treinamento do XGBoost concluído.")

        print("Treinando o modelo LSTM...")
        treinar_modelo_lstm(X_treino, y_treino)
        print("Treinamento do LSTM concluído.")

        print("--- Treinamento de todos os modelos concluído. ---")

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
            print("AVISO: Avaliação não foi executada.")

if __name__ == "__main__":
    iniciar_treinamento_acelerado(num_ciclos=1, data_limit=5000) # Use 5000 para um teste inicial
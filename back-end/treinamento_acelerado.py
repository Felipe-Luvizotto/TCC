# --- START OF FILE treinamento_acelerado.py ---
import pandas as pd
import sqlite3
from sklearn.model_selection import train_test_split
from core.treino_rf import treinar_modelo_rf
from core.treino_xgb import treinar_modelo_xgb
from core.treino_lstm import treinar_modelo_lstm
from core.evaluation import run_ensemble_evaluation
import numpy as np # Importar numpy para checagem

def ciclo_de_treinamento_acelerado():
    """
    Orquestra o processo de carregamento, divisão e treinamento dos modelos.
    """
    print("Iniciando o ciclo de treinamento acelerado...")

    try:
        conn = sqlite3.connect('database.db')
        # Seleciona as colunas esperadas pelos modelos (Temperatura, Umidade, Vento, Precipitacao)
        df = pd.read_sql_query("SELECT Temperatura, Umidade, Vento, Precipitacao, Enchente FROM clima", conn)
        conn.close()
        
        df.dropna(inplace=True)

        if len(df) < 20:
            print(f"AVISO: Dados insuficientes no banco de dados para um treinamento significativo. Mínimo de 20 linhas. Atualmente: {len(df)}")
            return

        # Verifica se ambas as classes (0 e 1 para 'Enchente') estão presentes
        if len(df['Enchente'].unique()) < 2:
            print("AVISO: A coluna 'Enchente' não contém ambas as classes (0 e 1). Não é possível realizar um split estratificado ou treinar corretamente.")
            print(f"Valores únicos em 'Enchente': {df['Enchente'].unique()}")
            # Se só houver uma classe, talvez ainda possamos treinar, mas não estratificar
            # Por enquanto, vamos retornar se não houver diversidade de classes
            return

        print(f"Dataset carregado com sucesso. Total de {len(df)} linhas e {len(df.columns)} colunas.")
        print("AVISO: Usando um subconjunto de dados para treinamento rápido. A precisão do modelo será reduzida.")

        X = df.drop('Enchente', axis=1) # Usar 'Enchente' conforme o nome da coluna
        y = df['Enchente']
        
        # Correção: Usar stratify se ambas as classes estiverem presentes
        X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        feature_columns = X.columns.tolist()

        print("--- Treinamento 1/1 ---")

        # Chama a função de treinamento do Random Forest
        print("Treinando o modelo de Random Forest...")
        treinar_modelo_rf(X_treino.values, y_treino.values)
        print("Treinamento do Random Forest concluído.")

        # Chama a função de treinamento do XGBoost
        print("Treinando o modelo XGBoost...")
        treinar_modelo_xgb(X_treino.values, y_treino.values)
        print("Treinamento do XGBoost concluído.")
        
        # Chama a função de treinamento do LSTM
        print("Treinando o modelo LSTM...")
        treinar_modelo_lstm(X_treino.values, y_treino.values, feature_columns)
        print("Treinamento do LSTM concluído.")

        print("--- Treinamento de todos os modelos concluído. ---\n")

        print("Iniciando a avaliação do ensemble...")
        metricas = run_ensemble_evaluation(X_teste.values, y_teste.values)
        if metricas:
            print("Avaliação concluída com sucesso:")
            for model_name, m in metricas.items():
                print(f"\n--- Métricas para {model_name} ---")
                print(f"  Acurácia: {m['accuracy']:.4f}")
                print(f"  Precisão: {m['precision']:.4f}")
                print(f"  Recall: {m['recall']:.4f}")
                print(f"  F1-Score: {m['f1_score']:.4f}")
                # Verifica se 'auc_roc' existe antes de tentar formatar
                if 'auc_roc' in m:
                    print(f"  AUC-ROC: {m['auc_roc']:.4f}")
        else:
            print("AVISO: Não foi possível realizar a avaliação do modelo.")
            
    except FileNotFoundError as e:
        print(f"ERRO: Arquivo não encontrado. Verifique os caminhos dos arquivos: {e}")
    except Exception as e:
        print(f"Ocorreu um erro no ciclo de treinamento: {e}")

if __name__ == "__main__":
    ciclo_de_treinamento_acelerado()
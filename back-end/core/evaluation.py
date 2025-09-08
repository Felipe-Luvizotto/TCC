# --- START OF FILE evaluation.py ---
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score
from core.models import lstm_model, rf_model, xgb_model, lstm_scaler # Importar o scaler também
import torch
import torch.nn.functional as F
import json # Para salvar as métricas
import os

def predict_lstm(data):
    """Prevê usando o modelo LSTM. Aplica o scaler antes da previsão."""
    if lstm_scaler is None:
        print("AVISO: Scaler LSTM não carregado. Não é possível prever com LSTM.")
        # Pode retornar um array de zeros ou lançar um erro, dependendo da robustez desejada
        return np.zeros(data.shape[0]) 
    
    # Aplica o scaler nos dados de entrada
    scaled_data = lstm_scaler.transform(data)
    
    # A entrada para o LSTM precisa ser um tensor 3D: (batch_size, sequence_length, input_size)
    # Cada linha de 'data' é uma observação, então sequence_length = 1
    data_tensor = torch.tensor(scaled_data, dtype=torch.float32).unsqueeze(1)
    
    # Coloca o modelo em modo de avaliação
    lstm_model.eval()
    with torch.no_grad():
        output = lstm_model(data_tensor)
        # O modelo LSTMModel já aplica sigmoid na forward, então 'output' já são probabilidades.
        # Precisamos converter para classes binárias (0 ou 1)
        predictions = (output > 0.5).int().flatten().numpy()
    return predictions

def run_ensemble_evaluation(X_teste, y_teste):
    """
    Avalia o desempenho de cada modelo individualmente e do ensemble, e salva as métricas.
    """
    print("DEBUG: Executando avaliação de modelos...")
    # Verifica se os modelos RF e XGB foram treinados (têm atributos específicos após fit)
    rf_trained = hasattr(rf_model, 'estimators_') and len(rf_model.estimators_) > 0
    xgb_trained = hasattr(xgb_model, '_Booster') # e.g., if xgb_model.is_trained

    # Verifica se o modelo LSTM e o scaler foram carregados/treinados
    lstm_trained = hasattr(lstm_model, 'lstm') and lstm_scaler is not None # Checagem mais robusta

    if not (rf_trained and xgb_trained and lstm_trained):
        print("INFO: Nem todos os modelos ou o scaler do LSTM foram treinados/carregados. Não é possível realizar a avaliação completa.")
        # Retorna métricas apenas para os modelos que foram carregados/treinados
        # Ou um dicionário vazio
        return {}
    
    metrics = {}
    
    try:
        # Previsões do Random Forest
        rf_predictions = rf_model.predict(X_teste)
        
        # Previsões do XGBoost
        xgb_predictions = xgb_model.predict(X_teste)
        
        # Previsões do LSTM (usando a função adaptada)
        lstm_predictions = predict_lstm(X_teste)

        # Previsões do Ensemble (voto majoritário)
        ensemble_predictions = (rf_predictions + xgb_predictions + lstm_predictions >= 2).astype(int)

        # Calcula métricas para cada modelo
        # Random Forest
        metrics['Random_Forest'] = { # Alinhando com os nomes do frontend
            'accuracy': accuracy_score(y_teste, rf_predictions),
            'precision': precision_score(y_teste, rf_predictions, zero_division=0),
            'recall': recall_score(y_teste, rf_predictions, zero_division=0),
            'f1_score': f1_score(y_teste, rf_predictions, zero_division=0),
            'matthews_corrcoef': matthews_corrcoef(y_teste, rf_predictions),
            'auc_roc': roc_auc_score(y_teste, rf_predictions)
        }
        
        # XGBoost
        metrics['XGBoost'] = {
            'accuracy': accuracy_score(y_teste, xgb_predictions),
            'precision': precision_score(y_teste, xgb_predictions, zero_division=0),
            'recall': recall_score(y_teste, xgb_predictions, zero_division=0),
            'f1_score': f1_score(y_teste, xgb_predictions, zero_division=0),
            'matthews_corrcoef': matthews_corrcoef(y_teste, xgb_predictions),
            'auc_roc': roc_auc_score(y_teste, xgb_predictions)
        }
        
        # LSTM
        metrics['LSTM'] = {
            'accuracy': accuracy_score(y_teste, lstm_predictions),
            'precision': precision_score(y_teste, lstm_predictions, zero_division=0),
            'recall': recall_score(y_teste, lstm_predictions, zero_division=0),
            'f1_score': f1_score(y_teste, lstm_predictions, zero_division=0),
            'matthews_corrcoef': matthews_corrcoef(y_teste, lstm_predictions),
            'auc_roc': roc_auc_score(y_teste, lstm_predictions)
        }
        
        # Ensemble
        metrics['Ensemble'] = {
            'accuracy': accuracy_score(y_teste, ensemble_predictions),
            'precision': precision_score(y_teste, ensemble_predictions, zero_division=0),
            'recall': recall_score(y_teste, ensemble_predictions, zero_division=0),
            'f1_score': f1_score(y_teste, ensemble_predictions, zero_division=0),
            'matthews_corrcoef': matthews_corrcoef(y_teste, ensemble_predictions),
            'auc_roc': roc_auc_score(y_teste, ensemble_predictions)
        }
        
        # Salvar as métricas em um arquivo JSON
        with open('evaluation_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)
        print("INFO: Métricas de avaliação salvas em 'evaluation_metrics.json'.")

        return metrics

    except Exception as e:
        print(f"ERRO: Falha na avaliação do ensemble. Verifique os dados. Erro: {e}")
        return None
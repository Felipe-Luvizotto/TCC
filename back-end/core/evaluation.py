import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score
from core.models import lstm_model, rf_model, xgb_model # CORREÇÃO AQUI
import torch
import torch.nn.functional as F

def predict_lstm(data):
    """Prevê usando o modelo LSTM."""
    # A entrada para o LSTM precisa ser um tensor 3D
    data_tensor = torch.tensor(np.expand_dims(data, axis=1), dtype=torch.float32)
    with torch.no_grad():
        output = lstm_model(data_tensor)
        # Aplica sigmoide para converter logits em probabilidades
        probabilities = F.sigmoid(output)
        predictions = (probabilities > 0.5).int().flatten().numpy()
    return predictions

def run_ensemble_evaluation(X_teste, y_teste):
    """
    Avalia o desempenho de cada modelo individualmente e do ensemble.
    """
    print("DEBUG: Executando avaliação de modelos...")
    if not (hasattr(rf_model, 'estimators_') and hasattr(xgb_model, '_Booster')):
        print("INFO: Modelos não foram treinados. Não é possível avaliar.")
        return None
    
    metrics = {}
    
    try:
        # Previsões do Random Forest
        rf_predictions = rf_model.predict(X_teste)
        
        # Previsões do XGBoost
        xgb_predictions = xgb_model.predict(X_teste)
        
        # Previsões do LSTM
        lstm_predictions = predict_lstm(X_teste)

        # Previsões do Ensemble (voto majoritário)
        ensemble_predictions = (rf_predictions + xgb_predictions + lstm_predictions >= 2).astype(int)

        # Calcula métricas para cada modelo
        metrics['RandomForest'] = {
            'accuracy': accuracy_score(y_teste, rf_predictions),
            'precision': precision_score(y_teste, rf_predictions, zero_division=0),
            'recall': recall_score(y_teste, rf_predictions, zero_division=0),
            'f1_score': f1_score(y_teste, rf_predictions, zero_division=0),
            'matthews_corrcoef': matthews_corrcoef(y_teste, rf_predictions),
            'auc_roc': roc_auc_score(y_teste, rf_predictions)
        }
        
        metrics['XGBoost'] = {
            'accuracy': accuracy_score(y_teste, xgb_predictions),
            'precision': precision_score(y_teste, xgb_predictions, zero_division=0),
            'recall': recall_score(y_teste, xgb_predictions, zero_division=0),
            'f1_score': f1_score(y_teste, xgb_predictions, zero_division=0),
            'matthews_corrcoef': matthews_corrcoef(y_teste, xgb_predictions),
            'auc_roc': roc_auc_score(y_teste, xgb_predictions)
        }
        
        metrics['LSTM'] = {
            'accuracy': accuracy_score(y_teste, lstm_predictions),
            'precision': precision_score(y_teste, lstm_predictions, zero_division=0),
            'recall': recall_score(y_teste, lstm_predictions, zero_division=0),
            'f1_score': f1_score(y_teste, lstm_predictions, zero_division=0),
            'matthews_corrcoef': matthews_corrcoef(y_teste, lstm_predictions),
            'auc_roc': roc_auc_score(y_teste, lstm_predictions)
        }
        
        metrics['Ensemble'] = {
            'accuracy': accuracy_score(y_teste, ensemble_predictions),
            'precision': precision_score(y_teste, ensemble_predictions, zero_division=0),
            'recall': recall_score(y_teste, ensemble_predictions, zero_division=0),
            'f1_score': f1_score(y_teste, ensemble_predictions, zero_division=0),
            'matthews_corrcoef': matthews_corrcoef(y_teste, ensemble_predictions),
            'auc_roc': roc_auc_score(y_teste, ensemble_predictions)
        }
        
        return metrics

    except Exception as e:
        print(f"ERRO: Falha na avaliação do ensemble. Verifique os dados. Erro: {e}")
        return None
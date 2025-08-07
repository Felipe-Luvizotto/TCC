# core/models.py
import os
import joblib
import torch
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from .model_lstm import LSTMModel # Supondo que model_lstm.py está no mesmo diretório (core)

# Instâncias iniciais (não treinadas ou com estrutura base)
model = LSTMModel()
rf_model = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=5, random_state=42, class_weight='balanced')
xgb_model = xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, eval_metric='logloss', use_label_encoder=False)

def carregar_modelos():
    global model, rf_model, xgb_model
    
    print("DEBUG [models.carregar_modelos]: Iniciando carregamento de modelos...")

    # LSTM
    if os.path.exists("modelo_lstm.pth"):
        model.load_state_dict(torch.load("modelo_lstm.pth"))
        model.eval()
        print("DEBUG [models.carregar_modelos]: Modelo LSTM carregado de 'modelo_lstm.pth'.")
    else:
        print("INFO [models.carregar_modelos]: 'modelo_lstm.pth' não encontrado. Usando instância LSTM não treinada.")

    # Random Forest
    print(f"DEBUG [models.carregar_modelos]: Verificando 'modelo_rf.pkl'. Instância atual rf_model é treinada? {hasattr(rf_model, 'estimators_')}")
    if os.path.exists("modelo_rf.pkl"):
        try:
            loaded_rf = joblib.load("modelo_rf.pkl")
            # Verifica se o modelo carregado parece treinado
            if hasattr(loaded_rf, 'estimators_') and len(loaded_rf.estimators_) > 0:
                rf_model = loaded_rf
                print(f"DEBUG [models.carregar_modelos]: Modelo RF carregado de 'modelo_rf.pkl' e PARECE TREINADO. N° estimators: {len(rf_model.estimators_)}")
            else:
                # Se o arquivo existe mas o modelo não parece treinado, pode ser um problema no salvamento
                # ou um arquivo antigo com um modelo não treinado.
                print(f"AVISO [models.carregar_modelos]: 'modelo_rf.pkl' encontrado e carregado, MAS NÃO PARECE TREINADO. Tipo: {type(loaded_rf)}. Usando instância RF original não treinada.")
                # Neste caso, rf_model continua sendo a instância não treinada original.
                # Para forçar o uso do carregado mesmo que pareça não treinado (para ver outros erros):
                # rf_model = loaded_rf
        except Exception as e:
            print(f"ERRO [models.carregar_modelos]: Falha ao carregar 'modelo_rf.pkl': {e}. Usando instância RF original não treinada.")
    else:
        print("INFO [models.carregar_modelos]: 'modelo_rf.pkl' não encontrado. Usando instância RF original não treinada.")
    print(f"DEBUG [models.carregar_modelos]: Após tentativa de carregar RF, rf_model é treinado? {hasattr(rf_model, 'estimators_')}")


    # XGBoost
    print(f"DEBUG [models.carregar_modelos]: Verificando 'modelo_xgb.pkl'. Instância atual xgb_model é treinada? {hasattr(xgb_model, '_Booster') and xgb_model._Booster is not None}")
    if os.path.exists("modelo_xgb.pkl"):
        try:
            loaded_xgb = joblib.load("modelo_xgb.pkl")
            # Verifica se o modelo carregado parece treinado (presença do _Booster e classes_)
            if hasattr(loaded_xgb, '_Booster') and loaded_xgb._Booster is not None and hasattr(loaded_xgb, 'classes_'):
                xgb_model = loaded_xgb
                print(f"DEBUG [models.carregar_modelos]: Modelo XGB carregado de 'modelo_xgb.pkl' e PARECE TREINADO.")
            else:
                print(f"AVISO [models.carregar_modelos]: 'modelo_xgb.pkl' encontrado e carregado, MAS NÃO PARECE TREINADO. Tipo: {type(loaded_xgb)}. Usando instância XGB original não treinada.")
        except Exception as e:
            print(f"ERRO [models.carregar_modelos]: Falha ao carregar 'modelo_xgb.pkl': {e}. Usando instância XGB original não treinada.")
    else:
        print("INFO [models.carregar_modelos]: 'modelo_xgb.pkl' não encontrado. Usando instância XGB original não treinada.")
    print(f"DEBUG [models.carregar_modelos]: Após tentativa de carregar XGB, xgb_model é treinado? {hasattr(xgb_model, '_Booster') and xgb_model._Booster is not None}")
    print("DEBUG [models.carregar_modelos]: Fim do carregamento de modelos.")
# --- START OF FILE models.py ---
import joblib
import os
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from core.model_lstm import LSTMModel # Importa a definição correta do LSTMModel

# Inicializa as instâncias dos modelos com parâmetros padrão
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
xgb_model = XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False, n_estimators=100, random_state=42)
# Instancia o modelo LSTM usando a definição de core.model_lstm
lstm_model = LSTMModel(input_size=4, hidden_size=50) # Ajuste input_size conforme suas features (Temperatura, Umidade, Vento, Precipitacao)

# Variável global para o scaler do LSTM
lstm_scaler = None

def carregar_modelos():
    """
    Tenta carregar os modelos pré-treinados e o scaler do LSTM.
    Se não existirem, usa as instâncias padrão (não treinadas).
    """
    global lstm_scaler # Permite modificar a variável global
    print("DEBUG [models.carregar_modelos]: Iniciando carregamento de modelos...")
    
    # Carregar modelo LSTM
    caminho_lstm = 'modelo_lstm.pth'
    if os.path.exists(caminho_lstm):
        try:
            # Carrega o estado do modelo
            lstm_model.load_state_dict(torch.load(caminho_lstm))
            lstm_model.eval() # Coloca o modelo em modo de avaliação
            print("INFO [models.carregar_modelos]: 'modelo_lstm.pth' carregado com sucesso.")
        except Exception as e:
            print(f"ERRO [models.carregar_modelos]: Falha ao carregar 'modelo_lstm.pth': {e}")
            print("INFO [models.carregar_modelos]: Usando instância LSTM não treinada.")
    else:
        print("INFO [models.carregar_modelos]: 'modelo_lstm.pth' não encontrado. Usando instância LSTM não treinada.")

    # Carregar scaler do LSTM
    caminho_scaler_lstm = 'scaler_lstm.pkl'
    if os.path.exists(caminho_scaler_lstm):
        try:
            lstm_scaler = joblib.load(caminho_scaler_lstm)
            print("INFO [models.carregar_modelos]: 'scaler_lstm.pkl' carregado com sucesso.")
        except Exception as e:
            print(f"ERRO [models.carregar_modelos]: Falha ao carregar 'scaler_lstm.pkl': {e}")
            print("INFO [models.carregar_modelos]: Scaler LSTM não disponível.")
    else:
        print("INFO [models.carregar_modelos]: 'scaler_lstm.pkl' não encontrado. Scaler LSTM não disponível.")

    # Carregar modelo Random Forest
    caminho_rf = 'modelo_rf.pkl'
    if os.path.exists(caminho_rf):
        try:
            rf_model_carregado = joblib.load(caminho_rf)
            rf_model.set_params(**rf_model_carregado.get_params())
            # Se o modelo carregado já foi treinado, ele terá o atributo .classes_
            if hasattr(rf_model_carregado, 'classes_'):
                 rf_model.classes_ = rf_model_carregado.classes_
            if hasattr(rf_model_carregado, 'n_features_in_'): # scikit-learn >= 0.24
                 rf_model.n_features_in_ = rf_model_carregado.n_features_in_
            elif hasattr(rf_model_carregado, 'n_features_'): # scikit-learn < 0.24
                 rf_model.n_features_ = rf_model_carregado.n_features_

            print(f"INFO [models.carregar_modelos]: Modelo RF carregado de '{caminho_rf}'. N° estimators: {rf_model.get_params()['n_estimators']}")
        except Exception as e:
            print(f"ERRO [models.carregar_modelos]: Falha ao carregar '{caminho_rf}': {e}")
            print("INFO [models.carregar_modelos]: Usando instância RF não treinada.")
    else:
        print(f"INFO [models.carregar_modelos]: '{caminho_rf}' não encontrado. Usando instância RF não treinada.")

    # Carregar modelo XGBoost
    caminho_xgb = 'modelo_xgb.pkl'
    if os.path.exists(caminho_xgb):
        try:
            xgb_model_carregado = joblib.load(caminho_xgb)
            xgb_model.set_params(**xgb_model_carregado.get_params())
            # É importante carregar o _Booster se existir
            if hasattr(xgb_model_carregado, '_Booster'):
                xgb_model._Booster = xgb_model_carregado._Booster
            print(f"INFO [models.carregar_modelos]: Modelo XGB carregado de '{caminho_xgb}'.")
        except Exception as e:
            print(f"ERRO [models.carregar_modelos]: Falha ao carregar '{caminho_xgb}': {e}")
            print("INFO [models.carregar_modelos]: Usando instância XGB não treinada.")
    else:
        print(f"INFO [models.carregar_modelos]: '{caminho_xgb}' não encontrado. Usando instância XGB não treinada.")

    print("DEBUG [models.carregar_modelos]: Fim do carregamento de modelos.")
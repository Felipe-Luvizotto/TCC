import joblib
import os
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Define o modelo LSTM
class LSTMPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1, num_layers=1):
        super(LSTMPredictor, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        lstm_out, _ = self.lstm(x, (h0, c0))
        out = self.linear(lstm_out[:, -1, :])
        return out

# Inicializa as instâncias dos modelos
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
xgb_model = XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False, n_estimators=100, random_state=42)
lstm_model = LSTMPredictor(input_dim=4, hidden_dim=50)


def carregar_modelos():
    """
    Tenta carregar os modelos pré-treinados. Se não existirem, usa as instâncias padrão.
    """
    print("DEBUG [models.carregar_modelos]: Iniciando carregamento de modelos...")
    
    # Carregar modelo LSTM
    caminho_lstm = 'modelo_lstm.pth'
    if os.path.exists(caminho_lstm):
        try:
            lstm_model.load_state_dict(torch.load(caminho_lstm))
            lstm_model.eval()
            print("INFO [models.carregar_modelos]: 'modelo_lstm.pth' carregado com sucesso.")
        except Exception as e:
            print(f"ERRO [models.carregar_modelos]: Falha ao carregar 'modelo_lstm.pth': {e}")
            print("INFO [models.carregar_modelos]: Usando instância LSTM não treinada.")
    else:
        print("INFO [models.carregar_modelos]: 'modelo_lstm.pth' não encontrado. Usando instância LSTM não treinada.")

    # Carregar modelo Random Forest
    caminho_rf = 'modelo_rf.pkl'
    if os.path.exists(caminho_rf):
        try:
            print(f"DEBUG [models.carregar_modelos]: Verificando '{caminho_rf}'. Instância atual rf_model é treinada? {hasattr(rf_model, 'estimators_')}")
            rf_model_carregado = joblib.load(caminho_rf)
            rf_model.set_params(**rf_model_carregado.get_params())
            rf_model.classes_ = rf_model_carregado.classes_
            rf_model.n_features_ = rf_model_carregado.n_features_
            print(f"DEBUG [models.carregar_modelos]: Modelo RF carregado de '{caminho_rf}' e PARECE TREINADO. N° estimators: {rf_model.get_params()['n_estimators']}")
        except Exception as e:
            print(f"ERRO [models.carregar_modelos]: Falha ao carregar '{caminho_rf}': {e}")
            print("INFO [models.carregar_modelos]: Usando instância RF não treinada.")
    else:
        print(f"INFO [models.carregar_modelos]: '{caminho_rf}' não encontrado. Usando instância RF não treinada.")
    print(f"DEBUG [models.carregar_modelos]: Após tentativa de carregar RF, rf_model é treinado? {hasattr(rf_model, 'estimators_')}")

    # Carregar modelo XGBoost
    caminho_xgb = 'modelo_xgb.pkl'
    if os.path.exists(caminho_xgb):
        try:
            print(f"DEBUG [models.carregar_modelos]: Verificando '{caminho_xgb}'. Instância atual xgb_model é treinada? {hasattr(xgb_model, '_Booster')}")
            xgb_model_carregado = joblib.load(caminho_xgb)
            xgb_model.set_params(**xgb_model_carregado.get_params())
            xgb_model._Booster = xgb_model_carregado._Booster
            print(f"DEBUG [models.carregar_modelos]: Modelo XGB carregado de '{caminho_xgb}' e PARECE TREINADO.")
        except Exception as e:
            print(f"ERRO [models.carregar_modelos]: Falha ao carregar '{caminho_xgb}': {e}")
            print("INFO [models.carregar_modelos]: Usando instância XGB não treinada.")
    else:
        print(f"INFO [models.carregar_modelos]: '{caminho_xgb}' não encontrado. Usando instância XGB não treinada.")
    print(f"DEBUG [models.carregar_modelos]: Após tentativa de carregar XGB, xgb_model é treinado? {hasattr(xgb_model, '_Booster')}")

    print("DEBUG [models.carregar_modelos]: Fim do carregamento de modelos.")
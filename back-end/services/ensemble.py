import os, joblib, torch, numpy as np, sqlite3
# A linha abaixo importa as instâncias globais dos modelos
from core.models import rf_model, xgb_model, model as lstm_model
from services.weather import get_weather_data

# O BLOCO DE CÓDIGO ABAIXO FOI REMOVIDO PARA EVITAR ERROS NA INICIALIZAÇÃO:
# try:
#     lstm_model.load_state_dict(torch.load("modelo_lstm.pth"))
#     rf_model = joblib.load("modelo_rf.pkl")
#     xgb_model = joblib.load("modelo_xgb.pkl")
#     lstm_model.eval()
# except Exception as e:
#     print(f"Erro ao carregar modelos: {e}")

def predict_ensemble(municipio):
    # O predict_ensemble AGORA USA as instâncias globais de modelo
    # que foram carregadas pelo `lifespan` na inicialização.

    # Abertura da conexão com o banco de dados dentro da função
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()

    cursor.execute("SELECT latitude, longitude FROM municipios WHERE nome=?", (municipio,))
    coords = cursor.fetchone()
    conn.close() 

    if not coords:
        return {"error": "Município não encontrado"}

    lat, lon = coords
    weather_data = get_weather_data(lat, lon)

    if weather_data is None:
        return {"error": "Não foi possível obter dados climáticos"}

    temp, humidity, wind, precipitation = weather_data
    
    # Prepara os dados para os modelos
    data_rf_xgb = np.array([temp, humidity, wind, precipitation]).reshape(1, -1)
    data_lstm = torch.tensor(data_rf_xgb.reshape(-1, 1, 4), dtype=torch.float32)

    # Previsões individuais
    pred_rf = rf_model.predict_proba(data_rf_xgb)[0][1]
    pred_xgb = xgb_model.predict_proba(data_rf_xgb)[0][1]
    
    with torch.no_grad():
        lstm_model.eval()
        pred_lstm = torch.sigmoid(lstm_model(data_lstm)).item()

    # Combinação das previsões (ensemble)
    pred_final = (pred_lstm * 0.5) + (pred_rf * 0.3) + (pred_xgb * 0.2)

    return {"prediction": float(pred_final)}
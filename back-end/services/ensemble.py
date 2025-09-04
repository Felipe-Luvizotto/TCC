import os, joblib, torch, numpy as np, sqlite3
from core.models import rf_model, xgb_model
from core.model_lstm import LSTMModel as lstm_model
from services.weather import get_weather_data
import json

def predict_ensemble(municipio):
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
    
    # Previsões individuais
    pred_rf = rf_model.predict_proba(data_rf_xgb)[0][1]
    pred_xgb = xgb_model.predict_proba(data_rf_xgb)[0][1]
    
    # LSTM requer um tensor 3D
    data_lstm = torch.tensor(data_rf_xgb.reshape(-1, 1, 4), dtype=torch.float32)
    with torch.no_grad():
        lstm_model.eval()
        pred_lstm = lstm_model(data_lstm).item()

    # Previsão final do ensemble (média das probabilidades)
    ensemble_prediction = (pred_rf + pred_xgb + pred_lstm) / 3

    # Define a probabilidade de enchente com base na previsão do ensemble
    flood_probability = ensemble_prediction * 100 
    flood_probability = min(100, max(0, flood_probability)) # Garante que o valor esteja entre 0 e 100

    return {
        "municipio": municipio,
        "probabilidade_enchente": flood_probability,
        "dados_atuais": {
            "Temperatura": temp,
            "Umidade": humidity,
            "Vento": wind,
            "Precipitacao": precipitation
        }
    }

def predict_historical(municipio: str):
    """
    Busca o histórico de dados e previsões para um município no banco de dados.
    """
    print(f"DEBUG: Buscando histórico para {municipio}...")
    try:
        conn = sqlite3.connect('database.db')
        df = pd.read_sql_query(
            "SELECT data_hora, probabilidade FROM historico_previsao WHERE municipio = ? ORDER BY data_hora ASC",
            conn,
            params=(municipio,)
        )
        conn.close()

        if df.empty:
            print(f"AVISO: Nenhum dado histórico encontrado para {municipio}.")
            return {"error": "Nenhum dado histórico encontrado."}

        # Converte a coluna de data para o formato necessário pelo front-end
        df['data_hora'] = pd.to_datetime(df['data_hora'])

        # Prepara a resposta no formato JSON
        historico = {
            "labels": df['data_hora'].dt.strftime('%d/%m %Hh').tolist(),
            "data": df['probabilidade'].tolist()
        }
        return {"historico": historico}

    except Exception as e:
        print(f"ERRO: Falha ao buscar histórico de previsões para {municipio}: {e}")
        return {"error": "Falha interna ao carregar o histórico."}
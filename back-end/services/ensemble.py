# --- START OF FILE ensemble.py ---
import os, joblib, torch, numpy as np, sqlite3
from core.models import rf_model, xgb_model, lstm_model, lstm_scaler # Importar o lstm_scaler
from core.model_lstm import LSTMModel # Importar a classe para instanciar se necessário (embora já instanciada em models.py)
from services.weather import get_weather_data
import json
import pandas as pd # Importar pandas para histórico

def predict_ensemble(lat: float, lon: float): # Recebe lat e lon diretamente
    """
    Realiza a previsão de enchente para uma dada latitude e longitude.
    """
    
    weather_data = get_weather_data(lat, lon)

    if weather_data is None:
        return {"error": "Não foi possível obter dados climáticos para as coordenadas fornecidas."}

    temp, humidity, wind, precipitation = weather_data
    
    # Prepara os dados para os modelos
    # As features devem estar na mesma ordem do treinamento: Temperatura, Umidade, Vento, Precipitacao
    data_for_models = np.array([temp, humidity, wind, precipitation]).reshape(1, -1)
    
    # Previsões individuais
    # Verifica se os modelos estão treinados antes de prever
    pred_rf = 0.5 # Valor padrão se não treinado
    if hasattr(rf_model, 'estimators_') and len(rf_model.estimators_) > 0:
        pred_rf = rf_model.predict_proba(data_for_models)[0][1]
    else:
        print("AVISO: Modelo Random Forest não treinado. Usando probabilidade padrão de 0.5.")

    pred_xgb = 0.5 # Valor padrão se não treinado
    if hasattr(xgb_model, '_Booster'):
        pred_xgb = xgb_model.predict_proba(data_for_models)[0][1]
    else:
        print("AVISO: Modelo XGBoost não treinado. Usando probabilidade padrão de 0.5.")
    
    pred_lstm = 0.5 # Valor padrão se não treinado
    if lstm_scaler is not None and hasattr(lstm_model, 'lstm'):
        # Aplica o scaler nos dados para o LSTM
        scaled_data_lstm = lstm_scaler.transform(data_for_models)
        data_lstm_tensor = torch.tensor(scaled_data_lstm.reshape(-1, 1, scaled_data_lstm.shape[1]), dtype=torch.float32)
        
        # Coloca o modelo LSTM em modo de avaliação
        lstm_model.eval()
        with torch.no_grad():
            pred_lstm = lstm_model(data_lstm_tensor).item() # O modelo já retorna a probabilidade
    else:
        print("AVISO: Modelo LSTM ou scaler não disponível/treinado. Usando probabilidade padrão de 0.5.")

    # Previsão final do ensemble (média das probabilidades)
    ensemble_prediction = (pred_rf + pred_xgb + pred_lstm) / 3

    # Define a probabilidade de enchente com base na previsão do ensemble
    flood_probability_percent = ensemble_prediction * 100 
    flood_probability_percent = min(100, max(0, flood_probability_percent)) # Garante que o valor esteja entre 0 e 100

    # Opcional: Salvar a previsão no histórico
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    # Primeiro, encontre o nome do município mais próximo ou use uma abordagem de identificação
    # Para simplificar, vamos usar uma string combinada lat_lon como identificador para o histórico
    municipio_id = f"{lat},{lon}"
    from datetime import datetime
    data_hora_atual = datetime.now().isoformat()
    cursor.execute("""
        INSERT INTO historico_previsao (municipio, data_hora, probabilidade)
        VALUES (?, ?, ?)
    """, (municipio_id, data_hora_atual, ensemble_prediction)) # Salva a probabilidade bruta (0-1)
    conn.commit()
    conn.close()

    return {
        "lat": lat,
        "lon": lon,
        "probabilidade": ensemble_prediction, # Mude para 'probabilidade' para corresponder ao frontend
        "dados_atuais": {
            "Temperatura": temp,
            "Umidade": humidity,
            "Vento": wind,
            "Precipitacao": precipitation
        }
    }

def predict_historical(lat: float, lon: float, limit: int = 30):
    """
    Busca o histórico de dados e previsões para uma latitude e longitude no banco de dados.
    """
    print(f"DEBUG: Buscando histórico para Lat:{lat}, Lon:{lon}...")
    try:
        conn = sqlite3.connect('database.db')
        # Use o mesmo identificador para o município que você usou ao salvar
        municipio_id = f"{lat},{lon}"
        df = pd.read_sql_query(
            "SELECT data_hora, probabilidade FROM historico_previsao WHERE municipio = ? ORDER BY data_hora DESC LIMIT ?", # DESC e LIMIT para os mais recentes
            conn,
            params=(municipio_id, limit)
        )
        conn.close()

        if df.empty:
            print(f"AVISO: Nenhum dado histórico encontrado para Lat:{lat}, Lon:{lon}.")
            return {"noData": True} # Retorna noData: true para o frontend

        # Converte a coluna de data para o formato necessário pelo front-end
        df['data_hora'] = pd.to_datetime(df['data_hora'])
        
        # O frontend espera 'timestamp' e 'probability'
        history_list = df.sort_values(by='data_hora', ascending=True).apply(lambda row: {
            "timestamp": row['data_hora'].strftime('%d/%m %Hh'),
            "probability": row['probabilidade']
        }, axis=1).tolist()
        
        return history_list # Retorna diretamente a lista de dicionários para o frontend

    except Exception as e:
        print(f"ERRO: Falha ao buscar histórico de previsões para Lat:{lat}, Lon:{lon}: {e}")
        return {"erro": True, "detail": f"Falha interna ao carregar o histórico: {e}"}
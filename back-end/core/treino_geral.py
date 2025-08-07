import numpy as np
import time
import json
import sqlite3
import joblib
import os
from services.weather import get_weather_data
from core.models import rf_model
from core.treino_lstm import treinar_modelo_lstm
from core.treino_xgb import treinar_modelo_xgb

# O caminho para o arquivo municipios.json agora é construído de forma dinâmica
caminho_municipios = os.path.join(os.path.dirname(__file__), '..', '..', 'front-end', 'src', 'municipios.json')

# O arquivo de municípios é lido apenas uma vez na inicialização
todos_municipios = []
try:
    with open(caminho_municipios, 'r', encoding='utf-8') as f:
        dados_json = json.load(f)
        # Assumindo que a lista de municípios está sob a chave 'municipios'
        if 'municipios' in dados_json:
            todos_municipios = dados_json['municipios']
        else:
            # Caso a estrutura seja uma lista direta, tentamos usar o arquivo
            todos_municipios = dados_json
            
except FileNotFoundError:
    print(f"ERRO: Arquivo {caminho_municipios} não encontrado. Verifique a localização do seu arquivo municipios.json.")
except Exception as e:
    print(f"ERRO ao ler o arquivo municipios.json: {e}")


# Esta função irá treinar o modelo de Random Forest
def treinar_modelo_rf():
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute("SELECT temperatura, umidade, vento, precipitacao, enchente FROM clima")
    dados = cursor.fetchall()
    conn.close()

    if len(dados) < 20:
        print("INFO: Dados insuficientes para treinamento RF. Ignorando...")
        return
        
    np.random.shuffle(dados)
    split = int(0.8 * len(dados))
    treino = dados[:split]
    
    X_treino = np.array([d[:4] for d in treino])
    y_treino = np.array([d[4] for d in treino])
    
    print("DEBUG: Treinando o modelo de Random Forest...")
    rf_model.fit(X_treino, y_treino)
    joblib.dump(rf_model, "modelo_rf.pkl")
    print("DEBUG: Treinamento de Random Forest concluído.")

# Esta função orquestra o processo de coleta e treinamento
def iniciar_ciclo_treinamento():
    global todos_municipios

    batch_size = 50 

    while True:
        if not todos_municipios:
            print("INFO: Lista de municípios vazia. Não é possível coletar dados.")
            time.sleep(3600)
            continue

        print("DEBUG: Iniciando ciclo de coleta e treinamento...")
        
        try:
            with open('estado_coleta.txt', 'r') as f:
                start_index = int(f.read().strip())
        except (FileNotFoundError, ValueError):
            start_index = 0
            
        end_index = start_index + batch_size
        municipios_do_lote = todos_municipios[start_index:end_index]
        
        if not municipios_do_lote:
            start_index = 0
            end_index = batch_size
            municipios_do_lote = todos_municipios[start_index:end_index]
        
        conn = sqlite3.connect("database.db")
        cursor = conn.cursor()
        for municipio in municipios_do_lote:
            lat, lon = municipio['lat'], municipio['lon']
            nome = municipio['nome']
            try:
                weather_data = get_weather_data(lat, lon)
                if weather_data:
                    temp, humidity, wind, precipitation = weather_data
                    cursor.execute(
                        "INSERT INTO clima (municipio, temperatura, umidade, vento, precipitacao, enchente) VALUES (?, ?, ?, ?, ?, ?)",
                        (nome, temp, humidity, wind, precipitation, 0)
                    )
                    print(f"INFO: Dados de {nome} coletados e salvos.")
            except Exception as e:
                print(f"ERRO: Falha ao coletar dados para {nome}: {e}")
        conn.commit()
        conn.close()

        with open('estado_coleta.txt', 'w') as f:
            f.write(str(end_index % len(todos_municipios)))

        treinar_modelo_rf()
        treinar_modelo_xgb()
        treinar_modelo_lstm()

        print("DEBUG: Treinamento concluído para todos os modelos.")
        time.sleep(3600)
import numpy as np, time, joblib, sqlite3
from core.models import rf_model

def treinar_modelo_rf():
    # OBSERVAÇÃO: A conexão com o banco de dados deve ser ajustada.
    # No seu código original, a conexão é com 'dados_climaticos.db', mas nos outros scripts é 'database.db'.
    # Para consistência, vou usar 'database.db' aqui.
    conn_local = sqlite3.connect("database.db")
    cursor_local = conn_local.cursor()
    
    while True:
        # 1. Alteração aqui: Adicionar a coluna 'precipitacao' na query SQL
        cursor_local.execute("SELECT temperatura, umidade, vento, precipitacao, enchente FROM clima")
        dados = cursor_local.fetchall()
        
        if len(dados) < 10:
            time.sleep(3600)
            continue
            
        np.random.shuffle(dados)
        split = int(0.8 * len(dados))
        
        # 2. Alteração aqui: Ajustar o slicing para considerar 4 features e 1 label
        X = np.array([d[:4] for d in dados])
        y = np.array([d[4] for d in dados])
        
        rf_model.fit(X[:split], y[:split])
        joblib.dump(rf_model, "modelo_rf.pkl")
        
        conn_local.close() # Adicionar o fechamento da conexão aqui
        time.sleep(3600)
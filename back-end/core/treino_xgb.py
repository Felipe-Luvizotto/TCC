import numpy as np, time, joblib, sqlite3
from core.models import xgb_model

def treinar_modelo_xgb():
    conn_local = sqlite3.connect("database.db")
    cursor_local = conn_local.cursor()
    while True:
        cursor_local.execute("SELECT temperatura, umidade, vento, precipitacao, enchente FROM clima")
        dados = cursor_local.fetchall()
        conn_local.close()

        if len(dados) < 10:
            time.sleep(3600)
            continue
            
        np.random.shuffle(dados)
        split = int(0.8 * len(dados))
        
        X = np.array([d[:4] for d in dados])
        y = np.array([d[4] for d in dados])

        # === NOVO CÓDIGO AQUI: Cálculo e aplicação do scale_pos_weight ===
        count_pos = np.sum(y == 1)
        count_neg = np.sum(y == 0)
        
        if count_pos > 0 and count_neg > 0:
            scale_pos_weight_value = count_neg / count_pos
            xgb_model.set_params(scale_pos_weight=scale_pos_weight_value)
            print(f"XGBoost: Configurado com scale_pos_weight={scale_pos_weight_value:.2f}")
        else:
            # Caso não haja exemplos de uma das classes, desabilite o parâmetro.
            xgb_model.set_params(scale_pos_weight=1) 
            print("XGBoost: Dados desbalanceados sem exemplos de ambas as classes. scale_pos_weight não ajustado.")
        # ===================================================================

        xgb_model.fit(X[:split], y[:split])
        joblib.dump(xgb_model, "modelo_xgb.pkl")
        
        time.sleep(3600)
import numpy as np, time, torch
import sqlite3
from core.models import model
from torch import nn

def treinar_modelo_lstm():
    while True:
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()

        cursor.execute("SELECT temperatura, umidade, vento, precipitacao, enchente FROM clima")
        dados = cursor.fetchall()
        conn.close()

        if len(dados) < 10:
            time.sleep(3600)
            continue
            
        np.random.shuffle(dados)
        split = int(0.8 * len(dados))
        treino, teste = dados[:split], dados[split:]
        
        X_treino = torch.tensor(np.array([d[:4] for d in treino]).reshape(-1, 1, 4), dtype=torch.float32)
        y_treino = torch.tensor(np.array([d[4] for d in treino]).reshape(-1, 1), dtype=torch.float32)
        
        # === NOVO CÓDIGO AQUI: Cálculo e aplicação do pos_weight ===
        count_pos = torch.sum(y_treino == 1).item()
        count_neg = torch.sum(y_treino == 0).item()
        
        if count_pos > 0 and count_neg > 0:
            pos_weight_value = count_neg / count_pos
            pos_weight_tensor = torch.tensor(pos_weight_value, dtype=torch.float32)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
            print(f"LSTM: Configurado com pos_weight={pos_weight_value:.2f}")
        else:
            # Caso não haja exemplos de uma das classes, use o BCELoss padrão.
            criterion = nn.BCELoss()
            print("LSTM: Dados desbalanceados sem exemplos de ambas as classes. pos_weight não ajustado.")
        # ==========================================================
        
        X_teste = torch.tensor(np.array([d[:4] for d in teste]).reshape(-1, 1, 4), dtype=torch.float32)
        y_teste = torch.tensor(np.array([d[4] for d in teste]).reshape(-1, 1), dtype=torch.float32)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        for _ in range(200):
            optimizer.zero_grad()
            # OBSERVAÇÃO: Se você usar BCEWithLogitsLoss, remova a camada sigmoid do seu modelo.
            # O código original do seu `model_lstm.py` já tinha a função de ativação tanh na última camada.
            # Eu presumo que você use BCELoss(). A melhor prática é usar BCEWithLogitsLoss() e remover o sigmoid no forward do modelo.
            loss = criterion(model(X_treino), y_treino)
            loss.backward()
            optimizer.step()
            
        with torch.no_grad():
            y_pred = model(X_teste)
            y_pred_classes = (y_pred > 0.5).float()
            accuracy = (y_pred_classes == y_teste).float().mean().item()
            print(f"Acurácia no conjunto de teste: {accuracy:.4f}")
            
        torch.save(model.state_dict(), "modelo_lstm.pth")
        time.sleep(3600)
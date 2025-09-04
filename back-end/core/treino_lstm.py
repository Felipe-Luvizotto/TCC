import torch
import torch.nn as nn
import torch.optim as optim
import joblib
import numpy as np
import pandas as pd  # Adicionada a importação do pandas
from sklearn.preprocessing import MinMaxScaler
from core.model_lstm import LSTMModel

# CORREÇÃO: A função agora aceita 'columns' como argumento
def treinar_modelo_lstm(X_treino, y_treino, columns):
    print("DEBUG: Iniciando treinamento do modelo LSTM...")

    try:
        # CORREÇÃO: Reconstruir o DataFrame a partir dos arrays NumPy recebidos
        df_treino = pd.DataFrame(X_treino, columns=columns)
        df_treino['enchente'] = y_treino

        # A partir daqui, o código funciona como o original, pois opera sobre um DataFrame
        df_treino.dropna(inplace=True)

        if df_treino.empty:
            print("AVISO: Dataset vazio após a limpeza de dados. Não é possível treinar o modelo LSTM.")
            return

        X_treino_clean = df_treino.drop('enchente', axis=1).values.astype(np.float32)
        y_treino_clean = df_treino['enchente'].values.astype(np.float32)

        # Normalização dos dados
        scaler = MinMaxScaler(feature_range=(0, 1))
        X_treino_scaled = scaler.fit_transform(X_treino_clean)
        joblib.dump(scaler, 'scaler_lstm.pkl')

        # Converte para tensores do PyTorch
        X_treino_tensor = torch.tensor(X_treino_scaled, dtype=torch.float32).unsqueeze(1)
        y_treino_tensor = torch.tensor(y_treino_clean, dtype=torch.float32).unsqueeze(1)

        # Inicializa o modelo, função de perda e otimizador
        input_size = X_treino_scaled.shape[1]
        hidden_layer_size = 50
        num_epochs = 100
        learning_rate = 0.001
        
        # Instancia o modelo corretamente
        model = LSTMModel(input_size=input_size, hidden_size=hidden_layer_size)
        loss_function = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Treinamento
        for i in range(num_epochs):
            if (i + 1) % 10 == 0:
                print(f"  LSTM - Época {i+1}/{num_epochs}")
            optimizer.zero_grad()
            y_pred = model(X_treino_tensor)
            loss = loss_function(y_pred, y_treino_tensor)
            loss.backward()
            optimizer.step()

        # Salva o modelo treinado
        torch.save(model.state_dict(), 'modelo_lstm.pth')
        print("DEBUG: Treinamento do modelo LSTM concluído e salvo.")

    except Exception as e:
        print(f"ERRO: Falha ao treinar ou salvar o modelo LSTM: {e}")
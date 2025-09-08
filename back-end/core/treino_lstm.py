# --- START OF FILE treino_lstm.py ---
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from core.model_lstm import LSTMModel # Importar a classe correta do model_lstm

def treinar_modelo_lstm(X_treino, y_treino, columns):
    print("DEBUG: Iniciando treinamento do modelo LSTM...")

    try:
        # Reconstruir o DataFrame a partir dos arrays NumPy recebidos
        df_treino = pd.DataFrame(X_treino, columns=columns)
        df_treino['Enchente'] = y_treino # Use 'Enchente' conforme o DataFrame final

        df_treino.dropna(inplace=True)

        if df_treino.empty:
            print("AVISO: Dataset vazio após a limpeza de dados para LSTM. Não é possível treinar o modelo LSTM.")
            return

        X_treino_clean = df_treino.drop('Enchente', axis=1).values.astype(np.float32)
        y_treino_clean = df_treino['Enchente'].values.astype(np.float32)

        if X_treino_clean.shape[0] == 0:
            print("AVISO: Nenhuma amostra válida para treinar o LSTM após limpeza. Abortando treinamento.")
            return

        # Normalização dos dados
        scaler = MinMaxScaler(feature_range=(0, 1))
        X_treino_scaled = scaler.fit_transform(X_treino_clean)
        joblib.dump(scaler, 'scaler_lstm.pkl') # Salva o scaler

        # Converte para tensores do PyTorch
        # LSTM espera entrada (batch_size, sequence_length, input_size)
        # Como estamos tratando cada observação como uma sequência de comprimento 1, unsqueeze(1) é apropriado.
        X_treino_tensor = torch.tensor(X_treino_scaled, dtype=torch.float32).unsqueeze(1)
        y_treino_tensor = torch.tensor(y_treino_clean, dtype=torch.float32).unsqueeze(1) # Target também precisa ser 2D

        # Inicializa o modelo, função de perda e otimizador
        input_size = X_treino_scaled.shape[1]
        hidden_layer_size = 50
        num_epochs = 100
        learning_rate = 0.001
        
        # Instancia o modelo corretamente (assumindo LSTMModel de core.model_lstm)
        model = LSTMModel(input_size=input_size, hidden_size=hidden_layer_size)
        loss_function = nn.BCELoss() # Binary Cross Entropy Loss para classificação binária
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Treinamento
        model.train() # Coloca o modelo em modo de treino
        for i in range(num_epochs):
            optimizer.zero_grad()
            y_pred = model(X_treino_tensor)
            loss = loss_function(y_pred, y_treino_tensor)
            loss.backward()
            optimizer.step()
            if (i + 1) % 10 == 0:
                print(f"  LSTM - Época {i+1}/{num_epochs}, Loss: {loss.item():.4f}")

        # Salva o modelo treinado
        torch.save(model.state_dict(), 'modelo_lstm.pth')
        print("DEBUG: Treinamento do modelo LSTM concluído e salvo.")

    except Exception as e:
        print(f"ERRO: Falha ao treinar ou salvar o modelo LSTM: {e}")
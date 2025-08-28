import torch
import torch.nn as nn
import torch.optim as optim
import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Modelo LSTM (mantido como está, pois o problema não é nele)
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq.view(len(input_seq), 1, -1))
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return self.sigmoid(predictions)

def treinar_modelo_lstm(X_treino, y_treino):
    """
    Treina e salva o modelo LSTM.
    """
    print("DEBUG: Iniciando treinamento do modelo LSTM...")

    try:
        # CONVERSÃO E NORMALIZAÇÃO DE DADOS
        X_treino_float = X_treino.astype(np.float32)
        y_treino_float = y_treino.astype(np.float32)
        
        # AQUI ESTÁ A CORREÇÃO: Normaliza os dados para a faixa de 0 a 1
        scaler = MinMaxScaler(feature_range=(0, 1))
        X_treino_scaled = scaler.fit_transform(X_treino_float)

        # Salva o normalizador para ser usado depois na predição
        joblib.dump(scaler, 'scaler_lstm.pkl')

        # Formata os dados de entrada para o LSTM
        X_treino_reshaped = X_treino_scaled.reshape(-1, X_treino_scaled.shape[1])
        y_treino_reshaped = y_treino_float.reshape(-1, 1)

        # Converte para tensores do PyTorch
        X_treino_tensor = torch.tensor(X_treino_reshaped, dtype=torch.float32)
        y_treino_tensor = torch.tensor(y_treino_reshaped, dtype=torch.float32)

        # Hiperparâmetros
        input_size = X_treino_reshaped.shape[1]
        hidden_layer_size = 50
        output_size = 1
        num_epochs = 100
        learning_rate = 0.001

        # Inicializa o modelo, função de perda e otimizador
        model = LSTMModel(input_size, hidden_layer_size, output_size)
        loss_function = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Treinamento
        for i in range(num_epochs):
            optimizer.zero_grad()
            y_pred = model(X_treino_tensor)
            single_loss = loss_function(y_pred, y_treino_tensor)
            single_loss.backward()
            optimizer.step()

        # Salva o modelo treinado
        torch.save(model.state_dict(), "modelo_lstm.pth")
        print("Treinamento do LSTM concluído e modelo salvo.")

    except Exception as e:
        print(f"ERRO: Falha ao treinar ou salvar o modelo LSTM: {e}")
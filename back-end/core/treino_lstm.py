import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from core.models import lstm_model # A CORREÇÃO ESTÁ AQUI
import numpy as np

def treinar_modelo_lstm(X_treino, y_treino):
    """
    Treina o modelo LSTM.
    """
    print("DEBUG: Iniciando treinamento do modelo LSTM...")

    # Ajusta os dados para o formato esperado pelo LSTM (sequência de 1)
    X_treino_reshaped = np.expand_dims(X_treino, axis=1)
    X_treino_tensor = torch.tensor(X_treino_reshaped, dtype=torch.float32)
    y_treino_tensor = torch.tensor(y_treino, dtype=torch.float32).unsqueeze(1)
    
    dataset = TensorDataset(X_treino_tensor, y_treino_tensor)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001)

    num_epochs = 10
    lstm_model.train()
    for epoch in range(num_epochs):
        for inputs, labels in dataloader:
            outputs = lstm_model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
    print(f"DEBUG: Treinamento LSTM concluído. Loss na última época: {loss.item():.4f}")
    
    # Salva o modelo treinado
    torch.save(lstm_model.state_dict(), "modelo_lstm.pth")
    print("DEBUG: Modelo LSTM salvo em 'modelo_lstm.pth'.")
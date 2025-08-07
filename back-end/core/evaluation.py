import numpy as np, sqlite3, torch, joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from core.models import model as lstm_model, rf_model, xgb_model # Importa as instâncias globais

def get_evaluation_data(test_size=0.2):
    # ... (código existente, sem alterações)
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute("SELECT temperatura, umidade, vento, precipitacao, enchente FROM clima")
    dados = cursor.fetchall()
    conn.close()
    
    if not dados or len(dados) < 20:
        return None, None

    np.random.shuffle(dados)
    split_idx = int((1.0 - test_size) * len(dados))
    
    dados_teste = dados[split_idx:]
    
    if not dados_teste:
        return None, None
        
    X_test = np.array([d[:4] for d in dados_teste])
    y_test = np.array([d[4] for d in dados_teste])
    
    return X_test, y_test

def run_ensemble_evaluation():
    # O BLOCO ABAIXO FOI REMOVIDO para evitar duplicação do carregamento.
    # try:
    #     lstm_model.load_state_dict(torch.load("modelo_lstm.pth"))
    #     rf_model = joblib.load("modelo_rf.pkl")
    #     xgb_model = joblib.load("modelo_xgb.pkl")
    #     lstm_model.eval()
    # except Exception as e:
    #     print(f"Erro ao carregar modelos para avaliação: {e}")
    #     return

    X_test, y_test = get_evaluation_data()
    if X_test is None:
        print("Avaliação não pode ser executada. Dados insuficientes.")
        return

    ensemble_predictions = []
    
    for i in range(len(X_test)):
        data_point = X_test[i].reshape(1, -1)
        
        # Previsões individuais
        pred_rf = rf_model.predict_proba(data_point)[0][1]
        pred_xgb = xgb_model.predict_proba(data_point)[0][1]
        
        with torch.no_grad():
            lstm_model.eval()
            data_lstm = torch.tensor(data_point.reshape(-1, 1, 4), dtype=torch.float32)
            pred_lstm = torch.sigmoid(lstm_model(data_lstm)).item()

        # Combinação das previsões (ensemble)
        pred_final = (pred_lstm * 0.5) + (pred_rf * 0.3) + (pred_xgb * 0.2)
        
        ensemble_predictions.append(1 if pred_final > 0.5 else 0)

    ensemble_predictions = np.array(ensemble_predictions)

    accuracy = accuracy_score(y_test, ensemble_predictions)
    precision = precision_score(y_test, ensemble_predictions, zero_division=0)
    recall = recall_score(y_test, ensemble_predictions, zero_division=0)
    f1 = f1_score(y_test, ensemble_predictions, zero_division=0)

    print("\n--- Avaliação do Ensemble ---")
    print(f"Acurácia: {accuracy:.4f}")
    print(f"Precisão: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("-----------------------------")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }
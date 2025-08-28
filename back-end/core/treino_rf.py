import numpy as np
import joblib
from core.models import rf_model

def treinar_modelo_rf(X_treino, y_treino):
    """
    Treina o modelo Random Forest com os dados fornecidos.
    """
    print("DEBUG: Iniciando treinamento do modelo Random Forest...")
    try:
        rf_model.fit(X_treino, y_treino)
        joblib.dump(rf_model, "modelo_rf.pkl")
        print("DEBUG: Treinamento do modelo Random Forest concluído e salvo.")
    except Exception as e:
        print(f"ERRO: Falha ao treinar ou salvar o modelo Random Forest: {e}")
import numpy as np
import joblib
from core.models import xgb_model

def treinar_modelo_xgb(X_treino, y_treino):
    """
    Treina o modelo XGBoost com os dados fornecidos.
    """
    print("DEBUG: Iniciando treinamento do modelo XGBoost...")

    # === Cálculo e aplicação do scale_pos_weight ===
    count_pos = np.sum(y_treino == 1)
    count_neg = np.sum(y_treino == 0)

    if count_pos > 0 and count_neg > 0:
        scale_pos_weight_value = count_neg / count_pos
        xgb_model.set_params(scale_pos_weight=scale_pos_weight_value)
        print(f"XGBoost: Configurado com scale_pos_weight={scale_pos_weight_value:.2f}")
    else:
        # Caso não haja exemplos de uma das classes, desabilite o parâmetro.
        xgb_model.set_params(scale_pos_weight=1)
        print("XGBoost: Dados desbalanceados sem exemplos positivos/negativos. scale_pos_weight desabilitado.")

    # Treinamento do modelo
    try:
        xgb_model.fit(X_treino, y_treino)
        joblib.dump(xgb_model, "modelo_xgb.pkl")
        print("DEBUG: Treinamento do modelo XGBoost concluído e salvo.")
    except Exception as e:
        print(f"ERRO: Falha ao treinar ou salvar o modelo XGBoost: {e}")
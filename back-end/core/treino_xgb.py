import numpy as np
import joblib
from core.models import xgb_model

def treinar_modelo_xgb(X_treino, y_treino):
    print("DEBUG: Iniciando treinamento do modelo XGBoost...")

    count_pos = np.sum(y_treino == 1)
    count_neg = np.sum(y_treino == 0)

    # Verifica se há amostras de ambas as classes para calcular o scale_pos_weight
    if count_pos > 0 and count_neg > 0:
        scale_pos_weight_value = count_neg / count_pos
        xgb_model.set_params(scale_pos_weight=scale_pos_weight_value)
        print(f"XGBoost: Configurado com scale_pos_weight={scale_pos_weight_value:.2f}")
    else:
        # Se não há amostras de uma das classes, desabilita o parâmetro para evitar erros
        xgb_model.set_params(scale_pos_weight=1, base_score=0.5)
        print("XGBoost: Dados desbalanceados. scale_pos_weight e base_score ajustados para padrão.")
    
    try:
        # CORREÇÃO: Removido '.values' pois os dados já são arrays NumPy
        xgb_model.fit(X_treino, y_treino)
        joblib.dump(xgb_model, "modelo_xgb.pkl")
        print("DEBUG: Treinamento do modelo XGBoost concluído e salvo.")
    except Exception as e:
        print(f"ERRO: Falha ao treinar ou salvar o modelo XGBoost: {e}")
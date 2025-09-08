# --- START OF FILE training.py ---
# Este arquivo não será usado na solução proposta,
# pois o treinamento é orquestrado por 'treinamento_acelerado.py'
# Se você deseja implementar treinamento assíncrono no futuro,
# este arquivo pode ser adaptado com um gerenciador de tarefas como Celery ou similar.
# No momento, ele pode ser removido ou ignorado para evitar confusão.
#
# Conteúdo original:
# import threading
# from core.treino_lstm import treinar_modelo_lstm
# from core.treino_rf import treinar_modelo_rf
# from core.treino_xgb import treinar_modelo_xgb
#
# def iniciar_threads_de_treinamento():
#     threading.Thread(target=treinar_modelo_lstm, daemon=True).start()
#     threading.Thread(target=treinar_modelo_rf, daemon=True).start()
#     threading.Thread(target=treinar_modelo_xgb, daemon=True).start()
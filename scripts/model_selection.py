# scripts/model_selection.py

# Этот скрипт будет отвечать за:
# 1. Загрузку лучшей модели.
# 2. Расчет ML метрик на всех фолдах (train/validation) и сохранение в ml_metrics_train.csv.
# 3. Построение графика metric_train.png (AUC).
# 4. Расчет важности признаков для каждого фолда и сохранение в top_10_feature_importance.csv.

print("Скрипт model_selection.py будет запущен после gridsearch.py для анализа результатов CV и сохранения метрик.")
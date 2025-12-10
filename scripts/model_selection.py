# scripts/model_selection.py

import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import TimeSeriesSplit

# Пути и константы
CLEANED_DATA_PATH = os.path.join('data', 'cleaned_features_target.csv')
SELECTED_MODEL_PATH = os.path.join('results', 'selected-model', 'selected_model.pkl')
METRICS_CSV_PATH = os.path.join('results', 'cross-validation', 'ml_metrics_train.csv')
FEATURE_IMPORTANCE_CSV_PATH = os.path.join('results', 'cross-validation', 'top_10_feature_importance.csv')
METRIC_PLOT_PATH = os.path.join('results', 'cross-validation', 'metric_train.png')

MIN_TRAIN_DAYS = int(365.25 * 2)
N_SPLITS_REQ = 10 # Используем 10 фолдов, как в Grid Search

def load_data():
    """Загрузка финального датасета и разделение на train/test."""
    df = pd.read_csv(CLEANED_DATA_PATH, index_col=['date', 'Ticker'], parse_dates=['date'])
    df = df.dropna(subset=['Target'])
    df = df[df['Target'] != 0].copy()
    
    X = df.drop(columns=['Target', 'future_return'])
    y = df['Target'].apply(lambda x: 1 if x > 0 else 0)
    
    features_to_drop = [col for col in X.columns if col in ['open', 'high', 'low', 'close', 'volume']]
    X = X.drop(columns=features_to_drop, errors='ignore')

    test_start_date = pd.to_datetime('2017-01-01')
    X_train = X.loc[X.index.get_level_values('date') < test_start_date]
    y_train = y.loc[y.index.get_level_values('date') < test_start_date]
    
    return X_train, y_train

def create_custom_ts_cv(X):
    """Создание кастомного CV-объекта TimeSeriesSplit."""
    dates = X.index.get_level_values('date').unique().sort_values()
    test_size_min = max(1, (len(dates) - MIN_TRAIN_DAYS) // N_SPLITS_REQ)
    
    ts_cv = TimeSeriesSplit(
        n_splits=N_SPLITS_REQ, 
        test_size=test_size_min,
    )
    
    # Кастомный итератор для обработки мультииндекса и требования MIN_TRAIN_DAYS
    class MultiIndexTimeSeriesSplit:
        def __init__(self, ts_cv, full_dates, full_index):
            self.ts_cv = ts_cv
            self.full_dates = full_dates
            self.full_index = full_index
            self.n_splits = ts_cv.n_splits
            self.valid_splits = []

        def split(self, X_dummy, y_dummy=None, groups=None):
            if not self.valid_splits:
                # X_dummy здесь - это просто np.arange(len(full_dates))
                for train_date_indices, val_date_indices in self.ts_cv.split(X_dummy):
                    
                    if len(train_date_indices) < MIN_TRAIN_DAYS:
                        continue
                    
                    train_dates = self.full_dates[train_date_indices]
                    val_dates = self.full_dates[val_date_indices]
                    
                    train_index = self.full_index.get_level_values('date').isin(train_dates)
                    val_index = self.full_index.get_level_values('date').isin(val_dates)
                    
                    self.valid_splits.append((np.where(train_index)[0], np.where(val_index)[0]))
            
            for train, val in self.valid_splits:
                yield train, val
        
        def get_n_splits(self, X=None, y=None, groups=None):
            # Важно: для сбора метрик мы возвращаем фактическое количество сплитов, 
            # но для GridSearch обычно требуется N_SPLITS_REQ. Здесь для простоты вернем 10.
            return self.n_splits

    return MultiIndexTimeSeriesSplit(ts_cv, dates, X.index)

def calculate_metrics_and_importance(X_train, y_train, custom_cv):
    """
    Обучает лучшую модель на каждом фолде и собирает метрики/важность признаков.
    """
    print("-> Загрузка лучшей модели и подготовка пайплайна...")
    try:
        pipeline = joblib.load(SELECTED_MODEL_PATH)
    except FileNotFoundError:
        print(f"ОШИБКА: Файл модели не найден: {SELECTED_MODEL_PATH}. Убедитесь, что gridsearch.py был запущен успешно.")
        return None, None
        
    # Извлекаем отдельные шаги пайплайна (особенно модель)
    model = pipeline.named_steps['model']
    imputer = pipeline.named_steps['imputer']
    scaler = pipeline.named_steps['scaler']

    # Датафреймы для сбора результатов
    metrics = []
    feature_importance_list = []
    
    # Сброс индекса для совместимости с CV-индексами
    X_flat = X_train.reset_index(drop=True)
    y_flat = y_train.reset_index(drop=True)

    print(f"-> Запуск обучения и оценки на {custom_cv.get_n_splits()} фолдах...")
    
    # Используем кастомный итератор
    for fold_idx, (train_idx, val_idx) in enumerate(custom_cv.split(np.arange(len(X_train.index.get_level_values('date').unique())))):
        
        X_fold_train, X_fold_val = X_flat.iloc[train_idx], X_flat.iloc[val_idx]
        y_fold_train, y_fold_val = y_flat.iloc[train_idx], y_flat.iloc[val_idx]

        # 1. Pipeline: Fit & Transform
        X_train_imputed = imputer.fit_transform(X_fold_train)
        X_val_imputed = imputer.transform(X_fold_val)
        
        X_train_scaled = scaler.fit_transform(X_train_imputed)
        X_val_scaled = scaler.transform(X_val_imputed)

        # 2. Обучение модели
        model.fit(X_train_scaled, y_fold_train)
        
        # 3. Предсказания (вероятности для AUC/LogLoss)
        y_train_pred_proba = model.predict_proba(X_train_scaled)[:, 1]
        y_val_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
        
        # Предсказания (классы для Accuracy)
        y_train_pred = model.predict(X_train_scaled)
        y_val_pred = model.predict(X_val_scaled)
        
        # 4. Расчет метрик
        
        # Метрики на Train
        metrics_train = {
            'Fold': fold_idx + 1,
            'Set': 'Train',
            'AUC': roc_auc_score(y_fold_train, y_train_pred_proba),
            'Accuracy': accuracy_score(y_fold_train, y_train_pred),
            'LogLoss': log_loss(y_fold_train, y_train_pred_proba)
        }
        metrics.append(metrics_train)
        
        # Метрики на Validation
        metrics_val = {
            'Fold': fold_idx + 1,
            'Set': 'Validation',
            'AUC': roc_auc_score(y_fold_val, y_val_pred_proba),
            'Accuracy': accuracy_score(y_fold_val, y_val_pred),
            'LogLoss': log_loss(y_fold_val, y_val_pred_proba)
        }
        metrics.append(metrics_val)
        
        # 5. Важность признаков
        feature_importances = pd.Series(model.feature_importances_, index=X_fold_train.columns)
        top_10 = feature_importances.sort_values(ascending=False).head(10).index.tolist()
        
        feature_importance_list.append({
            'Fold': fold_idx + 1,
            'Top_10_Features': top_10
        })
        
        print(f"  Фолд {fold_idx + 1}/{custom_cv.n_splits}: AUC Train={metrics_train['AUC']:.4f}, AUC Val={metrics_val['AUC']:.4f}")

    # 6. Сохранение метрик
    df_metrics = pd.DataFrame(metrics).set_index(['Fold', 'Set'])
    df_metrics.to_csv(METRICS_CSV_PATH)
    print(f"-> ML метрики сохранены в {METRICS_CSV_PATH}")

    # 7. Сохранение важности признаков
    df_importance = pd.DataFrame(feature_importance_list).set_index('Fold')
    df_importance.to_csv(FEATURE_IMPORTANCE_CSV_PATH)
    print(f"-> Топ-10 признаков сохранены в {FEATURE_IMPORTANCE_CSV_PATH}")
    
    return df_metrics, df_importance

def plot_metrics(df_metrics, metric_name='AUC'):
    """
    Построение графика метрики (как в примере аудита).
    """
    print(f"-> Построение графика метрики: {metric_name}")
    
    df_plot = df_metrics.unstack(level='Set')[metric_name]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Ширина столбца и позиция
    bar_width = 0.35
    folds = df_plot.index
    r1 = np.arange(len(folds))
    r2 = [x + bar_width for x in r1]

    # Гистограмма для Validation Set (зеленый)
    ax.bar(r1, df_plot['Validation'], color='olivedrab', width=bar_width, edgecolor='grey', label='Validation set')
    # Гистограмма для Train Set (серый, с контуром)
    ax.bar(r2, df_plot['Train'], color='lightgrey', width=bar_width, edgecolor='grey', label='Train set')

    # Настройка осей и заголовка
    ax.set_xlabel('Folds', fontweight='bold')
    ax.set_ylabel(metric_name, fontweight='bold')
    ax.set_xticks([r + bar_width/2 for r in r1], [f'Fold {i}' for i in folds])
    ax.set_title(f'{metric_name} на Train и Validation на всех фолдах')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(METRIC_PLOT_PATH)
    plt.close()
    print(f"-> График метрик сохранен в {METRIC_PLOT_PATH}")
    

if __name__ == '__main__':
    X_train, y_train = load_data()
    custom_cv = create_custom_ts_cv(X_train)
    
    df_metrics, df_importance = calculate_metrics_and_importance(X_train, y_train, custom_cv)
    
    if df_metrics is not None:
        plot_metrics(df_metrics, metric_name='AUC')
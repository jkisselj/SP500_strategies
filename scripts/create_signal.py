# scripts/create_signal.py

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Пути и константы
CLEANED_DATA_PATH = os.path.join('data', 'cleaned_features_target.csv')
SELECTED_MODEL_PATH = os.path.join('results', 'selected-model', 'selected_model.pkl')
ML_SIGNAL_PATH = os.path.join('results', 'selected-model', 'ml_signal.csv')

MIN_TRAIN_DAYS = int(365.25 * 2)
N_SPLITS_REQ = 10 

def load_data_full():
    """Загрузка финального датасета (Train + Test для удобства индексации)."""
    df = pd.read_csv(CLEANED_DATA_PATH, index_col=['date', 'Ticker'], parse_dates=['date'])
    df = df.dropna(subset=['Target'])
    df = df[df['Target'] != 0].copy()
    
    # Полный набор данных для генерации сигнала
    X_full = df.drop(columns=['Target', 'future_return'])
    y_full = df['Target'].apply(lambda x: 1 if x > 0 else 0)
    
    # Удаляем OHLCV, оставляем только признаки
    features_to_drop = [col for col in X_full.columns if col in ['open', 'high', 'low', 'close', 'volume']]
    X_full = X_full.drop(columns=features_to_drop, errors='ignore')

    # Разделение на train/test
    test_start_date = pd.to_datetime('2017-01-01')
    X_train = X_full.loc[X_full.index.get_level_values('date') < test_start_date].copy()
    y_train = y_full.loc[y_full.index.get_level_values('date') < test_start_date].copy()
    
    X_test = X_full.loc[X_full.index.get_level_values('date') >= test_start_date].copy()
    
    return X_train, y_train, X_test

def create_custom_ts_cv(X):
    """Создание кастомного CV-объекта TimeSeriesSplit для генерации сигнала на Train set."""
    dates = X.index.get_level_values('date').unique().sort_values()
    test_size_min = max(1, (len(dates) - MIN_TRAIN_DAYS) // N_SPLITS_REQ)
    
    ts_cv = TimeSeriesSplit(
        n_splits=N_SPLITS_REQ, 
        test_size=test_size_min,
    )
    
    # Кастомный итератор (тот же, что использовался для GridSearch)
    class MultiIndexTimeSeriesSplit:
        def __init__(self, ts_cv, full_dates, full_index):
            self.ts_cv = ts_cv
            self.full_dates = full_dates
            self.full_index = full_index
            self.n_splits = ts_cv.n_splits
            self.valid_splits = []

        def split(self, X_dummy, y_dummy=None, groups=None):
            if not self.valid_splits:
                X_dummy_dates = np.arange(len(self.full_dates))
                for train_date_indices, val_date_indices in self.ts_cv.split(X_dummy_dates):
                    
                    if len(train_date_indices) < MIN_TRAIN_DAYS:
                        continue
                    
                    train_dates = self.full_dates[train_date_indices]
                    val_dates = self.full_dates[val_date_indices]
                    
                    # Получаем индексы строк в X
                    train_index = self.full_index.get_level_values('date').isin(train_dates)
                    val_index = self.full_index.get_level_values('date').isin(val_dates)
                    
                    self.valid_splits.append((np.where(train_index)[0], np.where(val_index)[0]))
            
            for train, val in self.valid_splits:
                yield train, val
        
        def get_n_splits(self, X=None, y=None, groups=None):
            return self.n_splits

    return MultiIndexTimeSeriesSplit(ts_cv, dates, X.index)


def generate_ml_signal(X_train, y_train, X_test):
    """
    Генерирует ML-сигнал на Train set (через CV) и на Test set (единичное обучение).
    """
    print("-> Генерация ML-сигнала...")
    try:
        pipeline = joblib.load(SELECTED_MODEL_PATH)
    except FileNotFoundError:
        print(f"ОШИБКА: Файл модели не найден: {SELECTED_MODEL_PATH}. Невозможно сгенерировать сигнал.")
        return None
        
    # Разделение пайплайна на шаги
    model = pipeline.named_steps['model']
    imputer = SimpleImputer(strategy='median') # Создаем новый, чтобы избежать ошибок с fit/transform
    scaler = StandardScaler()

    # Сброс индекса для совместимости с CV-индексами
    X_flat_train = X_train.reset_index(drop=True)
    y_flat_train = y_train.reset_index(drop=True)
    
    # 1. Генерация сигнала на TRAIN SET (через Cross-Validation)
    
    custom_cv = create_custom_ts_cv(X_train)
    signal_train_parts = []
    
    print("  1. Генерация сигнала на TRAIN SET (CV-методом)...")
    
    for fold_idx, (train_idx, val_idx) in enumerate(custom_cv.split(np.arange(len(X_train)))):
        
        X_fold_train, X_fold_val = X_flat_train.iloc[train_idx], X_flat_train.iloc[val_idx]
        y_fold_train = y_flat_train.iloc[train_idx]
        
        # 1. Pipeline: Fit & Transform на трейне, Transform на валидации
        X_train_imputed = imputer.fit_transform(X_fold_train)
        X_val_imputed = imputer.transform(X_fold_val)
        
        X_train_scaled = scaler.fit_transform(X_train_imputed)
        X_val_scaled = scaler.transform(X_val_imputed)

        # 2. Обучение и предсказание
        model.fit(X_train_scaled, y_fold_train)
        y_val_pred_proba = model.predict_proba(X_val_scaled)[:, 1] # Вероятность роста (P=1)
        
        # 3. Сохранение предсказаний
        val_indices_in_original_X = X_train.iloc[val_idx].index # Получаем исходный мульти-индекс
        
        signal_part = pd.Series(y_val_pred_proba, index=val_indices_in_original_X, name='ML_Signal')
        signal_train_parts.append(signal_part)
        
        print(f"    Фолд {fold_idx + 1} сгенерирован (Размер: {len(signal_part)})")

    # Конкатенация сигнала на Train set
    signal_train = pd.concat(signal_train_parts)
    # Удаляем дубликаты (если были) и сортируем по дате
    signal_train = signal_train[~signal_train.index.duplicated(keep='first')].sort_index()

    # 2. Генерация сигнала на TEST SET (единичное обучение на всем Train set)
    
    print("  2. Генерация сигнала на TEST SET (Обучение на всем Train)...")
    
    # Обучение на всем трейн сете (X_train)
    X_full_train_imputed = imputer.fit_transform(X_train)
    X_full_train_scaled = scaler.fit_transform(X_full_train_imputed)
    
    model.fit(X_full_train_scaled, y_train)

    # Предсказание на Test set (X_test)
    X_test_imputed = imputer.transform(X_test)
    X_test_scaled = scaler.transform(X_test_imputed)
    
    y_test_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    signal_test = pd.Series(y_test_pred_proba, index=X_test.index, name='ML_Signal')
    
    # 3. Объединение сигналов
    final_signal = pd.concat([signal_train, signal_test])
    
    # 4. Сохранение
    final_signal.to_csv(ML_SIGNAL_PATH, header=True)
    print(f"-> Финальный ML-сигнал (Train + Test) сохранен в {ML_SIGNAL_PATH} (Размер: {len(final_signal)})")
    
    return final_signal

if __name__ == '__main__':
    X_train, y_train, X_test = load_data_full()
    generate_ml_signal(X_train, y_train, X_test)
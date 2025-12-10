# scripts/gridsearch.py

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import timedelta
import joblib

# Пути
CLEANED_DATA_PATH = os.path.join('data', 'cleaned_features_target.csv')
CV_PLOT_PATH_TS = os.path.join('results', 'cross-validation', 'TimeSeriesSplit.png')
CV_PLOT_PATH_BLOCKING = os.path.join('results', 'cross-validation', 'BlockingTimeSeriesSplit.png')
SELECTED_MODEL_PATH = os.path.join('results', 'selected-model', 'selected_model.pkl')
SELECTED_MODEL_TXT = os.path.join('results', 'selected-model', 'selected_model.txt')

# --- КОНСТАНТЫ ---
MIN_TRAIN_DAYS = int(365.25 * 2) # Минимум 2 года истории

def load_data():
    """Загрузка финального датасета."""
    df = pd.read_csv(CLEANED_DATA_PATH, index_col=['date', 'Ticker'], parse_dates=['date'])
    
    # Удаляем строки, где Target - NaN или 0
    df = df.dropna(subset=['Target'])
    df = df[df['Target'] != 0].copy()
    
    # Разделение на X и y
    X = df.drop(columns=['Target', 'future_return'])
    y = df['Target'].apply(lambda x: 1 if x > 0 else 0)
    
    # Исключаем колонки с OHLCV (на всякий случай)
    features_to_drop = [col for col in X.columns if col in ['open', 'high', 'low', 'close', 'volume']]
    X = X.drop(columns=features_to_drop, errors='ignore')

    # Разделение на train/test по дате (до 2017)
    test_start_date = pd.to_datetime('2017-01-01')
    X_train = X.loc[X.index.get_level_values('date') < test_start_date]
    y_train = y.loc[y.index.get_level_values('date') < test_start_date]

    return X_train, y_train

def plot_cv_splits(cv_method, X, n_splits, name="CV_Split", path=""):
    """Визуализация разбиения кросс-валидации."""
    dates = X.index.get_level_values('date').unique().sort_values()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Даты для отрисовки
    X_dummy = np.arange(len(dates)) 
    
    # Генерируем сплиты с помощью метода split
    splits_gen = list(cv_method.split(X_dummy))
    
    # Если количество фолдов в генераторе отличается от n_splits, используем len(splits_gen)
    n_splits_plot = len(splits_gen)
    
    for i, (train_index_all_rows, val_index_all_rows) in enumerate(splits_gen):
        
        # Получаем уникальные даты, попавшие в этот фолд (по MultiIndex)
        train_dates_in_fold = X.index.get_level_values('date')[train_index_all_rows].unique()
        val_dates_in_fold = X.index.get_level_values('date')[val_index_all_rows].unique()

        # Находим порядковые индексы этих дат относительно всего списка дат
        train_unique_dates_indices = np.where(dates.isin(train_dates_in_fold))[0]
        val_unique_dates_indices = np.where(dates.isin(val_dates_in_fold))[0]
        
        # Отображение
        if len(train_unique_dates_indices) > 0:
            train_start = train_unique_dates_indices.min()
            train_end = train_unique_dates_indices.max()
            ax.barh(i, train_end - train_start + 1, left=train_start, height=0.6, align='center', color='royalblue', label='Train set' if i == 0 else "")

        if len(val_unique_dates_indices) > 0:
            val_start = val_unique_dates_indices.min()
            val_end = val_unique_dates_indices.max()
            ax.barh(i, val_end - val_start + 1, left=val_start, height=0.6, align='center', color='firebrick', label='Validation set' if i == 0 else "")
            
    ax.set_title(name)
    ax.set_xlabel('Sample index (по уникальным датам)')
    ax.set_ylabel('CV iteration')
    ax.set_yticks(np.arange(n_splits_plot))
    
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='lower right')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"График кросс-валидации ({name}) сохранен в {path}")
    

def blocking_time_series_split(X, y, n_splits=10, min_train_years=2):
    """Реализация Blocking Time Series Split."""
    dates = X.index.get_level_values('date').unique().sort_values()
    
    min_train_days = int(365.25 * min_train_years)
    
    if dates[min_train_days:].empty:
         raise ValueError(f"Недостаточно данных. Требуется минимум {min_train_years} лет ({min_train_days} дней).")
        
    train_size_dates = min_train_days
    val_size_dates = int(train_size_dates / 10) 
    
    total_available_for_shift = len(dates) - train_size_dates - val_size_dates
    
    # Логика расчета n_splits и step для Blocked CV (остается без изменений)
    if n_splits > 1:
        step = total_available_for_shift // (n_splits - 1)
        if step <= 0:
            n_splits = max(2, total_available_for_shift // (val_size_dates + 1) + 1)
            step = total_available_for_shift // (n_splits - 1)
    else:
        step = 0
        n_splits = 1

    print(f"Используется Blocking CV: Folds={n_splits}, Train Days={train_size_dates}, Validation Days={val_size_dates}, Step={step}")
    
    splits = []
    
    for i in range(n_splits):
        start_idx = i * step
        
        train_idx_dates = np.arange(start_idx, start_idx + train_size_dates)
        val_idx_dates = np.arange(start_idx + train_size_dates, start_idx + train_size_dates + val_size_dates)
        
        if val_idx_dates.max() >= len(dates):
            break

        train_dates = dates[train_idx_dates]
        val_dates = dates[val_idx_dates]
        
        train_index = X.index.get_level_values('date').isin(train_dates)
        val_index = X.index.get_level_values('date').isin(val_dates)
        
        splits.append((np.where(train_index)[0], np.where(val_index)[0]))
        
    n_splits_final = len(splits)

    class BlockingCV:
        def __init__(self, splits, n_splits):
            self.splits = splits
            self.n_splits = n_splits
        def split(self, X_dummy, y_dummy=None, groups=None):
            for train, val in self.splits:
                yield train, val
        def get_n_splits(self, X=None, y=None, groups=None):
            return self.n_splits
    
    blocking_cv_obj = BlockingCV(splits, n_splits_final)
    
    plot_cv_splits(blocking_cv_obj, X, n_splits=n_splits_final, name="BlockingTimeSeriesSplit", path=CV_PLOT_PATH_BLOCKING)

    return blocking_cv_obj

def run_grid_search(X, y):
    """
    Запуск поиска по сетке с использованием TimeSeriesSplit.
    """
    print("-> Запуск Grid Search...")
    
    n_splits_req = 10
    dates = X.index.get_level_values('date').unique().sort_values()
    
    # 1. Time Series Split (для GridSearch)
    
    # Рассчитываем размер тестового фолда
    test_size_min = max(1, (len(dates) - MIN_TRAIN_DAYS) // n_splits_req)
    
    # Используем TimeSeriesSplit без min_train_size (ИСПРАВЛЕНО)
    ts_cv = TimeSeriesSplit(
        n_splits=n_splits_req, 
        test_size=test_size_min,
        # max_train_size=None # Оставим по умолчанию
    ) 
    
    # Создание кастомного CV-итератора для MultiIndex (ИСПРАВЛЕНО: добавлена проверка min_train_size)
    class MultiIndexTimeSeriesSplit:
        def __init__(self, ts_cv, full_dates, full_index):
            self.ts_cv = ts_cv
            self.full_dates = full_dates
            self.full_index = full_index
            self.n_splits = ts_cv.n_splits
            self.valid_splits = [] # Для хранения только валидных сплитов

        def split(self, X_dummy, y_dummy=None, groups=None):
            if not self.valid_splits:
                # Генерация сплитов только один раз
                for train_date_indices, val_date_indices in self.ts_cv.split(X_dummy):
                    
                    # Ручная проверка: Обучающий набор должен быть больше 2 лет (MIN_TRAIN_DAYS)
                    if len(train_date_indices) < MIN_TRAIN_DAYS:
                        # Пропускаем фолды с недостаточной историей
                        # print(f"Пропущен фолд: история {len(train_date_indices)} дней, требуется минимум {MIN_TRAIN_DAYS}.")
                        continue
                    
                    train_dates = self.full_dates[train_date_indices]
                    val_dates = self.full_dates[val_date_indices]
                    
                    train_index = self.full_index.get_level_values('date').isin(train_dates)
                    val_index = self.full_index.get_level_values('date').isin(val_dates)
                    
                    self.valid_splits.append((np.where(train_index)[0], np.where(val_index)[0]))
            
            # Возвращаем только валидные сплиты
            for train, val in self.valid_splits:
                yield train, val
        
        def get_n_splits(self, X=None, y=None, groups=None):
            # Мы не можем знать n_splits до первого запуска split, но GridSearchCV это не нравится.
            # Мы должны вернуть n_splits_req, а на графике будет видно, сколько фолдов реально используется.
            # Если scikit-learn старый, он может выдать ошибку, если split возвращает меньше, чем get_n_splits.
            # Но попробуем пока вернуть то, что мы запросили.
            return self.n_splits

    custom_ts_cv = MultiIndexTimeSeriesSplit(ts_cv, dates, X.index)
    
    # Визуализируем разбиение
    # Внимание: plot_cv_splits теперь вызовет split, и мы увидим, сколько фолдов реально используется
    plot_cv_splits(custom_ts_cv, X, n_splits=n_splits_req, name="TimeSeriesSplit", path=CV_PLOT_PATH_TS)

    # 2. Pipeline
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1))
    ])

    # 3. Hyperparameter Grid (Уменьшенная сетка для скорости)
    param_grid = {
        'model__n_estimators': [50, 100],
        'model__learning_rate': [0.1],
        'model__max_depth': [10]
    } 

    # 4. Grid Search
    grid_search = GridSearchCV(
        pipeline, 
        param_grid, 
        cv=custom_ts_cv, # Используем наш кастомный CV-объект
        scoring='roc_auc', 
        n_jobs=-1, 
        verbose=2
    )
    
    # Сброс мультииндекса для GridSearchCV
    X_flat = X.reset_index(drop=True)
    y_flat = y.reset_index(drop=True)
    
    print("Начало обучения GridSearch. Это может занять несколько минут...")
    grid_search.fit(X_flat, y_flat)

    # 5. Сохранение лучшей модели и параметров
    best_pipeline = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    print(f"\nЛучшая AUC: {grid_search.best_score_:.4f}")
    print(f"Лучшие параметры: {best_params}")
    
    joblib.dump(best_pipeline, SELECTED_MODEL_PATH)
    print(f"Лучшая модель сохранена в {SELECTED_MODEL_PATH}")

    with open(SELECTED_MODEL_TXT, 'w') as f:
        f.write(str(best_params))
    print(f"Параметры сохранены в {SELECTED_MODEL_TXT}")
    
    # 6. График Blocking CV 
    print("\nГенерация Blocking Time Series Split...")
    # blocking_cv_obj теперь сам регулирует n_splits
    blocking_cv_obj = blocking_time_series_split(X, y, n_splits=n_splits_req, min_train_years=2)

    return grid_search

if __name__ == '__main__':
    X_train, y_train = load_data()
    grid_search_results = run_grid_search(X_train, y_train)
    
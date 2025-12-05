
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier # Пример модели
from sklearn.model_selection import GridSearchCV
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import timedelta

# Пути
CLEANED_DATA_PATH = os.path.join('data', 'cleaned_features_target.csv')
CV_PLOT_PATH_TS = os.path.join('results', 'cross-validation', 'TimeSeriesSplit.png')
CV_PLOT_PATH_BLOCKING = os.path.join('results', 'cross-validation', 'BlockingTimeSeriesSplit.png')
SELECTED_MODEL_PATH = os.path.join('results', 'selected-model', 'selected_model.pkl')
SELECTED_MODEL_TXT = os.path.join('results', 'selected-model', 'selected_model.txt')

def load_data():
    """Загрузка финального датасета."""
    df = pd.read_csv(CLEANED_DATA_PATH, index_col=['Date', 'Ticker'], parse_dates=['Date'])
    # Удаляем строки, где Target - NaN (повторно, если остались) или 0 (если хотим бинарную классификацию)
    df = df.dropna(subset=['Target'])
    df = df[df['Target'] != 0].copy() # Убираем дни с нулевым возвратом
    
    # Разделение на X и y
    X = df.drop(columns=['Target'])
    y = df['Target'].apply(lambda x: 1 if x > 0 else 0) # Цель: 1 (рост), 0 (падение)
    
    # Исключаем колонки с OHLCV, оставляем только признаки
    features_to_drop = [col for col in X.columns if col in ['open', 'high', 'low', 'close', 'volume']]
    X = X.drop(columns=features_to_drop)

    # Разделение на train/test по дате (до 2017)
    test_start_date = pd.to_datetime('2017-01-01')
    X_train = X.loc[X.index.get_level_values('Date') < test_start_date]
    y_train = y.loc[y.index.get_level_values('Date') < test_start_date]

    return X_train, y_train

def plot_cv_splits(cv_method, X, n_splits=10, name="CV_Split", path=""):
    """
    Визуализация разбиения кросс-валидации.
    (Адаптировано для многомерного индекса)
    """
    dates = X.index.get_level_values('Date').unique().sort_values()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Используем только уникальные даты для разбиения
    date_indices = np.arange(len(dates))
    
    for i, (train_index, val_index) in enumerate(cv_method.split(date_indices)):
        train_dates = dates[train_index]
        val_dates = dates[val_index]

        # Вычисление временных позиций для графика (в долях или по порядку)
        # Для простоты используем порядковый индекс
        train_start = date_indices[train_index[0]]
        train_end = date_indices[train_index[-1]]
        val_start = date_indices[val_index[0]]
        val_end = date_indices[val_index[-1]]
        
        # Отображение тренировочной части (голубой)
        ax.barh(i, train_end - train_start + 1, left=train_start, height=0.6, align='center', color='royalblue', label='Train set' if i == 0 else "")
        # Отображение валидационной части (красный)
        ax.barh(i, val_end - val_start + 1, left=val_start, height=0.6, align='center', color='firebrick', label='Validation set' if i == 0 else "")

    ax.set_title(name)
    ax.set_xlabel('Sample index (по датам)')
    ax.set_ylabel('CV iteration')
    ax.set_yticks(np.arange(n_splits))
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"График кросс-валидации ({name}) сохранен в {path}")

def blocking_time_series_split(X, y, n_splits=10, min_train_years=2):
    """
    Реализация Blocking Time Series Split.
    Каждая фолд - это фиксированный блок, который сдвигается.
    """
    dates = X.index.get_level_values('Date').unique().sort_values()
    date_indices = np.arange(len(dates))
    
    # Минимальное количество дней для train set (2 года)
    min_train_days = int(365.25 * min_train_years)
    
    # Проверка, что первые 2 года истории есть
    if len(dates) < min_train_days:
        raise ValueError(f"Недостаточно данных. Требуется минимум {min_train_years} лет ({min_train_days} дней), имеется {len(dates)} дней.")

    # Размер шага и размер блока
    total_dates = len(dates)
    # Размер блока для валидации (для 10 фолдов)
    val_size = int(total_dates * 0.05) # Например, 5% для валидации
    train_size = int(total_dates * 0.15) # Например, 15% для обучения
    
    # Шаг для сдвига (чтобы получить 10 фолдов)
    step = (total_dates - train_size - val_size) // (n_splits - 1)
    
    splits = []
    
    for i in range(n_splits):
        # Начальный индекс блока
        start_idx = i * step
        
        # Индексы обучения и валидации в рамках уникальных дат
        train_idx_dates = np.arange(start_idx, start_idx + train_size)
        val_idx_dates = np.arange(start_idx + train_size, start_idx + train_size + val_size)
        
        if val_idx_dates.max() >= total_dates:
            # Прекращаем, если блок выходит за пределы данных
            break

        # Конвертируем индексы дат обратно в мультииндексы X
        train_dates = dates[train_idx_dates]
        val_dates = dates[val_idx_dates]
        
        # Получаем индексы строк в X, соответствующие этим датам
        train_index = X.index.get_level_values('Date').isin(train_dates)
        val_index = X.index.get_level_values('Date').isin(val_dates)
        
        splits.append((np.where(train_index)[0], np.where(val_index)[0]))

    # Plotting: используем упрощенный plot_cv_splits, но с логикой Blocking
    class BlockingCV:
        def __init__(self, splits):
            self.splits = splits
            self.n_splits = len(splits)
        def split(self, X):
            for train, val in self.splits:
                yield train, val
    
    # plot_cv_splits(BlockingCV(splits), X, n_splits=len(splits), name="BlockingTimeSeriesSplit", path=CV_PLOT_PATH_BLOCKING)

    return BlockingCV(splits) # Возвращаем объект CV

def run_grid_search(X, y):
    """
    Запуск поиска по сетке с использованием TimeSeriesSplit.
    """
    print("-> Запуск Grid Search...")
    
    # Параметры CV
    n_splits = 10
    test_size_ts = int(len(X.index.get_level_values('Date').unique()) / n_splits) # Размер валидационного набора
    
    # 1. Time Series Split (для GridSearch)
    # Используем TimeSeriesSplit на *уникальных* датах
    dates = X.index.get_level_values('Date').unique().sort_values()
    ts_cv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size_ts)
    
    # Создание кастомного CV-итератора для MultiIndex
    class MultiIndexTimeSeriesSplit:
        def __init__(self, ts_cv, full_dates, full_index):
            self.ts_cv = ts_cv
            self.full_dates = full_dates
            self.full_index = full_index
            self.n_splits = ts_cv.n_splits

        def split(self, X_dummy, y_dummy=None, groups=None):
            for train_date_indices, val_date_indices in self.ts_cv.split(self.full_dates):
                train_dates = self.full_dates[train_date_indices]
                val_dates = self.full_dates[val_date_indices]
                
                # Получаем индексы строк в X, соответствующие этим датам
                train_index = self.full_index.get_level_values('Date').isin(train_dates)
                val_index = self.full_index.get_level_values('Date').isin(val_dates)
                
                yield np.where(train_index)[0], np.where(val_index)[0]

    custom_ts_cv = MultiIndexTimeSeriesSplit(ts_cv, dates, X.index)
    plot_cv_splits(custom_ts_cv, X, n_splits=n_splits, name="TimeSeriesSplit", path=os.path.join('results', 'cross-validation', 'TimeSeriesSplit.png'))

    # 2. Pipeline
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')), # Обработка NaN
        ('scaler', StandardScaler()),                 # Масштабирование
        ('model', LGBMClassifier(random_state=42, n_jobs=-1)) # Модель
    ])

    # 3. Hyperparameter Grid
    param_grid = {
        'model__n_estimators': [50, 100],
        'model__learning_rate': [0.05, 0.1],
        'model__max_depth': [-1, 10]
    }

    # 4. Grid Search
    # Используем 'roc_auc' как метрику для бинарной классификации
    grid_search = GridSearchCV(
        pipeline, 
        param_grid, 
        cv=custom_ts_cv, 
        scoring='roc_auc', 
        n_jobs=-1, 
        verbose=1
    )
    
    # Запускаем поиск по сетке
    # Сброс мультииндекса для GridSearchCV
    X_flat = X.reset_index(drop=True)
    y_flat = y.reset_index(drop=True)
    
    grid_search.fit(X_flat, y_flat)

    # 5. Сохранение лучшей модели и параметров
    best_pipeline = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    print(f"\nЛучшая AUC: {grid_search.best_score_:.4f}")
    print(f"Лучшие параметры: {best_params}")
    
    # Сохранение модели
    import joblib
    joblib.dump(best_pipeline, SELECTED_MODEL_PATH)
    print(f"Лучшая модель сохранена в {SELECTED_MODEL_PATH}")

    # Сохранение параметров
    with open(SELECTED_MODEL_TXT, 'w') as f:
        f.write(str(best_params))
    print(f"Параметры сохранены в {SELECTED_MODEL_TXT}")
    
    # **Дополнительно: График Blocking CV для аудита**
    # Для целей аудита необходимо построить оба графика.
    blocking_cv_obj = blocking_time_series_split(X, y, n_splits=n_splits, min_train_years=2)
    plot_cv_splits(blocking_cv_obj, X, n_splits=blocking_cv_obj.n_splits, name="BlockingTimeSeriesSplit", path=os.path.join('results', 'cross-validation', 'BlockingTimeSeriesSplit.png'))

    return grid_search

if __name__ == '__main__':
    X_train, y_train = load_data()
    grid_search_results = run_grid_search(X_train, y_train)
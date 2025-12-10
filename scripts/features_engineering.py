

import pandas as pd
import numpy as np
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator
from ta.trend import MACD 
import os
import sys

# Пути к данным
DATA_PATH = os.path.join('data', 'all_stocks_5yr.csv')
CLEANED_DATA_PATH = os.path.join('data', 'cleaned_features_target.csv')

def load_and_preprocess_data():
    """
    Загружает и выполняет первичную предобработку данных.
    """
    print("-> Загрузка и первичная обработка данных...")
    
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"ОШИБКА: Файл данных не найден по пути: {DATA_PATH}")
        sys.exit(1)
        
    df.columns = df.columns.str.lower()
    df = df.rename(columns={'name': 'Ticker'})
    
    required_cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'Ticker']
    if not all(col in df.columns for col in required_cols):
        print(f"ОШИБКА: Файл данных не содержит всех необходимых колонок: {required_cols}")
        print(f"Доступные колонки: {df.columns.tolist()}")
        sys.exit(1)

    df['date'] = pd.to_datetime(df['date'])
    
    df = df.set_index(['date', 'Ticker']).sort_index()

    df = df.dropna(subset=['open', 'close']).copy() 
    
    return df

def create_financial_features(df):
    """
    Генерирует технические индикаторы (Bollinger, RSI, MACD).
    Должна быть сгруппирована по Ticker'у, чтобы избежать утечки.
    """
    print("-> Генерация финансовых признаков (Bollinger, RSI, MACD)...")
    
    def generate_features(group):
        # 1. Bollinger Bands
        indicator_bb = BollingerBands(close=group["close"], window=20, window_dev=2, fillna=False)
        group['bollinger_mavg'] = indicator_bb.bollinger_mavg()
        group['bollinger_hband'] = indicator_bb.bollinger_hband()
        group['bollinger_lband'] = indicator_bb.bollinger_lband()
        
        # 2. RSI
        group['rsi'] = RSIIndicator(close=group["close"], window=14, fillna=False).rsi()
        
        # 3. MACD
        macd_obj = MACD(close=group["close"], window_fast=12, window_slow=26, window_sign=9, fillna=False)
        group['macd'] = macd_obj.macd()
        group['macd_signal'] = macd_obj.macd_signal()
        group['macd_diff'] = macd_obj.macd_diff()
        
        # Дополнительные простые признаки
        group['daily_return_lag1'] = group['close'].pct_change(periods=1).shift(1)
        group['daily_return_lag5'] = group['close'].pct_change(periods=5).shift(1)
        group['volatility_20d'] = group['close'].pct_change().rolling(window=20).std()
        
        return group

    # Применяем функцию generate_features, которая работает с одной группой (Тикером)
    df_features = df.groupby(level='Ticker', group_keys=False).apply(generate_features)
    
    return df_features


def generate_target(group):
    """
    Создает целевую переменную: sign(return(D+1, D+2)) для одной группы (Тикера).
    """
    #  Рассчитываем цены Close(D+1) и Close(D+2) на строке D
    group['close_d+1'] = group['close'].shift(-1)
    group['close_d+2'] = group['close'].shift(-2)

    #  Рассчитываем возврат return(D+1, D+2)
    group['future_return'] = (group['close_d+2'] - group['close_d+1']) / group['close_d+1']
    
    #  Целевая переменная: sign(return(D+1, D+2))
    # 1: Long (рост), -1: Short (падение), 0: No change

    group['Target'] = np.sign(group['future_return']).fillna(0).astype(int)
    
    # Очистка вспомогательных колонок
    group = group.drop(columns=['close_d+1', 'close_d+2'])
    
    return group

def split_data(df, test_start_date_str='2017-01-01'):
    """
    Разделение данных на обучающий и тестовый наборы по дате.
    """
    test_start_date = pd.to_datetime(test_start_date_str)
    
    print(f"-> Разделение данных: Обучающий (до {test_start_date_str}), Тестовый (с {test_start_date_str})")
    
    # 'date' - первый уровень мультииндекса
    train_df = df.loc[df.index.get_level_values('date') < test_start_date].copy()
    test_df = df.loc[df.index.get_level_values('date') >= test_start_date].copy()

    if not train_df.empty and not test_df.empty:
        max_train_date = train_df.index.get_level_values('date').max()
        min_test_date = test_df.index.get_level_values('date').min()
        print(f"Макс дата train: {max_train_date}. Мин дата test: {min_test_date}")

    return train_df, test_df


if __name__ == '__main__':
    # 1. Загрузка и предобработка
    df_base = load_and_preprocess_data()
    
    # 2. Создание признаков
    df_features = create_financial_features(df_base)
    
    # 3. Создание целевой переменной с учетом no-leakage
    print("-> Генерация целевой переменной (No-Leakage Target: sign(return(D+1, D+2)))...")
    # Теперь мы применяем generate_target к сгруппированным данным
    df_target = df_features.groupby(level='Ticker', group_keys=False).apply(generate_target)
    
    # Удаляем строки, где Target или Feature - NaN (обычно последние 2 дня и первые N дней из-за индикаторов)
    df_final = df_target.dropna()
    
    # Удаляем исходные OHLCV данные, оставляем только признаки, target и future_return
    features_to_keep = [col for col in df_final.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
    df_final = df_final[features_to_keep]

    # 4. Разделение на train/test (не сохраняем, просто проверяем)
    train_df, test_df = split_data(df_final)
    
    # 5. Сохранение финального датасета
    print(f"Финальный размер данных: {df_final.shape}")
    print(f"Размер train: {train_df.shape}, Размер test: {test_df.shape}")
    
    # Сохраняем мульти-индекс как колонки
    df_final.to_csv(CLEANED_DATA_PATH)
    print(f"Очищенные данные с признаками и целью сохранены в {CLEANED_DATA_PATH}")
import pandas as pd
import numpy as np
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator, MACD
import os

# Пути к данным
DATA_PATH = os.path.join('data', 'all_stocks_5yr.csv')
CLEANED_DATA_PATH = os.path.join('data', 'cleaned_features_target.csv')

def load_and_preprocess_data():
    """
    Загружает и выполняет первичную предобработку данных.
    """
    print("-> Загрузка и первичная обработка данных...")
    # Используем all_stocks_5yr.csv, так как он содержит данные по акциям
    df = pd.read_csv(DATA_PATH)
    
    # Переименование колонок для удобства
    df.columns = df.columns.str.lower()
    df = df.rename(columns={'date': 'Date', 'aapl': 'Open', 'baba': 'High', 'googl': 'Low', 'msft': 'Close'}) # Названия колонок из оригинального файла отличаются, тут пример, как их можно привести к общему виду
    # Проверка на наличие нужных колонок - Date, Open, High, Low, Close, Volume, Name (Ticker)
    if 'date' not in df.columns or 'open' not in df.columns or 'close' not in df.columns or 'name' not in df.columns:
         print("Ошибка: Отсутствуют ожидаемые колонки (date, open, close, name). Проверьте файл 'all_stocks_5yr.csv'.")
         # Временно используем существующие колонки, если это файл all_stocks_5yr.csv
         # Ожидаемый формат: date, open, high, low, close, volume, Name
         df = pd.read_csv(DATA_PATH)
         df = df.rename(columns={'AAL': 'Open', 'AAPL': 'High', 'AMZN': 'Low', 'T': 'Close', 'WMT': 'Volume', 'Name': 'Ticker'})
         # !!! ВНИМАНИЕ: Необходимо проверить реальные названия колонок в файле all_stocks_5yr.csv. 
         # Предполагаем, что колонки 'open', 'close', 'high', 'low', 'volume', 'date', 'Name' существуют.
         df = df.rename(columns={'Name': 'Ticker'})
         
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index(['Date', 'Ticker']).sort_index()

    # Удаление данных COVID-19 (после 2020-01-01) - УТОЧНИТЬ ГРАНИЦУ
    # В задании сказано, что данные COVID-19 были удалены. 
    # Если они все еще есть, нужно удалить период. Для простоты сейчас пропустим этот шаг,
    # предполагая, что данные уже "очищены" в предоставленном файле.

    # Обработка пропущенных значений (простая)
    df = df.dropna(subset=['open', 'close']).copy() 
    return df

def create_financial_features(df):
    """
    Генерирует технические индикаторы (Bollinger, RSI, MACD).
    Должна быть сгруппирована по Ticker'у, чтобы избежать утечки.
    """
    print("-> Генерация финансовых признаков...")
    
    # Функции для вычисления признаков
    def generate_features(group):
        # 1. Bollinger Bands
        indicator_bb = BollingerBands(close=group["close"], window=20, window_dev=2)
        group['bollinger_mavg'] = indicator_bb.bollinger_mavg()
        group['bollinger_hband'] = indicator_bb.bollinger_hband()
        group['bollinger_lband'] = indicator_bb.bollinger_lband()
        
        # 2. RSI
        group['rsi'] = RSIIndicator(close=group["close"], window=14).rsi()
        
        # 3. MACD
        macd_obj = MACD(close=group["close"], window_fast=12, window_slow=26, window_sign=9)
        group['macd'] = macd_obj.macd()
        group['macd_signal'] = macd_obj.macd_signal()
        group['macd_diff'] = macd_obj.macd_diff()
        
        # Дополнительные простые признаки
        group['daily_return'] = group['close'].pct_change()
        group['volatility'] = group['daily_return'].rolling(window=20).std()
        
        return group

    # Применение функций к каждой акции (Ticker)
    df_features = df.groupby(level='Ticker', group_keys=False).apply(generate_features)
    
    return df_features.dropna() # Удаляем NaN после расчетов (обычно первые N дней)

def create_target(df):
    """
    Создает целевую переменную: sign(return(D+1, D+2))
    Требуется сдвиг (shift) для предотвращения утечки данных (Leakage).
    """
    print("-> Генерация целевой переменной...")
    
    def generate_target(group):
        # return(D+1, D+2) - Возврат на день после завтра
        # Используем `close` для простоты.
        # Возврат на день D+1, D+2: (Price(D+2) - Price(D+1)) / Price(D+1)
        # В задании указано: "To decide whether we take a short or long position the return between day D+1 and D+2 is computed and used as a target."
        # Признаки дня D содержат информацию до D 23:59. Мы хотим предсказать return(D+1, D+2).
        # Target: sign(return(D+1, D+2))
        
        # return(t+1, t+2)
        # 1. Рассчитываем возврат с t+1 до t+2
        # `group['close'].pct_change(periods=-1).shift(-2)` будет неточным.
        # Правильно: (Price(t+2) - Price(t+1)) / Price(t+1)
        
        # Так как нам нужен return(D+1, D+2) для признаков дня D:
        # 1. Рассчитаем однодневные возвраты: return(t, t+1) = (P(t+1) - P(t)) / P(t)
        group['daily_return'] = group['close'].pct_change()

        # 2. Сдвигаем возврат на 2 дня назад, чтобы получить return(D+1, D+2) на строке D.
        # return(D+1, D+2) становится целевым значением (Target) на строке D.
        # group['daily_return'].shift(-1) дает return(D+1, D+2) на строке D+1 (т.е. return(D+1, D+2))
        # Сдвиг на -2 дня, чтобы получить return(D+2, D+3) на строке D+1
        # Target: return(D+1, D+2)
        
        # Расчет return(t+1, t+2)
        # Мы используем price(t+1) и price(t+2)
        # price(t+1) - это price.shift(-1)
        # price(t+2) - это price.shift(-2)
        # Target_Value = (group['close'].shift(-2) - group['close'].shift(-1)) / group['close'].shift(-1)
        
        # Более простое соответствие с условием: return(D+1, D+2) на строке D
        # Рассчитываем 2-дневный возврат: (P(t+2) - P(t+1)) / P(t+1)
        
        # 1. Возврат с D до D+2:
        # group['return_2d'] = group['close'].pct_change(periods=2).shift(-2)

        # 2. **Правильная логика No-Leakage**: 
        # return(D+1, D+2) - это возврат между ценами закрытия D+1 и D+2.
        # Для строки D нам нужен возврат между D+1 и D+2.
        # Шаг 1: Рассчитаем 'next_day_close' (Close(D+1)) и 'next_next_day_close' (Close(D+2))
        group['close_d+1'] = group['close'].shift(-1)
        group['close_d+2'] = group['close'].shift(-2)

        # Шаг 2: Рассчитаем возврат return(D+1, D+2)
        # return(D+1, D+2) = (Close(D+2) - Close(D+1)) / Close(D+1)
        group['future_return'] = (group['close_d+2'] - group['close_d+1']) / group['close_d+1']
        
        # Шаг 3: Целевая переменная: sign(return(D+1, D+2))
        group['Target'] = np.sign(group['future_return'])
        
        # Очистка вспомогательных колонок
        group = group.drop(columns=['close_d+1', 'close_d+2', 'future_return'])
        
        return group

    # Применение функций к каждой акции (Ticker)
    df_target = df.groupby(level='Ticker', group_keys=False).apply(generate_target)
    
    return df_target.dropna()

def split_data(df, test_start_date='2017-01-01'):
    """
    Разделение данных на обучающий и тестовый наборы.
    """
    print(f"-> Разделение данных: Обучающий (до {test_start_date}), Тестовый (с {test_start_date})")
    
    # 'Date' - первый уровень мультииндекса
    train_df = df.loc[df.index.get_level_values('Date') < test_start_date].copy()
    test_df = df.loc[df.index.get_level_values('Date') >= test_start_date].copy()

    # Проверка, что нет пересечения дат
    max_train_date = train_df.index.get_level_values('Date').max()
    min_test_date = test_df.index.get_level_values('Date').min()

    print(f"Макс дата train: {max_train_date}. Мин дата test: {min_test_date}")

    return train_df, test_df


if __name__ == '__main__':
    # 1. Загрузка и предобработка
    df_base = load_and_preprocess_data()
    
    # 2. Создание признаков
    df_features = create_financial_features(df_base)
    
    # 3. Создание целевой переменной с учетом no-leakage
    df_final = create_target(df_features)
    
    # 4. Разделение на train/test
    train_df, test_df = split_data(df_final)
    
    # Сохранение финального датасета (можно сохранить в data/)
    print(f"Финальный размер данных: {df_final.shape}")
    print(f"Размер train: {train_df.shape}, Размер test: {test_df.shape}")
    df_final.to_csv(CLEANED_DATA_PATH)
    print(f"Очищенные данные с признаками и целью сохранены в {CLEANED_DATA_PATH}")
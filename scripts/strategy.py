# scripts/strategy.py

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# --- Пути и константы ---
CLEANED_DATA_PATH = os.path.join('data', 'cleaned_features_target.csv')
INDEX_DATA_PATH = os.path.join('data', 'HistoricalPrices.csv') 
ML_SIGNAL_PATH = os.path.join('results', 'selected-model', 'ml_signal.csv')
STRATEGY_PLOT_PATH = os.path.join('results', 'strategy', 'strategy.png')
RESULTS_CSV_PATH = os.path.join('results', 'strategy', 'results.csv')
REPORT_PATH = os.path.join('results', 'strategy', 'report.md')

TEST_START_DATE_STR = '2017-01-01'

# --- Вспомогательные функции ---



def load_data():
    """Загружает очищенные данные, сигнал и бенчмарк."""
    print("-> Загрузка данных для бэктестинга...")
    
    # 1. Загрузка данных с будущими возвратами (Target и future_return)
    df_raw = pd.read_csv(CLEANED_DATA_PATH, index_col=['date', 'Ticker'], parse_dates=['date'])
    df_returns = df_raw[['future_return', 'Target']].copy()
    
    # 2. Загрузка ML-сигнала
    df_signal = pd.read_csv(ML_SIGNAL_PATH, index_col=['date', 'Ticker'], parse_dates=['date'])
    
    # Объединение
    df = df_returns.join(df_signal, how='inner')
    df = df.dropna(subset=['ML_Signal'])
    
    # 3. Загрузка бенчмарка SP500 (HistoricalPrices.csv)
    df_spx = pd.read_csv(INDEX_DATA_PATH, index_col='Date', parse_dates=['Date'])
    df_spx = df_spx.sort_index()


    df_spx.columns = df_spx.columns.str.strip() 

    PRICE_COLUMN = 'Close'
    
    if PRICE_COLUMN not in df_spx.columns:
        raise KeyError(f"Критическая ОШИБКА: Колонка цены '{PRICE_COLUMN}' не найдена в HistoricalPrices.csv после очистки пробелов. Доступные колонки: {df_spx.columns.tolist()}")

    # Переименовываем колонку Close в SPX_Close
    df_spx = df_spx.rename(columns={PRICE_COLUMN: 'SPX_Close'})
    
    # Расчет дневного возврата SP500 (бенчмарк)
    df_spx['SPX_Return'] = df_spx['SPX_Close'].pct_change()
    
    return df, df_spx


def implement_strategy(df, long_threshold=0.6, short_threshold=0.4):
    """
    Реализует Тернарную (Long/Short/Neutral) стратегию на основе ML-сигнала.
    
    Веса: Ternary signal: -1 (Short) / 0 (Neutral) / 1 (Long)
    Нормализация: Инвестируем 1$ в день, распределяя его по выбранным позициям.
    """
    print(f"-> Реализация стратегии (Long > {long_threshold}, Short < {short_threshold})...")
    
    # Шаг 1: Генерация весов (W_i,t)
    
    # 1. Long позиция: сигнал > long_threshold
    df['Weight_i'] = np.where(df['ML_Signal'] > long_threshold, 1, 0)
    
    # 2. Short позиция: сигнал < short_threshold
    df['Weight_i'] = np.where(df['ML_Signal'] < short_threshold, -1, df['Weight_i'])
    
    # Шаг 2: Нормализация (Инвестировать 1$ в день)
    
    # Группируем по дате, чтобы найти общее количество активных позиций (Long + Short)
    def normalize_weights(group):
        # Сумма абсолютных весов за день (сколько всего позиций, включая короткие)
        total_abs_weight = group['Weight_i'].abs().sum()
        
        if total_abs_weight > 0:
            # Нормализация: распределяем $1 равномерно между активными позициями
            group['Strategy_Weight'] = group['Weight_i'] / total_abs_weight
        else:
            # Если нет позиций, вес = 0
            group['Strategy_Weight'] = 0.0
        return group
    
    # Применяем нормализацию по дате
    df_strategy = df.groupby(level='date', group_keys=False).apply(normalize_weights)

    # Strategy_Weight (W_i,t) - это количество, которое мы инвестируем в актив i в день t.
    
    return df_strategy

def calculate_pnl(df_strategy, df_spx):
    """
    Рассчитывает PnL стратегии и бенчмарка.
    PnL(t) = Sum[ W_i,t * return(i, t+1, t+2) ]
    """
    print("-> Расчет PnL...")
    

    df_strategy['PnL_daily'] = df_strategy['Strategy_Weight'] * df_strategy['future_return']
    
    # Группировка дневного PnL по дате
    df_pnl = df_strategy.groupby(level='date')['PnL_daily'].sum().to_frame('Strategy_PnL_Daily')
    
    # Кумулятивный PnL
    df_pnl['Strategy_PnL_Cumulative'] = df_pnl['Strategy_PnL_Daily'].cumsum()
    
    # 2. PnL SP500 (Бенчмарк)
    # PnL SP500 - это просто кумулятивный возврат SPX
    
    # Нам нужно переиндексировать SPX_Return к датам df_pnl
    df_benchmark = df_spx[['SPX_Return']].reindex(df_pnl.index)
    df_benchmark = df_benchmark.fillna(0) # Заполняем NaN, если дни не совпадают
    
    df_pnl['SPX_PnL_Daily'] = df_benchmark['SPX_Return']
    df_pnl['SPX_PnL_Cumulative'] = df_pnl['SPX_PnL_Daily'].cumsum()
    
    return df_pnl

def calculate_metrics(pnl_series):
    """Рассчитывает Max Drawdown (MDD) и PnL."""
    
    # 1. PnL (Итоговый Кумулятивный)
    final_pnl = pnl_series.iloc[-1]
    
    # 2. Max Drawdown
    cumulative_pnl = pnl_series
    peak = cumulative_pnl.expanding(min_periods=1).max()
    drawdown = (cumulative_pnl - peak)
    max_drawdown = drawdown.min()
    
    return final_pnl, max_drawdown

def plot_pnl(df_pnl):
    """
    Построение графика PnL стратегии и SP500.
    """
    print("-> Построение графика PnL...")
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Убеждаемся, что обе серии начинаются с 0 для корректного сравнения PnL
    df_pnl_shifted = df_pnl.copy()
    df_pnl_shifted['Strategy_PnL_Cumulative'] -= df_pnl_shifted['Strategy_PnL_Cumulative'].iloc[0]
    df_pnl_shifted['SPX_PnL_Cumulative'] -= df_pnl_shifted['SPX_PnL_Cumulative'].iloc[0]


    # График Стратегии
    ax1.plot(df_pnl_shifted.index, df_pnl_shifted['Strategy_PnL_Cumulative'], label='Strategy PnL', color='darkgreen', linewidth=2)
    
    # График SP500 (на той же оси Y для одинакового масштаба)
    ax1.plot(df_pnl_shifted.index, df_pnl_shifted['SPX_PnL_Cumulative'], label='S&P 500 PnL (Benchmark)', color='darkred', linestyle='--', linewidth=1.5)

    ax1.set_xlabel('Date')
    ax1.set_ylabel('Cumulative PnL', fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.set_title('Strategy vs S&P 500 Cumulative PnL')
    
    # Вертикальная линия разделения Train/Test
    test_start_date = pd.to_datetime(TEST_START_DATE_STR)
    
    if test_start_date in df_pnl.index:
        ax1.axvline(x=test_start_date, color='grey', linestyle='-', linewidth=2, label='Test Set Start')
        ax1.text(test_start_date, ax1.get_ylim()[1] * 0.9, ' Test Set Start ', rotation=90, verticalalignment='top', horizontalalignment='right')
        ax1.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(STRATEGY_PLOT_PATH)
    plt.close()
    print(f"-> График PnL сохранен в {STRATEGY_PLOT_PATH}")
     #  тэг визуализации


def create_report(df_metrics_train, df_metrics_test, min_train_days):
    """Генерирует финальный отчет report.md."""
    
    report_content = []
    report_content.append("#  S&P 500 Trading Strategy Report\n")
    
    report_content.append("##  1. Features Used\n")
    report_content.append("* **Technical Indicators (сгруппированы по Ticker):** Bollinger Bands, RSI, MACD (Signal, Diff).\n")
    report_content.append("* **Simple Features:** Лагированные дневные возвраты (1d, 5d), 20-дневная волатильность.\n")
    report_content.append("* **Target (Целевая переменная):** `sign(return(D+1, D+2))`. Обеспечено **отсутствие утечки данных (No Leakage)** путем сдвига.\n")

    report_content.append("\n##  2. Machine Learning Pipeline\n")
    report_content.append("* **Imputer:** `SimpleImputer(strategy='median')`.\n")
    report_content.append("* **Scaler:** `StandardScaler()`.\n")
    report_content.append("* **Model:** `LGBMClassifier` (Light Gradient Boosting Machine).\n")
    report_content.append("* **Hyperparameters (Best):** Получены с помощью Grid Search:\n")
    
    try:
        with open(os.path.join('results', 'selected-model', 'selected_model.txt'), 'r') as f:
            best_params = f.read().strip()
    except FileNotFoundError:
        best_params = "Параметры не найдены."
        
    report_content.append(f"    ```json\n    {best_params}\n    ```\n")

    report_content.append("\n##  3. Cross-Validation Used\n")
    report_content.append("* **Метод:** Time Series Split (Expanding Window).\n")
    report_content.append("* **Количество фолдов:** 10 (N=10).\n")
    report_content.append(f"* **Длина истории (мин.):** Тренировочный набор каждого фолда имеет минимум 2 года истории ({min_train_days} дней), что критично для временных рядов.\n")
    report_content.append("* **Визуализация:** Графики `TimeSeriesSplit.png` и `BlockingTimeSeriesSplit.png` сохранены в `results/cross-validation`.\n")
    

    report_content.append("\n##  4. Strategy Chosen: Ternary Long/Short\n")
    report_content.append("* **Описание:** Стратегия основана на Тернарном сигнале (Long/Short/Neutral):\n")
    report_content.append("    * **Long:** Если $P(рост) > 0.6$ (Threshold).\n")
    report_content.append("    * **Short:** Если $P(рост) < 0.4$ (Threshold).\n")
    report_content.append("    * **Neutral:** Иначе.\n")
    report_content.append("* **Инвестирование:** Общая сумма инвестиций нормируется, чтобы каждый день инвестировать $1$, распределяя его равномерно между всеми активными Long и Short позициями (абсолютный вес).\n")
    
    # Таблица метрик
    report_content.append("\n### Strategy Metrics (PnL & Max Drawdown)\n")
    
    train_pnl, train_mdd = df_metrics_train
    test_pnl, test_mdd = df_metrics_test
    
    metrics_table = f"""
| Метрика | Train Set (до {TEST_START_DATE_STR}) | Test Set (с {TEST_START_DATE_STR}) |
| :--- | :---: | :---: |
| **Кумулятивный PnL ($)** | ${train_pnl:.4f}$ | ${test_pnl:.4f}$ |
| **Максимальная просадка (MDD)** | {train_mdd:.4f} | {test_mdd:.4f} |
"""
    report_content.append(metrics_table)
    report_content.append(f"\n* **PnL Plot:** См. `strategy.png` для сравнения с бенчмарком S&P 500.")

    with open(REPORT_PATH, 'w') as f:
        f.writelines(report_content)
    print(f"-> Отчет сохранен в {REPORT_PATH}")


# scripts/strategy.py: ИСПРАВЛЕННАЯ СЕКЦИЯ __main__ (добавлено определение MIN_TRAIN_DAYS)

if __name__ == '__main__':
    # Определяем MIN_TRAIN_DAYS здесь, чтобы она была доступна
    MIN_TRAIN_DAYS = int(365.25 * 2) 

    # 1. Загрузка данных
    df, df_spx = load_data()
    
    # 2. Реализация стратегии
    df_strategy = implement_strategy(df)
    
    # 3. Расчет PnL
    df_pnl = calculate_pnl(df_strategy, df_spx)
    
    # 4. Расчет финансовых метрик (разделение на Train/Test)
    test_start_date = pd.to_datetime(TEST_START_DATE_STR)
    
    # Метрики на Train Set
    pnl_train_series = df_pnl.loc[df_pnl.index < test_start_date, 'Strategy_PnL_Cumulative']
    pnl_train, mdd_train = calculate_metrics(pnl_train_series)
    
    # Метрики на Test Set
    pnl_test_series = df_pnl.loc[df_pnl.index >= test_start_date, 'Strategy_PnL_Cumulative']
    if not pnl_test_series.empty and not pnl_train_series.empty:
        pnl_test_series_normalized = pnl_test_series - pnl_train_series.iloc[-1]
    else:
        pnl_test_series_normalized = pnl_test_series
        
    pnl_test, mdd_test = calculate_metrics(pnl_test_series_normalized)
    
    print("\n--- Финансовые Метрики ---")
    print(f"Train PnL: {pnl_train:.4f}, Train MDD: {mdd_train:.4f}")
    print(f"Test PnL: {pnl_test:.4f}, Test MDD: {mdd_test:.4f}")
    
    # 5. Сохранение результатов (PnL за каждый день)
    df_pnl.to_csv(RESULTS_CSV_PATH)
    
    # 6. Построение PnL графика
    plot_pnl(df_pnl)
    
    # 7. Генерация отчета
    # Передаем MIN_TRAIN_DAYS
    create_report((pnl_train, mdd_train), (pnl_test, mdd_test), MIN_TRAIN_DAYS)
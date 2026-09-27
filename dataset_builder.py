"""
Превращает свечи в обучающий датасет.

Идея:
1. Считаем коэффициенты роста цены и объёма (относительно предыдущей минуты).
2. Находим «скачки» — минуты, где и цена, и объём изменились достаточно сильно,
   и за последовательность накопилось изменение >= SPIKE_SEQ_PCT.
3. Для каждого скачка:
   - вход: окно признаков за LOOKBACK_MINUTES минут ДО скачка;
   - выход: величина коррекции (в %) через 1, 2, 3, 4, 5 минут ПОСЛЕ скачка.
"""

import numpy as np
import pandas as pd
from config import (
    SPIKE_MIN_PCT, SPIKE_VOLUME_MIN_PCT,
    SPIKE_SEQ_MINUTES, SPIKE_SEQ_PCT,
    LOOKBACK_MINUTES, HORIZONS,
)
from utils import log


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет к свечам признаки:
    - ret_price: относительное изменение цены за минуту
    - ret_volume: относительное изменение объёма за минуту
    - hour, weekday: временные признаки
    """
    out = df.copy()

    # Относительное изменение цены: (close - prev_close) / prev_close
    out["ret_price"] = out["close"].pct_change()

    # Относительное изменение объёма
    out["ret_volume"] = out["volume"].pct_change()

    # Временные признаки
    out["hour"] = out["open_time"].dt.hour
    out["weekday"] = out["open_time"].dt.weekday

    # Убираем бесконечности и NaN
    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    out.dropna(inplace=True)
    out.reset_index(drop=True, inplace=True)
    return out


def find_spikes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Находит строки-скачки.

    Условие скачка на минуте i:
    1. |ret_price[i]| >= SPIKE_MIN_PCT / 100
    2. |ret_volume[i]| >= SPIKE_VOLUME_MIN_PCT / 100
    3. |сумма ret_price за последние SPIKE_SEQ_MINUTES| >= SPIKE_SEQ_PCT / 100
    """
    spikes = []
    seq = SPIKE_SEQ_MINUTES
    min_price = SPIKE_MIN_PCT / 100.0
    min_vol = SPIKE_VOLUME_MIN_PCT / 100.0
    min_seq = SPIKE_SEQ_PCT / 100.0

    for i in range(seq, len(df)):
        if abs(df["ret_price"].iloc[i]) < min_price:
            continue
        if abs(df["ret_volume"].iloc[i]) < min_vol:
            continue
        # Суммарное изменение цены за последовательность
        seq_sum = df["ret_price"].iloc[i - seq + 1 : i + 1].sum()
        if abs(seq_sum) < min_seq:
            continue
        spikes.append(i)

    log(f"Найдено скачков: {len(spikes)}")
    return df.iloc[spikes].copy()


def build_dataset(df: pd.DataFrame, lookback: int = None, horizons: list = None):
    """
    Возвращает X (признаки) и Y (целевые переменные).

    X: для каждого скачка — матрица (lookback, n_features).
    Y: для каждого скачка — вектор длиной len(horizons),
       где каждый элемент — коррекция в % через h минут.
    """
    lookback = lookback or LOOKBACK_MINUTES
    horizons = horizons or HORIZONS

    feature_cols = ["ret_price", "ret_volume", "hour", "weekday"]
    spikes = find_spikes(df)

    X_list, Y_list = [], []

    for idx in spikes.index:
        # --- Вход: окно до скачка ---
        start = idx - lookback
        if start < 0:
            continue
        window = df.iloc[start:idx][feature_cols].values
        if window.shape[0] != lookback:
            continue

        # --- Выход: коррекция через h минут ---
        # «Коррекция» = (close[t+h] - close[t]) / close[t] * 100
        # где t — минута скачка.
        close_t = df["close"].iloc[idx]
        y = []
        valid = True
        for h in horizons:
            t_h = idx + h
            if t_h >= len(df):
                valid = False
                break
            close_h = df["close"].iloc[t_h]
            correction = (close_h - close_t) / close_t * 100.0
            y.append(correction)
        if not valid:
            continue

        X_list.append(window)
        Y_list.append(y)

    if not X_list:
        raise RuntimeError("Не удалось сформировать ни одного примера")

    X = np.array(X_list, dtype=np.float32)   # (N, lookback, features)
    Y = np.array(Y_list, dtype=np.float32)   # (N, len(horizons))

    # Нормализация X: вычитаем среднее, делим на std по каждому признаку
    mean = X.mean(axis=(0, 1), keepdims=True)
    std = X.std(axis=(0, 1), keepdims=True) + 1e-8
    X = (X - mean) / std

    # Сохраняем статистики, чтобы применять ту же нормализацию при инференсе
    np.savez(
        "normalization.npz",
        mean=mean.squeeze(),
        std=std.squeeze(),
    )

    log(f"Датасет: X={X.shape}, Y={Y.shape}")
    return X, Y
"""
Формирование датасета.

Отличия от предыдущей версии:
- Добавлены технические индикаторы: RSI, ATR, скользящая волатильность,
  отношение объёма к среднему, размер «тела» свечи.
- Временные признаки сделаны циклическими (sin/cos), чтобы модель
  понимала, что 23:00 и 00:00 — соседние часы.
"""

import numpy as np
import pandas as pd
from config import (
    SPIKE_MIN_PCT, SPIKE_VOLUME_MIN_PCT,
    SPIKE_SEQ_MINUTES, SPIKE_SEQ_PCT,
    LOOKBACK_MINUTES, HORIZONS,
)
from utils import log


# Список признаков, которые пойдут в нейросеть.
# Порядок важен только для отладки — сеть сама разберётся.
FEATURE_COLS = [
    "ret_price",     # лог-доходность цены за минуту
    "ret_volume",    # лог-доходность объёма за минуту
    "vol_5",         # волатильность (std) за 5 минут
    "vol_15",        # волатильность за 15 минут
    "vol_60",        # волатильность за 60 минут
    "rsi_norm",      # RSI(14), нормализованный к [-1, 1]
    "atr_norm",      # ATR(14) / close — относительная волатильность
    "vol_ratio_30",  # текущий объём / средний объём за 30 минут
    "candle_body",   # (close - open) / (high - low) — «тело» свечи
    "hour_sin", "hour_cos",
    "wd_sin", "wd_cos",
]


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    RSI — Relative Strength Index.
    Показывает, «перекуплена» цена (близко к 100) или «перепродана» (близко к 0).
    Классический RSI: средний рост / среднее падение за period минут.
    """
    delta = close.diff()
    gain = delta.clip(lower=0)         # только положительные изменения
    loss = (-delta).clip(lower=0)      # только отрицательные изменения
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / (avg_loss + 1e-8)
    return 100.0 - 100.0 / (1.0 + rs)


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    ATR — Average True Range. Средний «размах» свечи за period минут.
    True Range = max(high-low, |high-prev_close|, |low-prev_close|).
    """
    prev_close = df["close"].shift()
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет к свечам расширенный набор признаков.

    ВАЖНО: колонка original_index хранит позицию строки в исходном df.
    Это нужно, чтобы график мог корректно отметить скачки — иначе после
    dropna/reset_index индексы в df_feat уже не соответствуют df.
    """
    out = df.copy()
    out["original_index"] = np.arange(len(out), dtype=np.int64)

    # --- Базовые доходности ---
    out["ret_price"] = out["close"].pct_change()
    out["ret_volume"] = out["volume"].pct_change()

    # --- Волатильность на разных горизонтах ---
    out["vol_5"] = out["ret_price"].rolling(5).std()
    out["vol_15"] = out["ret_price"].rolling(15).std()
    out["vol_60"] = out["ret_price"].rolling(60).std()

    # --- RSI(14), нормализуем к [-1, 1] ---
    out["rsi_norm"] = (_rsi(out["close"], 14) - 50.0) / 50.0

    # --- ATR(14), нормализуем на цену ---
    out["atr_norm"] = _atr(out, 14) / out["close"]

    # --- Отношение объёма к среднему за 30 минут ---
    avg_vol_30 = out["volume"].rolling(30).mean()
    out["vol_ratio_30"] = (out["volume"] / (avg_vol_30 + 1e-8)).clip(upper=10.0)

    # --- «Тело» свечи ---
    rng = (out["high"] - out["low"]).replace(0, np.nan)
    out["candle_body"] = ((out["close"] - out["open"]) / rng).fillna(0.0)

    # --- Циклическое время ---
    hour = out["open_time"].dt.hour
    wd = out["open_time"].dt.weekday
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    out["wd_sin"] = np.sin(2 * np.pi * wd / 7.0)
    out["wd_cos"] = np.cos(2 * np.pi * wd / 7.0)

    # --- Чистим NaN ---
    # original_index выживает, потому что это просто целые числа.
    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    out.dropna(inplace=True)
    out.reset_index(drop=True, inplace=True)
    return out

def find_spikes(df: pd.DataFrame,
                spike_min_pct=None, spike_vol_min_pct=None,
                spike_seq_minutes=None, spike_seq_pct=None) -> pd.DataFrame:
    """
    Ищет скачки. Параметры можно передавать явно (для авто-режима),
    иначе берутся из config.
    """
    from config import (
        SPIKE_MIN_PCT as CFG_MIN,
        SPIKE_VOLUME_MIN_PCT as CFG_VOL,
        SPIKE_SEQ_MINUTES as CFG_SEQ,
        SPIKE_SEQ_PCT as CFG_SEQ_PCT,
    )
    spike_min_pct = spike_min_pct if spike_min_pct is not None else CFG_MIN
    spike_vol_min_pct = spike_vol_min_pct if spike_vol_min_pct is not None else CFG_VOL
    spike_seq_minutes = spike_seq_minutes if spike_seq_minutes is not None else CFG_SEQ
    spike_seq_pct = spike_seq_pct if spike_seq_pct is not None else CFG_SEQ_PCT

    min_price = spike_min_pct / 100.0
    min_vol = spike_vol_min_pct / 100.0
    min_seq = spike_seq_pct / 100.0

    spikes = []
    for i in range(spike_seq_minutes, len(df)):
        if abs(df["ret_price"].iloc[i]) < min_price:
            continue
        if abs(df["ret_volume"].iloc[i]) < min_vol:
            continue
        seq_sum = df["ret_price"].iloc[i - spike_seq_minutes + 1 : i + 1].sum()
        if abs(seq_sum) < min_seq:
            continue
        spikes.append(i)

    log(f"Найдено скачков: {len(spikes)}")
    return df.iloc[spikes].copy()


def build_dataset(df: pd.DataFrame, lookback: int = None, horizons: list = None):
    """
    Возвращает:
        X          — (N, lookback, n_features)
        Y          — (N, len(horizons))
        orig_idx   — (N,)  исходные позиции скачков в СЫРОМ df.
                     Нужны для визуализации: plot использует ту же ось X,
                     что и сырой df, поэтому маркеры встают на свои места.
    """
    lookback = lookback or LOOKBACK_MINUTES
    horizons = horizons or HORIZONS

    spikes = find_spikes(df)
    X_list, Y_list, idx_list = [], [], []

    for idx in spikes.index:
        start = idx - lookback
        if start < 0:
            continue
        window = df.iloc[start:idx][FEATURE_COLS].values
        if window.shape[0] != lookback:
            continue

        close_t = df["close"].iloc[idx]
        y = []
        valid = True
        for h in horizons:
            t_h = idx + h
            if t_h >= len(df):
                valid = False
                break
            close_h = df["close"].iloc[t_h]
            y.append((close_h - close_t) / close_t * 100.0)
        if not valid:
            continue

        X_list.append(window)
        Y_list.append(y)
        # Запоминаем ИСХОДНУЮ позицию этого скачка в сыром df
        idx_list.append(int(df["original_index"].iloc[idx]))

    if not X_list:
        raise RuntimeError("Не удалось сформировать ни одного примера")

    X = np.array(X_list, dtype=np.float32)
    Y = np.clip(np.array(Y_list, dtype=np.float32), -2.0, 2.0)
    orig_idx = np.array(idx_list, dtype=np.int64)

    mean = X.mean(axis=(0, 1), keepdims=True)
    std = X.std(axis=(0, 1), keepdims=True) + 1e-8
    X = (X - mean) / std

    np.savez("normalization.npz", mean=mean.squeeze(), std=std.squeeze())
    log(f"Датасет: X={X.shape}, Y={Y.shape}, скачков={len(orig_idx)}")
    return X, Y, orig_idx
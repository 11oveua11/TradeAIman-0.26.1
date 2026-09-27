"""
Загрузка 1-минутных свечей BTCUSDT за последние N лет.

Алгоритм:
1. Пробуем MEXC (спотовый API).
2. Если MEXC не отдаёт данные — переключаемся на Binance (фьючерсы).
3. Идём «назад» по времени порциями по LIMIT свечей за запрос.
4. Каждая порция сохраняется в CSV, чтобы не качать повторно.
"""

import os
import time
import requests
import pandas as pd
from config import (
    SYMBOL, INTERVAL, YEARS_BACK,
    MEXC_BASE_URL, MEXC_KLINES_PATH, MEXC_LIMIT,
    BINANCE_BASE_URL, BINANCE_KLINES_PATH, BINANCE_LIMIT,
    DATA_DIR,
)
from utils import interval_to_ms, sleep_rate_limit, log

# Колонки, которые возвращают обе биржи (порядок совпадает)
KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades",
    "taker_buy_base", "taker_buy_quote", "ignore"
]

def _parse_klines(raw, limit):
    """Преобразует «сырой» ответ биржи в DataFrame."""
    if not raw:
        return pd.DataFrame()
    df = pd.DataFrame(raw, columns=KLINE_COLUMNS[:len(raw[0])])
    # Приводим типы
    for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
    return df

def fetch_mexc(start_ms, end_ms, limit=MEXC_LIMIT):
    """
    Один запрос к MEXC.
    startTime и endTime должны передаваться вместе (требование API).
    """
    url = MEXC_BASE_URL + MEXC_KLINES_PATH
    params = {
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": limit,
    }
    try:
        r = requests.get(url, params=params, timeout=15)
        if r.status_code == 200:
            data = r.json()
            if isinstance(data, list):
                return _parse_klines(data, limit)
    except Exception as e:
        log(f"MEXC ошибка: {e}")
    return pd.DataFrame()

def fetch_binance(start_ms, end_ms, limit=BINANCE_LIMIT):
    """
    Один запрос к Binance Futures.
    Если startTime и endTime не переданы — вернутся последние свечи.
    """
    url = BINANCE_BASE_URL + BINANCE_KLINES_PATH
    params = {
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": limit,
    }
    try:
        r = requests.get(url, params=params, timeout=15)
        if r.status_code == 200:
            data = r.json()
            if isinstance(data, list):
                return _parse_klines(data, limit)
        elif r.status_code == 429:
            log("Binance: превышен лимит запросов, пауза 5 сек")
            time.sleep(5)
    except Exception as e:
        log(f"Binance ошибка: {e}")
    return pd.DataFrame()

def load_klines(progress_callback=None):
    """
    Главная функция загрузки.

    Идём от «сейчас» назад на YEARS_BACK лет.
    На каждом шаге берём порцию LIMIT свечей.
    Сначала пробуем MEXC, если пусто — Binance.
    """
    path = os.path.join(DATA_DIR, f"{SYMBOL}_{INTERVAL}_{YEARS_BACK}y.csv")
    if os.path.exists(path):
        log(f"Загружаю из кэша: {path}")
        df = pd.read_csv(path, parse_dates=["open_time", "close_time"])
        if progress_callback:
            progress_callback(1.0)
        return df

    step_ms = interval_to_ms(INTERVAL)
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - YEARS_BACK * 365 * 24 * 60 * 60 * 1000

    all_parts = []
    cursor = end_ms
    total = (end_ms - start_ms) // (step_ms * MEXC_LIMIT)

    while cursor > start_ms:
        chunk_start = cursor - MEXC_LIMIT * step_ms
        if chunk_start < start_ms:
            chunk_start = start_ms

        # --- MEXC ---
        df = fetch_mexc(chunk_start, cursor)
        # --- Binance (если MEXC пусто) ---
        if df.empty:
            log("MEXC не отдал данные, пробую Binance...")
            df = fetch_binance(chunk_start, cursor, BINANCE_LIMIT)

        if not df.empty:
            all_parts.append(df)
            log(f"Загружено {len(df)} свечей до {df['open_time'].min()}")

        if progress_callback:
            done = (end_ms - chunk_start) / (end_ms - start_ms)
            progress_callback(min(done, 1.0))

        cursor = chunk_start
        sleep_rate_limit()

    if not all_parts:
        raise RuntimeError("Не удалось загрузить свечи ни с MEXC, ни с Binance")

    full = pd.concat(all_parts, ignore_index=True)
    full = full.drop_duplicates(subset=["open_time"]).sort_values("open_time")
    full.reset_index(drop=True, inplace=True)

    full.to_csv(path, index=False)
    log(f"Сохранено {len(full)} свечей в {path}")
    return full
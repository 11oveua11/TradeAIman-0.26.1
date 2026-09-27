"""
Мелкие утилиты: логирование, таймстемпы, нормализация.
"""

import time
import datetime as dt

def now_str():
    """Возвращает текущее время в виде строки для логов."""
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def log(msg, widget=None):
    """
    Печатает сообщение в консоль и, если передан widget,
    добавляет его в текстовое поле GUI.
    """
    line = f"[{now_str()}] {msg}"
    print(line)
    if widget is not None:
        widget.insert("end", line + "\n")
        widget.see("end")

def interval_to_ms(interval: str) -> int:
    """Переводит строку '1m', '5m', '1h' в миллисекунды."""
    unit = interval[-1]
    value = int(interval[:-1])
    mult = {"m": 60_000, "h": 3_600_000, "d": 86_400_000}[unit]
    return value * mult

def sleep_rate_limit():
    """Пауза между запросами к API, чтобы не получить бан."""
    time.sleep(0.25)
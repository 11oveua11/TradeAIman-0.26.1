
"""
Все настройки приложения в одном месте.
Меняя значения здесь, вы меняете поведение всей программы.
"""

import os

# --- Пути ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# --- Символ и биржа ---
SYMBOL = "BTCUSDT"
INTERVAL = "1m"                    # 1-минутные свечи
YEARS_BACK = 1                     # глубина истории

# MEXC: спотовый API
MEXC_BASE_URL = "https://api.mexc.com"
MEXC_KLINES_PATH = "/api/v3/klines"
MEXC_LIMIT = 500                   # максимум свечей за 1 запрос у MEXC

# Binance: фьючерсный API (больше лимит)
BINANCE_BASE_URL = "https://fapi.binance.com"
BINANCE_KLINES_PATH = "/fapi/v1/klines"
BINANCE_LIMIT = 1500               # максимум свечей за 1 запрос у Binance

# --- Определение скачка (spike) ---
# Скачок = за одну минуту цена и объём изменились минимум на SPIKE_MIN_PCT,
# И за последовательность (SPIKE_SEQ_MINUTES) суммарное изменение >= SPIKE_SEQ_PCT.
SPIKE_MIN_PCT = 0.15               # % изменения цены за 1 минуту
SPIKE_VOLUME_MIN_PCT = 0.15        # % изменения объёма за 1 минуту
SPIKE_SEQ_MINUTES = 5              # длина «последовательности»
SPIKE_SEQ_PCT = 0.6                # % суммарного изменения за последовательность

# --- Датасет ---
# Сколько минут «до» скачка подаём на вход сети (окно признаков)
LOOKBACK_MINUTES = 10
# Через сколько минут после скачка измеряем коррекцию (целевые переменные)
HORIZONS = [1, 2, 3, 4, 5]

# --- Обучение MLP ---
HIDDEN_LAYERS = [64, 32]          # размеры скрытых слоёв
DROPOUT = 0.2
EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
TRAIN_SPLIT = 0.8                  # 80% на обучение, 20% на валидацию
RANDOM_SEED = 42

# --- Автоматический режим ---
AUTO_PARAM_GRID = {
    "SPIKE_MIN_PCT": [0.10, 0.15, 0.20],
    "SPIKE_SEQ_PCT": [0.4, 0.6, 0.8],
    "LOOKBACK_MINUTES": [5, 10, 15],
    "HIDDEN_LAYERS": [[64, 32], [128, 64], [256, 128]],
    "LEARNING_RATE": [1e-3, 5e-4, 1e-4],
}

# --- Early stopping ---
# Сколько эпох ждать улучшения val_loss, прежде чем остановить обучение.
EARLY_STOP_PATIENCE = 15
# Минимальное улучшение, которое считается «прогрессом» (иначе — шум).
EARLY_STOP_MIN_DELTA = 1e-5
# L2-регуляризация: штраф за большие веса. Помогает от переобучения.
WEIGHT_DECAY = 1e-4
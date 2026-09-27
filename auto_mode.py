"""
Автоматический режим: перебирает параметры, обучает модели,
ищет наилучший результат по метрике (например, h1_MAE).
"""

import itertools
import numpy as np
from config import AUTO_PARAM_GRID, HORIZONS
import dataset_builder
import trainer
from utils import log


def _parse_hidden(s):
    """'128,64' -> [128, 64]"""
    return [int(x) for x in s.split(",") if x.strip()]


def run_auto(app):
    """
    app — экземпляр App, из которого берём данные и чекбоксы.
    """
    log("=== Авто-режим: перебор параметров ===", app.report)

    if app.df is None:
        log("Нет свечей. Загрузите данные.", app.report)
        return

    # Какие параметры перебираем
    selected = {k: v.get() for k, v in app.auto_vars.items()}
    grid_keys = [k for k, on in selected.items() if on]
    if not grid_keys:
        log("Не выбрано ни одного параметра для перебора.", app.report)
        return

    grid_values = [AUTO_PARAM_GRID[k] for k in grid_keys]
    combos = list(itertools.product(*grid_values))
    log(f"Комбинаций: {len(combos)}", app.report)

    best_score = float("inf")
    best_params = None
    best_model = None
    best_metrics = None

    for i, combo in enumerate(combos, 1):
        params = dict(zip(grid_keys, combo))
        log(f"\n--- Комбинация {i}/{len(combos)}: {params} ---", app.report)

        # Подменяем параметры в config (или передаём явно)
        spike_min = params.get("SPIKE_MIN_PCT", None)
        seq_pct = params.get("SPIKE_SEQ_PCT", None)
        lookback = params.get("LOOKBACK_MINUTES", None)
        hidden = params.get("HIDDEN_LAYERS", None)
        lr = params.get("LEARNING_RATE", None)

        # Перестраиваем датасет с новыми параметрами
        try:
            df_feat = dataset_builder.build_features(app.df)
            # Временно подменяем глобальные настройки (простой способ)
            import config
            old = {}
            if spike_min is not None:
                old["SPIKE_MIN_PCT"] = config.SPIKE_MIN_PCT
                config.SPIKE_MIN_PCT = spike_min
            if seq_pct is not None:
                old["SPIKE_SEQ_PCT"] = config.SPIKE_SEQ_PCT
                config.SPIKE_SEQ_PCT = seq_pct
            if lookback is not None:
                old["LOOKBACK_MINUTES"] = config.LOOKBACK_MINUTES
                config.LOOKBACK_MINUTES = lookback

            X, Y = dataset_builder.build_dataset(df_feat)

            # Восстанавливаем
            for k, v in old.items():
                setattr(config, k, v)
        except Exception as e:
            log(f"Ошибка датасета: {e}", app.report)
            continue

        # Обучаем
        try:
            model, metrics, history = trainer.train_model(
                X, Y, log_widget=app.report,
                hidden_layers=hidden, lr=lr,
                epochs=min(30, config.EPOCHS),  # для скорости
            )
        except Exception as e:
            log(f"Ошибка обучения: {e}", app.report)
            continue

        # Оценка: средний MAE по всем горизонтам
        score = np.mean([metrics[f"h{h}_MAE"] for h in HORIZONS])
        log(f"Средний MAE: {score:.6f}", app.report)

        if score < best_score:
            best_score = score
            best_params = params.copy()
            best_model = model
            best_metrics = metrics

    log(f"\n=== Лучший результат ===", app.report)
    log(f"Параметры: {best_params}", app.report)
    log(f"MAE: {best_score:.6f}", app.report)
    log(f"Метрики: {best_metrics}", app.report)

    # Сохраняем лучшую модель в app
    app.model = best_model
    app.metrics = best_metrics
    app.history = None
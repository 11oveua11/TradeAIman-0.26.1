"""
GUI на CustomTkinter.

Окно разделено по вертикали на 3 блока:
1. Верх — кнопки и поля ввода.
2. Середина — график (matplotlib внутри CustomTkinter).
3. Низ — текстовый отчёт.

Для встраивания matplotlib используется FigureCanvasTkAgg.
"""

import threading
import customtkinter as ctk
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import numpy as np

from config import (
    SPIKE_MIN_PCT, SPIKE_VOLUME_MIN_PCT, SPIKE_SEQ_MINUTES, SPIKE_SEQ_PCT,
    LOOKBACK_MINUTES, HORIZONS, EPOCHS, LEARNING_RATE, HIDDEN_LAYERS,
)
import data_loader
import dataset_builder
import trainer
from utils import log


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("BTC Correction Predictor")
        self.geometry("1200x900")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # Данные, которые будут заполняться по ходу работы
        self.df = None          # свечи
        self.dataset = None     # (X, Y)
        self.model = None
        self.metrics = None
        self.history = None
        self.predictions = None

        self._build_top()
        self._build_plot()
        self._build_report()

    # ---------- Блок 1: кнопки и поля ----------
    def _build_top(self):
        frame = ctk.CTkFrame(self, height=200)
        frame.pack(fill="x", padx=10, pady=5)
        frame.pack_propagate(False)

        # Левая колонка — кнопки
        btn_frame = ctk.CTkFrame(frame)
        btn_frame.pack(side="left", fill="y", padx=10, pady=10)

        ctk.CTkButton(btn_frame, text="1. Загрузить свечи",
                      command=self.on_load).pack(fill="x", pady=2)
        ctk.CTkButton(btn_frame, text="2. Построить датасет",
                      command=self.on_build).pack(fill="x", pady=2)
        ctk.CTkButton(btn_frame, text="3. Обучить модель",
                      command=self.on_train).pack(fill="x", pady=2)
        ctk.CTkButton(btn_frame, text="4. Показать график",
                      command=self.on_plot).pack(fill="x", pady=2)
        ctk.CTkButton(btn_frame, text="Авто-режим",
                      command=self.on_auto).pack(fill="x", pady=2)

        # Правая колонка — параметры
        param_frame = ctk.CTkFrame(frame)
        param_frame.pack(side="left", fill="both", expand=True, padx=10, pady=10)

        self.entries = {}
        params = [
            ("Порог скачка, %", SPIKE_MIN_PCT),
            ("Порог объёма, %", SPIKE_VOLUME_MIN_PCT),
            ("Длина последовательности, мин", SPIKE_SEQ_MINUTES),
            ("Сумма за последовательность, %", SPIKE_SEQ_PCT),
            ("Окно до скачка, мин", LOOKBACK_MINUTES),
            ("Эпох обучения", EPOCHS),
            ("Learning rate", LEARNING_RATE),
            ("Скрытые слои (через запятую)", ",".join(map(str, HIDDEN_LAYERS))),
        ]
        for i, (label, default) in enumerate(params):
            row = ctk.CTkFrame(param_frame)
            row.pack(fill="x", pady=1)
            ctk.CTkLabel(row, text=label, width=280, anchor="w").pack(side="left")
            e = ctk.CTkEntry(row, width=120)
            e.insert(0, str(default))
            e.pack(side="right")
            self.entries[label] = e

        # Чекбоксы: что перебирать в авто-режиме
        self.auto_vars = {}
        check_frame = ctk.CTkFrame(param_frame)
        check_frame.pack(fill="x", pady=5)
        ctk.CTkLabel(check_frame, text="Перебирать в авто-режиме:").pack(anchor="w")
        for name in ["SPIKE_MIN_PCT", "SPIKE_SEQ_PCT", "LOOKBACK_MINUTES",
                     "HIDDEN_LAYERS", "LEARNING_RATE"]:
            var = ctk.BooleanVar(value=True)
            self.auto_vars[name] = var
            ctk.CTkCheckBox(check_frame, text=name, variable=var).pack(side="left", padx=5)

    # ---------- Блок 2: график ----------
    def _build_plot(self):
        self.plot_frame = ctk.CTkFrame(self)
        self.plot_frame.pack(fill="both", expand=True, padx=10, pady=5)

        self.fig = Figure(figsize=(10, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Цена BTC и скачки")
        self.ax.set_xlabel("Индекс свечи")
        self.ax.set_ylabel("Цена, USDT")

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        toolbar = NavigationToolbar2Tk(self.canvas, self.plot_frame)
        toolbar.update()
        toolbar.pack(side="bottom", fill="x")

    # ---------- Блок 3: отчёт ----------
    def _build_report(self):
        self.report = ctk.CTkTextbox(self, height=180)
        self.report.pack(fill="x", padx=10, pady=5)

    # ---------- Обработчики кнопок ----------
    def _run_async(self, func, *args):
        """Запускает тяжёлую функцию в отдельном потоке, чтобы GUI не зависал."""
        threading.Thread(target=func, args=args, daemon=True).start()

    def on_load(self):
        self._run_async(self._load)

    def _load(self):
        log("=== Загрузка свечей ===", self.report)
        self.df = data_loader.load_klines(
            progress_callback=lambda p: log(f"Прогресс: {p*100:.1f}%", self.report)
        )
        log(f"Готово. Свечей: {len(self.df)}", self.report)

    def on_build(self):
        self._run_async(self._build)

    def _build(self):
        if self.df is None:
            log("Сначала загрузите свечи!", self.report)
            return
        log("=== Построение датасета ===", self.report)
        df_feat = dataset_builder.build_features(self.df)
        X, Y = dataset_builder.build_dataset(df_feat)
        self.dataset = (X, Y)
        self.df_feat = df_feat
        log(f"X: {X.shape}, Y: {Y.shape}", self.report)

    def on_train(self):
        self._run_async(self._train)

    def _train(self):
        if self.dataset is None:
            log("Сначала постройте датасет!", self.report)
            return
        log("=== Обучение модели ===", self.report)
        X, Y = self.dataset
        self.model, self.metrics, self.history = trainer.train_model(
            X, Y, log_widget=self.report,
            progress_callback=lambda p: None
        )
        log("Обучение завершено.", self.report)

    def on_plot(self):
        self._run_async(self._plot)

    def _plot(self):
        if self.df is None:
            log("Нет данных для графика.", self.report)
            return
        self.ax.clear()
        df = self.df.reset_index(drop=True)
        self.ax.plot(df.index, df["close"], linewidth=0.5, label="Close")

        # Отмечаем скачки, если датасет построен
        if hasattr(self, "df_feat"):
            spikes = dataset_builder.find_spikes(self.df_feat)
            self.ax.scatter(spikes.index, spikes["close"],
                            color="red", s=10, label="Скачки")

        # Отмечаем удачные / ошибочные предсказания
        if self.model is not None and self.dataset is not None:
            import torch
            X, Y = self.dataset
            device = next(self.model.parameters()).device
            with torch.no_grad():
                pred = self.model(torch.tensor(X, dtype=torch.float32).to(device)).cpu().numpy()
            # Для простоты берём горизонт 1 минута
            err = np.abs(pred[:, 0] - Y[:, 0])
            good = err < np.percentile(err, 50)
            bad = ~good
            # Индексы в исходном df: скачки находятся по индексам df_feat
            spike_idx = dataset_builder.find_spikes(self.df_feat).index
            # Отображаем только те, для которых есть предсказание
            n = min(len(spike_idx), len(good))
            self.ax.scatter(spike_idx[:n][good[:n]],
                            self.df_feat["close"].iloc[spike_idx[:n][good[:n]]],
                            color="lime", s=15, label="Верно")
            self.ax.scatter(spike_idx[:n][bad[:n]],
                            self.df_feat["close"].iloc[spike_idx[:n][bad[:n]]],
                            color="orange", s=15, label="Ошибка")

        self.ax.legend()
        self.canvas.draw()
        log("График обновлён.", self.report)

    def on_auto(self):
        self._run_async(self._auto)

    def _auto(self):
        from auto_mode import run_auto
        run_auto(self)


def run_app():
    app = App()
    app.mainloop()
"""
GUI на CustomTkinter.

Структура окна:
    ┌──────────────────────────────────────────┐
    │ 1. Панель кнопок и параметров           │  (фиксированная высота)
    ├──────────────────────────────────────────┤
    │ 2. График с тулбаром и скроллом         │  ← плавающий разделитель
    │    ═══════════════════════════════════  │
    │ 3. Текстовый отчёт                      │
    └──────────────────────────────────────────┘

Что нового по сравнению с предыдущей версией:
- вернули тулбар matplotlib (зум, пан, сохранение PNG);
- график растягивается на весь блок 2 и по ширине, и по высоте;
- если пользователь тянет ползунок «мин. ширина» дальше размера окна,
  появляется горизонтальный скроллбар;
- между блоком 2 и блоком 3 — плавающий разделитель (sash), можно тянуть мышью.
"""

import threading
import tkinter as tk
import customtkinter as ctk
import matplotlib
matplotlib.use("TkAgg")
import numpy as np
import torch
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

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
        self.geometry("1300x950")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # Данные, заполняются по ходу работы
        self.df = None
        self.dataset = None
        self.model = None
        self.metrics = None
        self.history = None
        self.predictions = None
        self.df_feat = None
        self.spike_idx = None

        self._build_top()
        self._build_split()   # блоки 2 и 3 + плавающий разделитель

    # ------------------------------------------------------------------
    # Блок 1: кнопки и параметры
    # ------------------------------------------------------------------
    def _build_top(self):
        frame = ctk.CTkFrame(self, height=190)
        frame.pack(fill="x", padx=10, pady=(10, 5))
        frame.pack_propagate(False)

        # --- Кнопки ---
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

        # --- Параметры ---
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
        for label, default in params:
            row = ctk.CTkFrame(param_frame)
            row.pack(fill="x", pady=1)
            ctk.CTkLabel(row, text=label, width=280, anchor="w").pack(side="left")
            e = ctk.CTkEntry(row, width=120)
            e.insert(0, str(default))
            e.pack(side="right")
            self.entries[label] = e

        # --- Чекбоксы авто-режима ---
        self.auto_vars = {}
        check_frame = ctk.CTkFrame(param_frame)
        check_frame.pack(fill="x", pady=5)
        ctk.CTkLabel(check_frame, text="Перебирать в авто-режиме:").pack(anchor="w")
        for name in ["SPIKE_MIN_PCT", "SPIKE_SEQ_PCT", "LOOKBACK_MINUTES",
                     "HIDDEN_LAYERS", "LEARNING_RATE"]:
            var = ctk.BooleanVar(value=True)
            self.auto_vars[name] = var
            ctk.CTkCheckBox(check_frame, text=name, variable=var).pack(side="left", padx=5)

    # ------------------------------------------------------------------
    # Блоки 2 и 3 с плавающим разделителем
    # ------------------------------------------------------------------
    def _build_split(self):
        """
        tk.PanedWindow — стандартный tk-контейнер с «плавающим» разделителем.
        Пользователь тянет серую полосу мышью и меняет высоту блоков.
        В CustomTkinter своего аналога нет, поэтому используем чистый tk.
        """
        self.paned = tk.PanedWindow(
            self,
            orient="vertical",       # делим сверху вниз
            sashwidth=8,             # толщина разделителя, px
            sashrelief="raised",
            showhandle=True,         # показываем «ручку» на разделителе
            handlesize=8,
            bg="#3a3a3a",            # цвет фона вокруг разделителя
            bd=0,
        )
        self.paned.pack(fill="both", expand=True, padx=10, pady=(5, 10))

        # --- Блок 2: график ---
        self.plot_container = ctk.CTkFrame(self.paned)
        # stretch="always" — при растяжении окна лишнее место уходит сюда
        self.paned.add(self.plot_container, minsize=200, stretch="always")

        # --- Блок 3: отчёт ---
        self.report_container = ctk.CTkFrame(self.paned)
        self.paned.add(self.report_container, minsize=80, stretch="never")

        self._build_plot(self.plot_container)
        self._build_report(self.report_container)

    # ------------------------------------------------------------------
    # Блок 2: график
    # ------------------------------------------------------------------
    def _build_plot(self, parent):
        # --- Панель управления графиком (сверху) ---
        ctrl = ctk.CTkFrame(parent)
        ctrl.pack(fill="x", padx=5, pady=(5, 2))

        ctk.CTkLabel(ctrl, text="Мин. ширина (дюймы):").pack(side="left", padx=5)
        self.width_slider = ctk.CTkSlider(
            ctrl, from_=6, to=80, number_of_steps=74,
            command=self._on_width_slider,
        )
        self.width_slider.set(12)
        self.width_slider.pack(side="left", fill="x", expand=True, padx=5)

        self.width_label = ctk.CTkLabel(ctrl, text='12"', width=50)
        self.width_label.pack(side="left", padx=5)

        self.autoscale_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            ctrl, text="Auto-scale Y по видимой области",
            variable=self.autoscale_var,
            command=self._update_plot,
        ).pack(side="left", padx=15)

        # --- Тулбар matplotlib (внизу) ---
        # Создаём заранее как ctk-фрейм-обёртку, внутрь кладём сам тулбар.
        toolbar_frame = ctk.CTkFrame(parent, height=32)
        toolbar_frame.pack(side="bottom", fill="x")

        # --- Горизонтальный скроллбар (внизу, над тулбаром) ---
        hscroll = tk.Scrollbar(parent, orient="horizontal")
        hscroll.pack(side="bottom", fill="x")

        # --- Canvas со скроллом занимает всё оставшееся место ---
        self.hcanvas = tk.Canvas(parent, bg="#2b2b2b", highlightthickness=0)
        self.hcanvas.pack(fill="both", expand=True, side="top")
        self.hcanvas.configure(xscrollcommand=hscroll.set)
        hscroll.configure(command=self.hcanvas.xview)
        self.hcanvas.bind("<Configure>", self._on_canvas_configure)

        # --- Matplotlib ---
        self.fig = Figure(figsize=(12, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Цена BTC и скачки")
        self.ax.set_xlabel("Индекс свечи")
        self.ax.set_ylabel("Цена, USDT")

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.hcanvas)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget_id = self.hcanvas.create_window(
            (0, 0), window=self.canvas_widget, anchor="nw"
        )

        # --- Тулбар (после создания canvas) ---
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()
        self.toolbar.pack(side="left", fill="x", expand=True)
        # Подгоняем цвет тулбара под тёмную тему
        try:
            self.toolbar.configure(bg="#2b2b2b")
        except Exception:
            pass

        self.canvas.draw()
        # Авто-масштаб Y при зуме/пане
        self.ax.callbacks.connect("xlim_changed", self._on_xlim_changed)

    def _on_canvas_configure(self, event):
        """
        Срабатывает при любом изменении размеров блока 2
        (растяжение окна, движение разделителя).
        Пересчитывает размер matplotlib-фигуры.
        """
        self._relayout_figure(event.width, event.height)

    def _relayout_figure(self, width_px, height_px):
        """
        Ключевая функция «растягивания»:
        - ширина = max(min_width_из_слайдера, доступная_ширина);
        - высота = доступная_высота (график всегда вписан в блок 2).

        Если min_width > доступной ширины, фигура становится шире окна,
        и появляется горизонтальный скроллбар.
        """
        if width_px < 30 or height_px < 30:
            return

        dpi = self.fig.dpi
        min_w_px = int(self.width_slider.get() * dpi)
        new_w_px = max(min_w_px, width_px)
        new_h_px = height_px

        cur_w_px = int(self.fig.get_figwidth() * dpi)
        cur_h_px = int(self.fig.get_figheight() * dpi)

        # Небольшой допуск, чтобы не дёргать перерисовку на каждый пиксель
        if abs(cur_w_px - new_w_px) < 3 and abs(cur_h_px - new_h_px) < 3:
            return

        self.fig.set_size_inches(new_w_px / dpi, new_h_px / dpi)
        self.hcanvas.itemconfig(
            self.canvas_widget_id, width=new_w_px, height=new_h_px
        )
        self.hcanvas.configure(scrollregion=(0, 0, new_w_px, new_h_px))
        self.canvas.draw_idle()

    def _on_width_slider(self, value):
        """Ползунок меняет минимальную ширину фигуры."""
        self.width_label.configure(text=f'{int(value)}"')
        # Даём tkinter применить новое значение, потом пересчитываем layout
        self.after(10, lambda: self._relayout_figure(
            self.hcanvas.winfo_width(), self.hcanvas.winfo_height()
        ))

    def _on_xlim_changed(self, ax):
        """Пользователь зумит/панорамирует — пересчитываем вертикаль."""
        self._autoscale_y()
        self.canvas.draw_idle()

    def _autoscale_y(self):
        """
        Подгоняет Y-диапазон под данные, видимые на текущем X-диапазоне.
        Учитывает и линии, и scatter-точки.
        """
        if not getattr(self, "autoscale_var", None):
            return
        if not self.autoscale_var.get():
            return

        xlim = self.ax.get_xlim()
        ymin, ymax = float("inf"), float("-inf")

        # Линии (цена)
        for line in self.ax.get_lines():
            xdata = np.asarray(line.get_xdata())
            ydata = np.asarray(line.get_ydata())
            if xdata.size == 0:
                continue
            mask = (xdata >= xlim[0]) & (xdata <= xlim[1])
            if mask.any():
                ymin = min(ymin, float(ydata[mask].min()))
                ymax = max(ymax, float(ydata[mask].max()))

        # Scatter-точки
        for coll in self.ax.collections:
            offsets = coll.get_offsets()
            if len(offsets) == 0:
                continue
            offsets = np.asarray(offsets)
            mask = (offsets[:, 0] >= xlim[0]) & (offsets[:, 0] <= xlim[1])
            if mask.any():
                ymin = min(ymin, float(offsets[mask, 1].min()))
                ymax = max(ymax, float(offsets[mask, 1].max()))

        if ymin < float("inf") and ymax > float("-inf") and ymax > ymin:
            pad = (ymax - ymin) * 0.05
            self.ax.set_ylim(ymin - pad, ymax + pad)

    def _update_plot(self):
        """Вызывается чекбоксом autoscale."""
        self._autoscale_y()
        self.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Блок 3: отчёт
    # ------------------------------------------------------------------
    def _build_report(self, parent):
        self.report = ctk.CTkTextbox(parent, height=150)
        self.report.pack(fill="both", expand=True, padx=5, pady=5)

    # ------------------------------------------------------------------
    # Обработчики кнопок
    # ------------------------------------------------------------------
    def _run_async(self, func, *args):
        """Тяжёлые функции — в отдельном потоке, чтобы GUI не подвисал."""
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
        X, Y, orig_idx = dataset_builder.build_dataset(df_feat)
        self.dataset = (X, Y)
        self.df_feat = df_feat
        self.spike_idx = orig_idx  # <-- сохраняем исходные позиции скачков
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
        )
        log("Обучение завершено.", self.report)

    def on_plot(self):
        self._run_async(self._plot)

    def _plot(self):
        """
        Рисует цену BTC и маркеры скачков.
        ВАЖНО: и линия, и все маркеры используют ОДНУ И ТУ ЖЕ ось X —
        позиции строк в сыром df (0 .. N-1). Тогда при любом зуме ничего не смещается.
        """
        if self.df is None:
            log("Нет данных для графика.", self.report)
            return

        # Сырой df, индексы 0..N-1
        df = self.df.reset_index(drop=True)

        self.ax.clear()
        self.ax.plot(df.index, df["close"], linewidth=0.5, color="#4fc3f7",
                     label="Close", zorder=1)

        # --- Скачки ---
        if self.df_feat is None or "original_index" not in self.df_feat.columns:
            log("Скачки не будут показаны: сначала нажмите "
                "«2. Построить датасет».", self.report)
        else:
            spikes = dataset_builder.find_spikes(self.df_feat)
            # x — ИСХОДНЫЕ позиции в df, y — цена закрытия в момент скачка.
            x = spikes["original_index"].values
            y = spikes["close"].values
            # Крупные треугольники с обводкой — видны даже на плотном графике.
            self.ax.scatter(x, y, marker="^", s=45, c="red",
                            edgecolors="black", linewidths=0.4,
                            zorder=5, label=f"Скачки ({len(x)})")

        # --- Предсказания модели ---
        if self.model is not None and self.spike_idx is not None:
            X, Y = self.dataset
            device = next(self.model.parameters()).device
            with torch.no_grad():
                pred = self.model(
                    torch.tensor(X, dtype=torch.float32).to(device)
                ).cpu().numpy()

            # Ошибка по горизонту 1 мин — показываем «верно/ошибка»
            err = np.abs(pred[:, 0] - Y[:, 0])
            good = err < np.percentile(err, 50)  # лучшая половина
            bad = ~good

            # self.spike_idx — те же исходные позиции, что использовались
            # при построении датасета (в том же порядке, что X и Y).
            positions = self.spike_idx
            prices = df["close"].values[positions]

            self.ax.scatter(positions[good], prices[good],
                            marker="o", s=30, c="lime",
                            edgecolors="black", linewidths=0.3,
                            zorder=6, label="Верно")
            self.ax.scatter(positions[bad], prices[bad],
                            marker="o", s=30, c="orange",
                            edgecolors="black", linewidths=0.3,
                            zorder=6, label="Ошибка")

        self.ax.legend(loc="upper left")
        self._autoscale_y()
        self.canvas.draw()
        log(f"График обновлён. Точек цены: {len(df)}, "
            f"скачков отмечено: "
            f"{len(self.df_feat) and len(dataset_builder.find_spikes(self.df_feat))}",
            self.report)

    def on_auto(self):
        self._run_async(self._auto)

    def _auto(self):
        from auto_mode import run_auto
        run_auto(self)


def run_app():
    app = App()
    app.mainloop()
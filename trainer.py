"""
Обучение MLP с early stopping и L2-регуляризацией.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error

from config import (
    MODEL_DIR, EPOCHS, BATCH_SIZE, LEARNING_RATE,
    TRAIN_SPLIT, RANDOM_SEED, HIDDEN_LAYERS, HORIZONS,
    EARLY_STOP_PATIENCE, EARLY_STOP_MIN_DELTA, WEIGHT_DECAY,
)
from model import MLPRegressor
from utils import log


def set_seed(seed=RANDOM_SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class EarlyStopping:
    """
    Останавливает обучение, если val_loss не улучшается N эпох подряд.

    Аналогия: если студент 15 занятий подряд не улучшает результат
    на пробных экзаменах — дальше зубрить бесполезно, он уже переобучился.
    """
    def __init__(self, patience=15, min_delta=1e-5):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.best_epoch = 0
        self.counter = 0
        self.should_stop = False

    def step(self, val_loss, epoch):
        if val_loss < self.best - self.min_delta:
            self.best = val_loss
            self.best_epoch = epoch
            self.counter = 0
            return True
        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False


def train_model(X, Y, log_widget=None, progress_callback=None,
                hidden_layers=None, lr=None, epochs=None,
                weight_decay=None, patience=None):
    """
    Обучает модель. Возвращает (model, metrics, history).
    """
    set_seed()
    hidden_layers = hidden_layers or HIDDEN_LAYERS
    lr = lr or LEARNING_RATE
    epochs = epochs or EPOCHS
    weight_decay = weight_decay if weight_decay is not None else WEIGHT_DECAY
    patience = patience or EARLY_STOP_PATIENCE

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Устройство: {device} | lr={lr} | wd={weight_decay} | patience={patience}",
        log_widget)

    # --- Train / val split ---
    n = len(X)
    n_train = int(n * TRAIN_SPLIT)
    idx = np.random.permutation(n)
    train_idx, val_idx = idx[:n_train], idx[n_train:]

    X_train = torch.tensor(X[train_idx], dtype=torch.float32)
    Y_train = torch.tensor(Y[train_idx], dtype=torch.float32)
    X_val = torch.tensor(X[val_idx], dtype=torch.float32)
    Y_val = torch.tensor(Y[val_idx], dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train, Y_train),
                              batch_size=BATCH_SIZE, shuffle=True)

    input_dim = X.shape[1] * X.shape[2]
    output_dim = Y.shape[1]
    model = MLPRegressor(input_dim, output_dim, hidden_layers).to(device)

    # weight_decay = L2-регуляризация: штрафует большие веса.
    # Это мешает модели «зазубривать» отдельные примеры.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=weight_decay)
    criterion = nn.MSELoss()

    best_path = os.path.join(MODEL_DIR, "best_model.pt")
    history = {"train": [], "val": []}
    stopper = EarlyStopping(patience=patience, min_delta=EARLY_STOP_MIN_DELTA)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val.to(device))
            val_loss = criterion(val_pred, Y_val.to(device)).item()

        history["train"].append(train_loss)
        history["val"].append(val_loss)

        improved = stopper.step(val_loss, epoch)
        if improved:
            torch.save({
                "model_state": model.state_dict(),
                "input_dim": input_dim,
                "output_dim": output_dim,
                "hidden_layers": hidden_layers,
                "feature_cols": None,  # заполним позже при желании
            }, best_path)

        if progress_callback:
            progress_callback(epoch / epochs)

        if epoch % 5 == 0 or epoch == 1 or improved:
            marker = "  *best" if improved else ""
            log(f"Эпоха {epoch:3d} | train {train_loss:.6f} | "
                f"val {val_loss:.6f}{marker}", log_widget)

        if stopper.should_stop:
            log(f"Early stopping: val не улучшался {patience} эпох. "
                f"Лучшая эпоха = {stopper.best_epoch} (val={stopper.best:.6f})",
                log_widget)
            break

    # Загружаем лучшую модель и считаем метрики
    model.load_state_dict(torch.load(best_path, map_location=device)["model_state"])
    model.eval()
    with torch.no_grad():
        preds = model(X_val.to(device)).cpu().numpy()
    trues = Y_val.numpy()

    metrics = {}
    for i, h in enumerate(HORIZONS):
        mae = mean_absolute_error(trues[:, i], preds[:, i])
        rmse = np.sqrt(mean_squared_error(trues[:, i], preds[:, i]))
        metrics[f"h{h}_MAE"] = float(mae)
        metrics[f"h{h}_RMSE"] = float(rmse)

    # Baseline: «всегда предсказываем среднее» — точка отсчёта
    baseline_pred = np.tile(trues.mean(axis=0), (len(trues), 1))
    baseline_mae = np.mean(np.abs(baseline_pred - trues), axis=0)
    log(f"Метрики модели: {metrics}", log_widget)
    log(f"Baseline (среднее) MAE по горизонтам: "
        f"{[round(float(x), 4) for x in baseline_mae]}", log_widget)
    for i, h in enumerate(HORIZONS):
        diff = baseline_mae[i] - metrics[f"h{h}_MAE"]
        verdict = "лучше baseline" if diff > 0 else "хуже/на уровне baseline"
        log(f"  h={h} мин: улучшение относительно baseline "
            f"{diff:+.4f}% ({verdict})", log_widget)

    return model, metrics, history
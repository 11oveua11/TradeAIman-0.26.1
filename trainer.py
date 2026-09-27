"""
Обучение MLP и расчёт метрик.

Что делает:
1. Делит датасет на train / val.
2. Обучает модель MSE-лоссом.
3. Считает MAE, RMSE по каждому горизонту.
4. Сохраняет лучшую модель на диск.
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
)
from model import MLPRegressor
from utils import log


def set_seed(seed=RANDOM_SEED):
    """Фиксирует seed, чтобы результаты воспроизводились."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_model(X, Y, log_widget=None, progress_callback=None,
                hidden_layers=None, lr=None, epochs=None):
    """
    Обучает модель и возвращает:
    - обученную модель,
    - словарь метрик,
    - историю лосса (для графика).
    """
    set_seed()
    hidden_layers = hidden_layers or HIDDEN_LAYERS
    lr = lr or LEARNING_RATE
    epochs = epochs or EPOCHS

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Устройство: {device}", log_widget)

    # --- Разделение на train / val ---
    n = len(X)
    n_train = int(n * TRAIN_SPLIT)
    idx = np.random.permutation(n)
    train_idx, val_idx = idx[:n_train], idx[n_train:]

    X_train = torch.tensor(X[train_idx], dtype=torch.float32)
    Y_train = torch.tensor(Y[train_idx], dtype=torch.float32)
    X_val = torch.tensor(X[val_idx], dtype=torch.float32)
    Y_val = torch.tensor(Y[val_idx], dtype=torch.float32)

    train_ds = TensorDataset(X_train, Y_train)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    # --- Модель ---
    input_dim = X.shape[1] * X.shape[2]
    output_dim = Y.shape[1]
    model = MLPRegressor(input_dim, output_dim, hidden_layers).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val = float("inf")
    best_path = os.path.join(MODEL_DIR, "best_model.pt")
    history = {"train": [], "val": []}

    for epoch in range(1, epochs + 1):
        # --- Обучение ---
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
        train_loss /= len(train_ds)

        # --- Валидация ---
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val.to(device))
            val_loss = criterion(val_pred, Y_val.to(device)).item()

        history["train"].append(train_loss)
        history["val"].append(val_loss)

        if progress_callback:
            progress_callback(epoch / epochs)

        if epoch % 10 == 0 or epoch == 1:
            log(f"Эпоха {epoch:3d} | train {train_loss:.6f} | val {val_loss:.6f}",
                log_widget)

        # Сохраняем лучшую модель
        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model_state": model.state_dict(),
                "input_dim": input_dim,
                "output_dim": output_dim,
                "hidden_layers": hidden_layers,
            }, best_path)

    # --- Метрики на валидации ---
    model.load_state_dict(torch.load(best_path, map_location=device)["model_state"])
    model.eval()
    with torch.no_grad():
        preds = model(X_val.to(device)).cpu().numpy()
    trues = Y_val.numpy()

    metrics = {}
    for i, h in enumerate(HORIZONS):
        mae = mean_absolute_error(trues[:, i], preds[:, i])
        rmse = np.sqrt(mean_squared_error(trues[:, i], preds[:, i]))
        metrics[f"h{h}_MAE"] = mae
        metrics[f"h{h}_RMSE"] = rmse

    log(f"Метрики: {metrics}", log_widget)
    return model, metrics, history
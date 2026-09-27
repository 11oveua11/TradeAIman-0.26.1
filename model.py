"""
Простая полносвязная нейросеть (MLP) для регрессии.

Вход: «сплющенное» окно признаков (lookback * n_features).
Выход: len(HORIZONS) чисел — предсказанная коррекция в %.
"""

import torch
import torch.nn as nn
from config import HIDDEN_LAYERS, DROPOUT

class MLPRegressor(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers=None, dropout=None):
        """
        input_dim: размер входного вектора (lookback * features)
        output_dim: число предсказываемых горизонтов
        hidden_layers: список размеров скрытых слоёв, например [128, 64]
        """
        super().__init__()
        hidden_layers = hidden_layers or HIDDEN_LAYERS
        dropout = dropout if dropout is not None else DROPOUT

        layers = []
        prev = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: (batch, lookback, features) -> (batch, lookback*features)
        x = x.view(x.size(0), -1)
        return self.net(x)
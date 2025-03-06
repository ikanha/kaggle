import torch
import torch.nn as nn


class SMAPELoss(nn.Module):
    def __init__(self, epsilon: float = 1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, pred, target):
        numerator = 2.0 * (pred - target).abs()
        denominator = pred.abs() + target.abs() + self.epsilon
        return 100.0 * torch.mean(numerator / denominator)


def mape_loss(y_pred, y_true, epsilon: float = 1e-8):
    numerator = (y_true - y_pred).abs()
    denominator = (y_true.abs() + epsilon)
    return 100.0 * torch.mean(numerator / denominator)


def to_device_helper(objects: tuple, device: str) -> tuple:
    return tuple([obj.to(device) for obj in objects])


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=16):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)  # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)
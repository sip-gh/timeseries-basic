import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric Mean Absolute Percentage Error (0〜2の間)."""
    denom = (np.abs(y_true) + np.abs(y_pred)) + 1e-8  # 0除算防止のため微小値を足す
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom))

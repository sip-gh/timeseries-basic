import numpy as np

from src.metrics import rmse, smape


def test_rmse_simple_case() -> None:
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 4.0])
    expected = np.sqrt(1.0 / 3.0)
    # np.isclose: 浮動小数の「ほぼ等しい」を判定する関数（丸め誤差対策）
    assert np.isclose(rmse(y_true, y_pred), expected)


def test_smape_zero_error() -> None:
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([10.0, 20.0, 30.0])
    assert smape(y_true, y_pred) == 0.0


def test_smape_simple_asymmetric() -> None:
    y_true = np.array([10.0])
    y_pred = np.array([20.0])
    expected = 2.0 / 3.0
    assert np.isclose(smape(y_true, y_pred), expected)

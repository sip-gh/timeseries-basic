import numpy as np
import pandas as pd

from src.features import add_lag_ma_features


def test_add_lag_ma_features_basic_columns() -> None:
    df = pd.DataFrame({"sales": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]})

    out = add_lag_ma_features(df, target_col="sales")

    assert "lag_1" in out.columns
    assert "ma_7" in out.columns
    assert "ret_1" in out.columns
    assert "dist_ma_7" in out.columns
    assert "vol_7" in out.columns


def test_add_lag_ma_features_lag_1_value() -> None:
    df = pd.DataFrame({"sales": [10.0, 20.0, 30.0]})

    out = add_lag_ma_features(df, target_col="sales")

    assert out.loc[1, "lag_1"] == 10.0


def test_add_lag_ma_features_ma_7_value() -> None:
    df = pd.DataFrame({"sales": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]})

    out = add_lag_ma_features(df, target_col="sales")

    # np.isclose: 浮動小数の「ほぼ等しい」を判定する関数
    assert np.isclose(out.loc[6, "ma_7"], 40.0)


def test_add_lag_ma_features_inf_is_replaced_with_nan() -> None:
    df = pd.DataFrame({"sales": [0.0, 10.0, 20.0]})

    out = add_lag_ma_features(df, target_col="sales")

    # 前日0による pct_change の inf を NaN に置き換えていることを確認
    assert np.isnan(out.loc[1, "ret_1"])

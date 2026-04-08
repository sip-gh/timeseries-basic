# src/features.py
import numpy as np
import pandas as pd


def add_lag_ma_features(
    df: pd.DataFrame,
    target_col: str = "sales",
) -> pd.DataFrame:
    """単一系列の売上データに、ラグ・移動平均・変化率の特徴量を追加する。"""
    out = df.copy()

    # 1日前の値（ナイーブ予測にも使える）
    out["lag_1"] = out[target_col].shift(1)

    # 移動平均
    out["ma_7"] = out[target_col].rolling(7).mean()
    out["ma_14"] = out[target_col].rolling(14).mean()

    # 変化率
    out["ret_1"] = out[target_col].pct_change(1)
    out["ret_7"] = out[target_col].pct_change(7)

    # 移動平均との差（相対距離）
    out["dist_ma_7"] = (out[target_col] - out["ma_7"]) / out["ma_7"]
    out["dist_ma_14"] = (out[target_col] - out["ma_14"]) / out["ma_14"]

    # 変化率のボラティリティ
    out["vol_7"] = out["ret_1"].rolling(7).std()

    # 0除算などで出る inf を NaN に置き換える
    out = out.replace([np.inf, -np.inf], np.nan)

    return out

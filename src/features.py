# src/features.py
from typing import Sequence
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


def add_store_sales_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Store Sales データの date 列からカレンダー特徴量を追加する。

    Parameters
    ----------
    df : pd.DataFrame
        date 列を含むDataFrame。date 列は datetime64 型を想定する。

    Returns
    -------
    pd.DataFrame
        カレンダー特徴量を追加したDataFrameのコピー。
    """
    out = df.copy()

    out["year"] = out["date"].dt.year
    out["month"] = out["date"].dt.month
    out["day"] = out["date"].dt.day
    out["dow"] = out["date"].dt.dayofweek
    out["weekofyear"] = out["date"].dt.isocalendar().week.astype(int)

    out["is_weekend"] = out["dow"].isin([5, 6]).astype(int)
    out["is_month_start"] = out["date"].dt.is_month_start.astype(int)
    out["is_month_end"] = out["date"].dt.is_month_end.astype(int)
    out["is_quarter_end"] = out["date"].dt.is_quarter_end.astype(int)
    out["is_year_end"] = out["date"].dt.is_year_end.astype(int)

    return out


def add_store_sales_group_lag_ma_features(
    df: pd.DataFrame,
    group_cols: Sequence[str] = ("store_nbr", "family"),
    target_col: str = "sales",
    lags: Sequence[int] = (1, 7, 14),
    windows: Sequence[int] = (7, 14, 28),
    drop_na: bool = True,
) -> pd.DataFrame:
    """
    Store Sales の多系列データに対して、店舗×familyごとの
    ラグ特徴量と移動平均特徴量を追加する。

    Parameters
    ----------
    df : pd.DataFrame
        date 列、group_cols、target_col を含むDataFrame。
    group_cols : Sequence[str], default ("store_nbr", "family")
        系列を識別するグループ列。
    target_col : str, default "sales"
        ラグ・移動平均を計算する対象列。
    lags : Sequence[int], default (1, 7, 14)
        追加するラグ日数。
    windows : Sequence[int], default (7, 14, 28)
        追加する移動平均窓。
    drop_na : bool, default True
        特徴量計算で発生したNaN行を最後に削除するかどうか。

    Returns
    -------
    pd.DataFrame
        ラグ特徴量・移動平均特徴量を追加したDataFrameのコピー。
    """
    out = df.copy()

    # shift / rolling が時系列順になるように並べる
    sort_cols = [*group_cols, "date"]
    out = out.sort_values(sort_cols)

    # ラグ特徴量
    for lag in lags:
        out[f"lag_{lag}"] = out.groupby(list(group_cols))[target_col].shift(lag)

    # 移動平均特徴量
    for window in windows:
        out[f"rolling_mean_{window}"] = (
            out.groupby(list(group_cols))[target_col]
            .shift(1)
            .rolling(window)
            .mean()
        )

    if drop_na:
        out = out.dropna().reset_index(drop=True)

    return out


def get_store_sales_feature_cols() -> list[str]:
    """
    Store Sales の LightGBM/線形回帰で使う特徴量カラム名を返す。

    Returns
    -------
    list[str]
        学習に使う特徴量カラム名のリスト。
    """
    return [
        "onpromotion",
        "year",
        "month",
        "day",
        "dow",
        "weekofyear",
        "is_weekend",
        "is_month_start",
        "is_month_end",
        "is_quarter_end",
        "is_year_end",
        "lag_1",
        "lag_7",
        "lag_14",
        "rolling_mean_7",
        "rolling_mean_14",
        "rolling_mean_28",
    ]

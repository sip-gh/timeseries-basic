import numpy as np
import pandas as pd

from src.features import (
    add_lag_ma_features,
    add_store_sales_group_lag_ma_features,
    get_store_family_series,
)


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


def test_add_store_sales_group_lag_ma_features_lag_per_group() -> None:
    # 2店舗 × 1ファミリー × 5日分の簡単なデータ
    df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=5).tolist()
            + pd.date_range("2020-01-01", periods=5).tolist(),
            "store_nbr": [1] * 5 + [2] * 5,
            "family": ["GROCERY I"] * 10,
            "sales": [10, 20, 30, 40, 50] + [100, 200, 300, 400, 500],
        }
    )


    out = add_store_sales_group_lag_ma_features(
        df,
        group_cols=("store_nbr", "family"),
        target_col="sales",
        lags=(1,),
        windows=(2,),
        drop_na=False,
    )

    # lag_1 列が追加されていること
    assert "lag_1" in out.columns

    # store_nbr=1, 2020-01-02 の lag_1 は 10 になるはず
    row_1_day2 = out[
        (out["store_nbr"] == 1)
        & (out["date"] == pd.Timestamp("2020-01-02"))
    ]
    assert row_1_day2["lag_1"].iloc[0] == 10

    # store_nbr=2, 2020-01-02 の lag_1 は 100 になるはず
    row_2_day2 = out[
        (out["store_nbr"] == 2)
        & (out["date"] == pd.Timestamp("2020-01-02"))
    ]
    assert row_2_day2["lag_1"].iloc[0] == 100

    # グループをまたいで値が混ざっていないこと（2店舗目の最初は NaN）
    row_2_first = out[
        (out["store_nbr"] == 2)
        & (out["date"] == pd.Timestamp("2020-01-01"))
    ]
    assert np.isnan(row_2_first["lag_1"].iloc[0])


def test_add_store_sales_group_lag_ma_features_drop_na() -> None:
    # 1店舗 × 1ファミリー × 5日（シンプルに落ちる行数を確認したいだけ）
    df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=5),
            "store_nbr": [1] * 5,
            "family": ["GROCERY I"] * 5,
            "sales": [10, 20, 30, 40, 50],
        }
    )

    out = add_store_sales_group_lag_ma_features(
        df,
        group_cols=("store_nbr", "family"),
        target_col="sales",
        lags=(1,),
        windows=(3,),
        drop_na=True,
    )

    # window=3, lag=1 → 先頭2行は NaN を含む → drop_na=True なら 3 行だけ残る想定
    assert len(out) == 2

    # 残るのは 2020-01-04, 2020-01-05
    assert list(out["date"]) == [
        pd.Timestamp("2020-01-04"),
        pd.Timestamp("2020-01-05"),
    ]


def test_get_store_family_series_filters_and_sorts() -> None:
    # 2店舗 × 2ファミリー × 3日分のデータ（わざと日付をシャッフル）
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2020-01-03", "2020-01-01", "2020-01-02",  # store 1, GROCERY I
                    "2020-01-01", "2020-01-02", "2020-01-03",  # store 1, BEVERAGES
                    "2020-01-01", "2020-01-02", "2020-01-03",  # store 2, GROCERY I
                ]
            ),
            "store_nbr": [1, 1, 1, 1, 1, 1, 2, 2, 2],
            "family": [
                "GROCERY I", "GROCERY I", "GROCERY I",
                "BEVERAGES", "BEVERAGES", "BEVERAGES",
                "GROCERY I", "GROCERY I", "GROCERY I",
            ],
            "sales": [10, 20, 30, 40, 50, 60, 100, 200, 300],
        }
    )

    out = get_store_family_series(df, store_nbr=1, family="GROCERY I")

    # 条件に合うのは store_nbr=1 & family=GROCERY I の3行だけ
    assert len(out) == 3
    assert out["store_nbr"].nunique() == 1
    assert out["family"].nunique() == 1
    assert out["store_nbr"].iloc[0] == 1
    assert out["family"].iloc[0] == "GROCERY I"

    # 日付が昇順に並んでいること（もともとシャッフルしておいた）
    assert list(out["date"]) == [
        pd.Timestamp("2020-01-01"),
        pd.Timestamp("2020-01-02"),
        pd.Timestamp("2020-01-03"),
    ]

def test_naive_1d_prediction_on_simple_series() -> None:
    # 7日分のシンプルな系列
    df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=7),
            "sales": [10, 20, 30, 40, 50, 60, 70],
        }
    )

    # Naive 1日前予測
    df["pred_naive_1d"] = df["sales"].shift(1)

    # 予測が「1日前の値」になっているかを確認
    expected = [np.nan, 10, 20, 30, 40, 50, 60]
    # NaN 比較のために isna / == で分けてチェック
    assert np.isnan(df["pred_naive_1d"].iloc[0])
    assert list(df["pred_naive_1d"].iloc[1:]) == expected[1:]


def test_ma7_prediction_on_simple_series() -> None:
    df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=7),
            "sales": [10, 20, 30, 40, 50, 60, 70],
        }
    )

    # 7日移動平均（Notebook と同じ rolling(7).mean()）
    df["pred_ma_7d"] = df["sales"].rolling(7).mean()

    # 最初の6行は NaN、7行目は全体の平均 40 になるはず
    assert df["pred_ma_7d"].iloc[:6].isna().all()
    assert df["pred_ma_7d"].iloc[6] == 40

# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: timeseries-basic
#     language: python
#     name: python3
# ---

# %%
import sys
from pathlib import Path
from typing import Iterable, Tuple

import lightgbm as lgb
import mlflow
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# notebooks_local/ から 1つ上（プロジェクトルート）を取得
PROJECT_ROOT = Path.cwd().parent

# プロジェクトルートを Python のモジュール探索パスに追加
# （src.* を import できるようにするため）
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features import (
    add_store_sales_calendar_features,
    add_store_sales_group_lag_ma_features,
    get_store_sales_feature_cols,
)
from src.time_series_cv import make_yearly_expanding_splits

# MLflow: プロジェクト直下の mlruns/ を tracking 用ディレクトリに指定
# 例: file:///Users/xxx/devs/timeseries-basic/mlruns
MLFLOW_TRACKING_URI = f"file://{PROJECT_ROOT / 'mlruns'}"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# このノート／スクリプト用の実験名を設定
# MLflow UI では "demand_lgbm_cv" という Experiment として表示される
mlflow.set_experiment("demand_lgbm_cv")



# %%
# 学習用データを読み込み、日付からカレンダー特徴量を追加
train_path = "../data/store-sales-time-series-forecasting/train.csv"
df = pd.read_csv(train_path, parse_dates=["date"])

df = add_store_sales_calendar_features(df)
df.head()


# %%
# 学習に使う基本カラムを整理
base_cols = [
    "date",
    "store_nbr",
    "family",
    "sales",         # 目的変数
    "onpromotion",   # 説明変数
    # カレンダー系
    "year", "month", "day", "dow", "weekofyear",
    "is_weekend", "is_month_start", "is_month_end",
    "is_quarter_end", "is_year_end",
]

df_feat = df[base_cols].copy()
df_feat.head()



# %%
target_col = "sales"

# 店舗×ファミリーごとにラグ・移動平均特徴量を追加
group_cols = ["store_nbr", "family"]
df_feat = add_store_sales_group_lag_ma_features(
    df_feat,
    group_cols=group_cols,
    target_col=target_col,
    lags=(1, 7, 14),
    windows=(7, 14, 28),
    drop_na=True,
)

df_feat.head()



# %%

# 学習に使う特徴量（あとで増減しやすいように関数で一元管理）
feature_cols = get_store_sales_feature_cols()

X = df_feat[feature_cols].copy()
y = df_feat["sales"].values
dates = df_feat["date"].values  # 分割確認用に日付も持っておく



# %%
# 年ベースのCV split（最後の3年をテストに使う）
splits = make_yearly_expanding_splits(df_feat["year"], n_folds=3)
splits_list = list(splits)



# %%
for i, (train_mask, test_mask) in enumerate(splits, start=1):
    print(f"Fold {i}")
    print("  train rows:", train_mask.sum())
    print("  test rows :", test_mask.sum())
    print("  train years:", sorted(df_feat.loc[train_mask, 'year'].unique()))
    print("  test years :", sorted(df_feat.loc[test_mask, 'year'].unique()))


# %% [markdown]
# ### LinearRegression

# %%
model = LinearRegression()

fold_results = []

for i, (train_mask, test_mask) in enumerate(splits, start=1):
    # ブーリアンマスクをそのまま使って行を選択
    X_train = X.loc[train_mask]
    y_train = y[train_mask]
    X_test  = X.loc[test_mask]
    y_test  = y[test_mask]

    # モデル学習
    model.fit(X_train, y_train)

    # 予測
    y_pred = model.predict(X_test)

    # 評価指標（RMSE）
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    fold_results.append({"fold": i, "rmse": rmse})

    print(f"Fold {i} RMSE: {rmse:.4f}")



# %% [markdown]
# ### LGBMRegressor

# %%
def run_lgbm_cv(
    params: dict,
    splits: Iterable[Tuple[np.ndarray, np.ndarray]],
    X: pd.DataFrame,
    y: np.ndarray,
) -> pd.DataFrame:
    """
    LightGBM用の時系列クロスバリデーションを1セット実行し、
    各foldのRMSEをまとめたDataFrameを返す。

    - splits: (train_mask, test_mask) の反復可能オブジェクト
              各マスクは bool の1次元配列（len == len(X)）を想定。
    - X: 特徴量DataFrame
    - y: 目的変数（1次元の NumPy 配列）
    """
    results = []

    for i, (train_mask, test_mask) in enumerate(splits, start=1):
        # ブーリアンマスクで行を選択
        X_train = X.loc[train_mask]
        y_train = y[train_mask]
        X_test  = X.loc[test_mask]
        y_test  = y[test_mask]

        # 毎foldごとに新しいモデルを作成
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train)

        # 予測とRMSE計算（明示的に MSE → sqrt で計算）
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)

        print(f"Fold {i} RMSE: {rmse:.4f}")
        results.append({"fold": i, "rmse": rmse})

    # 各foldのRMSEをまとめたDataFrameを返す
    return pd.DataFrame(results)



# %%
def run_naive_lag1_cv(
    splits: Iterable[Tuple[np.ndarray, np.ndarray]],
    X: pd.DataFrame,
    y: np.ndarray,
) -> pd.DataFrame:
    """
    lag_1 をそのまま予測として使う Naive ベースラインのCV。
    """
    results = []

    for i, (train_mask, test_mask) in enumerate(splits, start=1):
        # test 期間の特徴量と目的変数
        X_test = X.loc[test_mask]
        y_test = y[test_mask]

        # 1日前の売上（特徴量 lag_1）をそのまま予測に使う
        y_pred = X_test["lag_1"].values

        # RMSE 計算
        mse = mean_squared_error(y_test, y_pred)
        rmse = float(np.sqrt(mse))
        print(f"[Naive lag1] Fold {i} RMSE: {rmse:.4f}")
        results.append({"fold": i, "rmse": rmse})

    return pd.DataFrame(results)



# %%
# Naive (lag_1) ベースラインを MLflow に1本記録
with mlflow.start_run(run_name="naive_lag1_cv"):
    # パラメータ（ほぼ説明用）
    mlflow.log_param("model_type", "naive_lag1")
    mlflow.log_param("note", "use lag_1 as prediction")

    df_naive_cv = run_naive_lag1_cv(splits_list, X, y)

    mean_rmse_naive = df_naive_cv["rmse"].mean()
    mlflow.log_metric("rmse_mean", mean_rmse_naive)

    # rmse_df 用に列名を合わせておく
    df_naive_cv = df_naive_cv.rename(columns={"rmse": "rmse_naive_lag1"})


# %%
mlflow.set_experiment("demand_lgbm_cv")

# 3パターンの LightGBM 設定
# v1: ベースラインとなるLightGBMパラメータでCV
lgb_params_sets = [
    ("lgbm_v1_base", {
        "objective": "regression",
        "metric": "rmse",       # lgb内部用の指標
        "random_state": 42,
        "n_estimators": 200,
        "learning_rate": 0.1,
        "num_leaves": 31,
        "n_jobs": -1,
    }),
    # v2: n_estimators（木の本数）だけ増やしたバージョンでCV
    ("lgbm_v2_more_trees", {
        "objective": "regression",
        "metric": "rmse",
        "random_state": 42,
        "n_estimators": 500,   # ← 200 → 500 に増やす
        "learning_rate": 0.1,
        "num_leaves": 31,
        "n_jobs": -1,
    }),
    # v3: num_leaves（葉の数）だけ増やしたバージョンでCV
    ("lgbm_v3_more_leaves", {
        "objective": "regression",
        "metric": "rmse",
        "random_state": 42,
        "n_estimators": 200,   # 本数はベースラインと同じ
        "learning_rate": 0.1,
        "num_leaves": 63,      # ← 31 → 63 に増やす
        "n_jobs": -1,
    }),
]

lgb_results = []
# ---- ここから MLflow で1 run として記録 ----
for run_name, params in lgb_params_sets:
    with mlflow.start_run(run_name=run_name):
        # パラメータをログ
        mlflow.log_params(params)

        # CV 実行
        df_cv = run_lgbm_cv(params, splits_list, X, y)

        # fold 平均 RMSE を算出してログ
        mean_rmse = df_cv["rmse"].mean()
        mlflow.log_metric("rmse_mean", mean_rmse)

        # 後で結合するため列名を run ごとに変える
        df_cv = df_cv.rename(columns={"rmse": f"rmse_{run_name}"})
        lgb_results.append(df_cv)



# %%
# v1/v2/v3 の RMSE を fold ごとに横持ち結合
lgb_df_all = lgb_results[0].copy()
for df_cv in lgb_results[1:]:
    lgb_df_all = lgb_df_all.merge(df_cv, on="fold")

# Naive の RMSE を結合
lgb_df_all = lgb_df_all.merge(df_naive_cv, on="fold")

# 線形回帰の結果をDataFrameに（列名を rmse_linear に統一）
lr_df = pd.DataFrame(fold_results).rename(columns={"rmse": "rmse_linear"})

# 各モデルのRMSEをfold単位で横持ちに結合
rmse_df = lr_df.merge(lgb_df_all, on="fold")

# ベースライン（v1）との差分
rmse_df["rmse_diff"] = rmse_df["rmse_linear"] - rmse_df["rmse_lgbm_v1_base"]

rmse_df


# %% [markdown]
#

# %%
rmse_df.describe()


# %%
rmse_path = "../data/lgbm_cv_rmse.csv"
rmse_df.to_csv(rmse_path, index=False)
rmse_path


# %%
# README 用のモデル別 RMSE 平均サマリ
rmse_summary = pd.DataFrame({
    "model": [
        "naive_lag1",
        "linear",
        "lgbm_v1_base",
    ],
    "rmse_mean": [
        rmse_df["rmse_naive_lag1"].mean(),
        rmse_df["rmse_linear"].mean(),
        rmse_df["rmse_lgbm_v1_base"].mean(),
    ],
})

rmse_summary


# %%
# 全期間を使って LGBM v1 で学習（FI/SHAP 用）
lgb_params_v1 = {
    "objective": "regression",
    "metric": "rmse",
    "random_state": 42,
    "n_estimators": 200,
    "learning_rate": 0.1,
    "num_leaves": 31,
    "n_jobs": -1,
}

lgb_model_v1 = lgb.LGBMRegressor(**lgb_params_v1)
lgb_model_v1.fit(X, y)


# %%
plt.figure(figsize=(8, 6))

# LightGBM の特徴量重要度を可視化
ax = lgb.plot_importance(
    lgb_model_v1,
    importance_type="gain",   # 各特徴量が「損失改善」にどれだけ貢献したか
    max_num_features=20,      # 上位20特徴量だけ表示
)

# タイトルなどレイアウト調整
plt.title("LightGBM Feature Importance (gain)")
plt.tight_layout()

# PNGとして保存（README・MLflow用）
fig_path = "../data/feature_importance_lgbm_v1.png"
plt.savefig(fig_path, dpi=150)

# Notebook上にも表示
plt.show()

# 保存パスを確認用に返す
fig_path


# %%
# Feature Importance 図を MLflow に保存する run
with mlflow.start_run(run_name="lgbm_v1_feature_importance"):
    # 図を作成するのに使ったモデルのハイパーパラメータを記録
    mlflow.log_params(lgb_params_v1)
    # 保存したPNGファイルを artifact としてアップロード
    # MLflow UI の「Artifacts > figures」配下から確認できる
    mlflow.log_artifact(fig_path, artifact_path="figures")


# %% [markdown]
#

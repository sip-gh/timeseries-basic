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
#     name: timeseries-basic
# ---

# %%
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from lightgbm import LGBMRegressor

# データ読み込み
train_path = "../data/store-sales-time-series-forecasting/train.csv"
df = pd.read_csv(train_path, parse_dates=["date"])

# カレンダー特徴量の追加
df["year"] = df["date"].dt.year          # 年
df["month"] = df["date"].dt.month        # 月 (1-12)
df["day"] = df["date"].dt.day            # 日 (1-31)

df["dow"] = df["date"].dt.dayofweek      # 曜日 (0=月, 6=日)
df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)  # 何週目か[web:205]

# 週末フラグ (土日)
df["is_weekend"] = df["dow"].isin([5, 6]).astype(int)

# 月初・月末フラグ
df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
df["is_month_end"] = df["date"].dt.is_month_end.astype(int)

# 四半期末・年末フラグ
df["is_quarter_end"] = df["date"].dt.is_quarter_end.astype(int)
df["is_year_end"] = df["date"].dt.is_year_end.astype(int)

# 追加できたか軽く確認
df[[
    "date", "year", "month", "day", "dow", "weekofyear",
    "is_weekend", "is_month_start", "is_month_end",
    "is_quarter_end", "is_year_end"
]].head()


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

# 1. 時系列順にソートしておく
#    → グループごとのshift / rollingが「過去→未来」順になるようにするため
df_feat = df_feat.sort_values([
    "store_nbr",
    "family",
    "date",
])

# 2. 多系列の「グループキー」を定義
group_cols = ["store_nbr", "family"]

# 3. ラグ特徴量（過去1日, 7日, 14日の売上）
for lag in [1, 7, 14]:
    df_feat[f"lag_{lag}"] = (
        df_feat
        .groupby(group_cols)[target_col]  # 店舗×ファミリーごとにsalesをグループ化
        .shift(lag)                       # 各グループ内でlag日前の値にずらす
    )

# 4. 移動平均特徴量（過去7, 14, 28日の平均売上）
for window in [7, 14, 28]:
    df_feat[f"rolling_mean_{window}"] = (
        df_feat
        .groupby(group_cols)[target_col]  # 店舗×ファミリーごとにsalesをグループ化
        .shift(1)                         # 当日を含めないよう、まず1日分だけ過去にずらす
        .rolling(window)                  # 各グループ内で直近window行の窓を取る
        .mean()                           # その平均を計算
    )

# 5. ラグ・移動平均が計算できない先頭部分（NaN行）を削除
df_feat = df_feat.dropna().reset_index(drop=True)

df_feat.head()


# %%
# 学習に使う特徴量（あとで増減しやすいようにリストにまとめる）
feature_cols = [
    "onpromotion",
    "year", "month", "day", "dow", "weekofyear",
    "is_weekend", "is_month_start", "is_month_end",
    "is_quarter_end", "is_year_end",
    "lag_1", "lag_7", "lag_14",
    "rolling_mean_7", "rolling_mean_14", "rolling_mean_28",
]

X = df_feat[feature_cols].copy()
y = df_feat["sales"].values
dates = df_feat["date"].values  # 分割確認用に日付も持っておく


# %%
# 年ベースのCV split（最後の3年をテストに使う）
years = np.sort(df_feat["year"].unique())
cv_years = years[-3:]  # 例: [2015, 2016, 2017]

splits = []
for test_year in cv_years:
    train_mask = df_feat["year"] < test_year   # 過去すべてをtrain
    test_mask  = df_feat["year"] == test_year  # その年だけtest
    splits.append((train_mask, test_mask))


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


# %%
# とりあえずのパラメータ（十分シンプル）
lgb_params = {
    "objective": "regression",
    "metric": "rmse",       # lgb内部用の指標
    "random_state": 42,
    "n_estimators": 200,
    "learning_rate": 0.1,
    "num_leaves": 31,
    "n_jobs": -1,
}

lgb_results = []

for i, (train_mask, test_mask) in enumerate(splits, start=1):
    # ★ここは train_idx / test_idx ではなく、train_mask / test_mask をそのまま使う
    X_train = X.loc[train_mask]
    y_train = y[train_mask]
    X_test  = X.loc[test_mask]
    y_test  = y[test_mask]

    # モデル作成（毎foldで新しく作る）
    model = lgb.LGBMRegressor(**lgb_params)

    # 学習
    model.fit(X_train, y_train)

    # 予測
    y_pred = model.predict(X_test)

    # RMSE（外側で自分でも計算する）
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    lgb_results.append({"fold": i, "rmse": rmse})
    print(f"Fold {i} RMSE: {rmse:.4f}")


# %%
# 線形回帰の結果をDataFrameに
lr_df = pd.DataFrame(fold_results)        # columns: ["fold", "rmse"]
lr_df = lr_df.rename(columns={"rmse": "rmse_linear"})

# LightGBMの結果をDataFrameに
lgb_df = pd.DataFrame(lgb_results)        # columns: ["fold", "rmse"]
lgb_df = lgb_df.rename(columns={"rmse": "rmse_lgbm"})

rmse_df = lr_df.merge(lgb_df, on="fold")

rmse_df["rmse_diff"] = rmse_df["rmse_linear"] - rmse_df["rmse_lgbm"]

rmse_df


# %%
# 2つ目のパラメータセット（樹木数だけ増やす例）
lgb_params_v2 = {
    "objective": "regression",
    "metric": "rmse",
    "random_state": 42,
    "n_estimators": 500,   # ← 200 → 500 に増やす
    "learning_rate": 0.1,
    "num_leaves": 31,
    "n_jobs": -1,
}

lgb_results_v2 = []

for i, (train_mask, test_mask) in enumerate(splits, start=1):
    X_train = X.loc[train_mask]
    y_train = y[train_mask]
    X_test  = X.loc[test_mask]
    y_test  = y[test_mask]

    model = lgb.LGBMRegressor(**lgb_params_v2)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    lgb_results_v2.append({"fold": i, "rmse": rmse})
    print(f"[v2] Fold {i} RMSE: {rmse:.4f}")


# %%
# v2 の結果をDataFrame化
lgb_df_v2 = pd.DataFrame(lgb_results_v2).rename(columns={"rmse": "rmse_lgbm_v2"})

# 既存の rmse_df にマージ
rmse_df = rmse_df.merge(lgb_df_v2, on="fold")

rmse_df


# %%
# num_leaves を増やしたバージョン
lgb_params_v3 = {
    "objective": "regression",
    "metric": "rmse",
    "random_state": 42,
    "n_estimators": 200,   # 本数は良かった200のまま
    "learning_rate": 0.1,
    "num_leaves": 63,      # ここだけ 31 → 63
    "n_jobs": -1,
}

lgb_results_v3 = []

for i, (train_mask, test_mask) in enumerate(splits, start=1):
    X_train = X.loc[train_mask]
    y_train = y[train_mask]
    X_test  = X.loc[test_mask]
    y_test  = y[test_mask]

    model = lgb.LGBMRegressor(**lgb_params_v3)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    lgb_results_v3.append({"fold": i, "rmse": rmse})
    print(f"[v3] Fold {i} RMSE: {rmse:.4f}")


# %%
lgb_df_v3 = pd.DataFrame(lgb_results_v3).rename(columns={"rmse": "rmse_lgbm_v3"})

rmse_df = rmse_df.merge(lgb_df_v3, on="fold")

rmse_df


# %% [markdown]
# ### モデル比較と時系列CVの設定
#
# 2013–2017年の学習データを用いて、「過去の全期間で学習し、次の1年をテストする」年ベースの3foldクロスバリデーションを行いました。各foldのテスト期間はそれぞれ 2015年, 2016年, 2017年 です。
#
# | fold | model                | RMSE  |
# |------|----------------------|-------|
# | 1    | Linear Regression    | 278.6 |
# |      | LightGBM (best)      | 204.0 |
# | 2    | Linear Regression    | 424.1 |
# |      | LightGBM (best)      | 370.2 |
# | 3    | Linear Regression    | 341.7 |
# |      | LightGBM (best)      | 219.5 |
#
# LightGBM は、カレンダー特徴量と売上のラグ・移動平均のみで、線形回帰に対して各foldで RMSE を約50〜120程度改善しました。  
# また `n_estimators=200, num_leaves=31` のベース設定から、木の本数や葉の数を増やしたバージョン（500本や num_leaves=63）も検証しましたが、本タスクではいずれも汎化性能が悪化したため、よりシンプルなパラメータ設定を最終モデルとしています。

# %% [markdown]
# ### 使用した特徴量
#
# 本モデルでは、Kaggle Store Salesデータのうち、以下の特徴量を用いています。
#
# - 基本特徴量  
#   - `store_nbr` : 店舗ID  
#   - `family` : 商品ファミリー  
#   - `onpromotion` : プロモーション中商品数
#
# - カレンダー特徴量  
#   - `year` : 年（2013〜2017）  
#   - `month` : 月（1〜12）  
#   - `day` : 日（1〜31）  
#   - `dow` : 曜日（0=月曜, …, 6=日曜）  
#   - `weekofyear` : 年内の週番号  
#   - `is_weekend` : 週末フラグ（土日=1, それ以外=0）  
#   - `is_month_start` : 月初フラグ  
#   - `is_month_end` : 月末フラグ  
#   - `is_quarter_end` : 四半期末フラグ  
#   - `is_year_end` : 年末フラグ
#
# - 時系列ラグ特徴量（店舗×ファミリーごとに算出）  
#   - `lag_1` : 1日前の売上  
#   - `lag_7` : 7日前の売上  
#   - `lag_14` : 14日前の売上
#
# - 時系列移動平均特徴量（店舗×ファミリーごとに算出）  
#   - `rolling_mean_7` : 過去7日間の平均売上（当日を除く）  
#   - `rolling_mean_14` : 過去14日間の平均売上（当日を除く）  
#   - `rolling_mean_28` : 過去28日間の平均売上（当日を除く）

# %% [markdown]
# ### LightGBM と時系列クロスバリデーション
#
# 本プロジェクトでは、まず `demand_02_baseline.ipynb` で代表系列（Store 1, GROCERY I）を対象に、Naive法（1日前の売上）と7日移動平均によるシンプルなベースラインモデルを構築し、RMSE/SMAPEで精度感を把握しました。そのうえで `demand_03_lgbm_cv.ipynb` では、全店舗・全ファミリーに拡張し、カレンダー特徴量と売上のラグ・移動平均を入力とした LightGBM 回帰モデルを作成し、年ごとのexpanding window 3foldクロスバリデーションで汎化性能を評価しています。
#
# ### 時系列クロスバリデーションの設定
#
# 学習データ（2013-01-29〜2017-08-15）は、将来情報を使わないように「年ごとのexpanding window」で3分割しました。
#
# - Fold 1  
#   - train: 2013-01-29 〜 2014-12-31  
#   - test : 2015-01-01 〜 2015-12-31
#
# - Fold 2  
#   - train: 2013-01-29 〜 2015-12-31  
#   - test : 2016-01-01 〜 2016-12-31
#
# - Fold 3  
#   - train: 2013-01-29 〜 2016-12-31  
#   - test : 2017-01-01 〜 2017-08-15
#
# テキストイメージ：
#
# - Fold1: **[train] 2013–2014** → **[test] 2015**  
# - Fold2: **[train] 2013–2015** → **[test] 2016**  
# - Fold3: **[train] 2013–2016** → **[test] 2017(〜8/15)**
#
# この設定により、常に「過去データで学習し、翌年以降のデータで評価する」形になっており、売上予測の実運用に近い条件で汎化性能を評価しています。

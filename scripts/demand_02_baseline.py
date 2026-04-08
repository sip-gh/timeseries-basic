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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 代表系列の条件
target_store = 1
target_family = "GROCERY I"

# 学習データ読み込み
train_path = "../data/store-sales-time-series-forecasting/train.csv"
df = pd.read_csv(train_path, parse_dates=["date"])

# 代表系列だけ抽出（EDAと同じ条件）
df_rep = df[(df["store_nbr"] == target_store) & (df["family"] == target_family)].copy()

# 日付順にソートしておく（時系列なので必須）
df_rep = df_rep.sort_values("date").reset_index(drop=True)

df_rep.head()


# %%
# Naive法: 1日前の売上をそのまま予測に使う
# shift(1): 1行上の値（1日前）を参照するメソッド
df_rep["pred_naive_1d"] = df_rep["sales"].shift(1)

# 7日移動平均ベースライン
# rolling(7): 直近7行の窓をとるメソッド、mean(): その平均
df_rep["pred_ma_7d"] = df_rep["sales"].rolling(7).mean()

# 予測に使えない最初の方の行（NaNを含む行）を落とす
df_rep_baseline = df_rep.dropna(subset=["pred_naive_1d", "pred_ma_7d"]).copy()

df_rep_baseline[["date", "sales", "pred_naive_1d", "pred_ma_7d"]].head()


# %%
# RMSE: 二乗誤差の平均の平方根
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


# SMAPE: 実測と予測の大きさの合計に対する「対称な」割合誤差
# 分母が0近くになるのを防ぐために、非常に小さい値(1e-8)を足している
def smape(y_true, y_pred):
    denom = (np.abs(y_true) + np.abs(y_pred)) + 1e-8
    return np.mean(2.0 * np.abs(y_pred - y_true) / denom)



# %%
y_true = df_rep_baseline["sales"].values

rmse_naive = rmse(y_true, df_rep_baseline["pred_naive_1d"].values)
rmse_ma7 = rmse(y_true, df_rep_baseline["pred_ma_7d"].values)

smape_naive = smape(y_true, df_rep_baseline["pred_naive_1d"].values)
smape_ma7 = smape(y_true, df_rep_baseline["pred_ma_7d"].values)

rmse_naive, rmse_ma7, smape_naive, smape_ma7


# %%
results = pd.DataFrame(
    {
        "model": ["naive_1d", "ma_7d"],
        "rmse": [rmse_naive, rmse_ma7],
        "smape": [smape_naive, smape_ma7],
    }
)

results


# %%
# 直近90日分だけを抽出
df_plot = df_rep_baseline.tail(90).copy()

plt.figure(figsize=(12, 4))

# 実測
plt.plot(df_plot["date"], df_plot["sales"], label="actual", marker="o", linewidth=1)

# 7日移動平均ベースライン
plt.plot(df_plot["date"], df_plot["pred_ma_7d"], label="ma_7d", marker="o", linewidth=1)

plt.title(f"Store {target_store} - {target_family} (last 90 days)")
plt.xlabel("date")
plt.ylabel("sales")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# %% [markdown]
# ### ベースラインモデル（単一系列）
#
# 代表系列（Store 1, GROCERY I）に対して、Naive法と7日移動平均のベースラインを評価しました。
#
# | model     | RMSE   | SMAPE  |
# |----------|--------|--------|
# | naive_1d | 918.78 | 0.3623 |
# | ma_7d    | 671.25 | 0.2363 |
#
# Naive法は「1日前の売上=今日の予測」、7日移動平均は「直近7日間の平均売上」を予測としています。

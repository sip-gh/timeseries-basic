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
import matplotlib.pyplot as plt

# CSV読み込み
train_path = "../data/store-sales-time-series-forecasting/train.csv"
df = pd.read_csv(train_path, parse_dates=["date"])

df.head()


# %%
# 各列のユニーク数を確認
# nunique(): その列に含まれる「異なる値の個数」を返すメソッド
unique_counts = df.nunique()
print(unique_counts)

# 代表的なカテゴリの中身を確認
# unique(): その列に含まれる異なる値を配列として返すメソッド
print("store_nbr unique (first 10):", df["store_nbr"].unique()[:10])

print("family unique:", df["family"].unique())

# 日付の範囲を確認
print("date min/max:", df["date"].min(), "→", df["date"].max())


# %%
# 代表系列の条件を変数で定義しておくと、後から差し替えやすい
target_store = 1
target_family = "GROCERY I"

# 条件でフィルタ
# （store_nbr列がtarget_store、family列がtarget_familyに一致する行だけ残す）
df_rep = df[(df["store_nbr"] == target_store) & (df["family"] == target_family)].copy()

# 日付でソート（念のため）
df_rep = df_rep.sort_values("date")

df_rep.head()


# %%
df_rep.shape


# %%
plt.figure(figsize=(12, 4))
plt.plot(df_rep["date"], df_rep["sales"])
plt.title(f"Store {target_store} - {target_family}")
plt.xlabel("date")
plt.ylabel("sales")
plt.grid(True)
plt.show()


# %%
# 曜日・月の情報を追加
# dt: datetime列にアクセスするアクセサ
df_rep["dow"] = df_rep["date"].dt.dayofweek  # 0=月曜, 6=日曜
df_rep["month"] = df_rep["date"].dt.month    # 1〜12月

# 曜日ごとの平均売上
df_rep.groupby("dow")["sales"].mean()


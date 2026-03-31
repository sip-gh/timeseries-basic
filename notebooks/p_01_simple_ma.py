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
import matplotlib.pyplot as plt


# %%
np.random.seed(42)

n_periods = 200
date_index = pd.date_range(start="2020-01-01", periods=n_periods, freq="D")

x = np.linspace(0, 2 * np.pi, n_periods)
signal = np.sin(x)

noise = 0.3 * np.random.randn(n_periods)
values = signal + noise

df = pd.DataFrame({"value": values}, index=date_index)

df.head()


# %%
df["ma_3"] = df["value"].rolling(window=3).mean()
df["ma_7"] = df["value"].rolling(window=7).mean()
df["ma_14"] = df["value"].rolling(window=14).mean()

df.head(15)


# %%
plt.figure(figsize=(12, 5))

plt.plot(df.index, df["value"], label="observed", color="lightgray")
plt.plot(df.index, df["ma_3"], label="ma_3", linewidth=1.5)
plt.plot(df.index, df["ma_7"], label="ma_7", linewidth=1.5)
plt.plot(df.index, df["ma_14"], label="ma_14", linewidth=1.5)

plt.title("Moving Averages (3, 7, 14)")
plt.xlabel("date")
plt.ylabel("value")
plt.legend()
plt.tight_layout()
plt.show()


# %%
df["pred_ma_3"] = df["ma_3"].shift(1)
df["pred_ma_7"] = df["ma_7"].shift(1)
df["pred_ma_14"] = df["ma_14"].shift(1)

df[["value", "ma_3", "pred_ma_3", "ma_7", "pred_ma_7"]].head(15)


# %%
plot_start = 20

plt.figure(figsize=(12, 5))
plt.plot(df.index[plot_start:], df["value"].iloc[plot_start:], label="observed", color="black")

plt.plot(df.index[plot_start:], df["pred_ma_3"].iloc[plot_start:], label="MA 3 (1-step ahead)", alpha=0.8)

plt.plot(df.index[plot_start:], df["pred_ma_7"].iloc[plot_start:], label="MA 7 (1-step ahead)", alpha=0.8)

plt.plot(df.index[plot_start:], df["pred_ma_14"].iloc[plot_start:], label="MA 14 (1-step ahead)", alpha=0.8)

plt.title("One-step-ahead Forecasts by Moving Averages")
plt.xlabel("date")
plt.ylabel("value")
plt.legend()
plt.tight_layout()
plt.show()


# %%
eval_df = df[["value", "pred_ma_3", "pred_ma_7", "pred_ma_14"]].dropna()
eval_df


# %%
def mae(actual, predicted):
    return np.mean(np.abs(actual - predicted))

def rmse(actual, predicted):
    return np.sqrt(np.mean((actual - predicted) ** 2))

for col, label in [("pred_ma_3", "MA 3"), ("pred_ma_7", "MA 7"), ("pred_ma_14", "MA 14")]:
    m = mae(eval_df["value"], eval_df[col])
    r = rmse(eval_df["value"], eval_df[col])
    print(f"{label:8s} | MAE {m:.4f} RMSE: {r:.4f}")


# %%

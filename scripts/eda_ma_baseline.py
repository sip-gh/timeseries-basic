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

# %% [markdown]
# ### p01: Moving Average baseline metrics
#
# このノートでは、`data/ma_baseline_metrics.csv` から
# SIMULATED データの MAE / RMSE を集計し、window ごとの誤差を確認する。

# %%
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt


project_root = Path.cwd().parent  # notebooks_local/ から見て1つ上がルート
csv_path = project_root / "data" / "ma_baseline_metrics.csv"

metrics_df = pd.read_csv(csv_path)
metrics_df.head()

# 将来、他のデータも同じ CSV に入れる前提で、まずは SIMULATED 行だけに絞る
sim_df = metrics_df.query("symbol == 'SIMULATED'").copy()
sim_df.sort_values("window")


# %%
agg_df = (
    sim_df
    .groupby("window", as_index=False)[["mae", "rmse"]]
    .mean()
    .sort_values("window")
)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(agg_df["window"], agg_df["mae"], marker="o", label="MAE")
ax.plot(agg_df["window"], agg_df["rmse"], marker="o", label="RMSE")

ax.set_xlabel("window")
ax.set_ylabel("error")
ax.set_title("MA baseline metrics (SIMULATED)")
ax.legend()
plt.tight_layout()
plt.show()

agg_df


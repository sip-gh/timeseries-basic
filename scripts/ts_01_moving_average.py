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
from pathlib import Path
from datetime import datetime



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
from pathlib import Path
from datetime import datetime

# Notebook / スクリプト両対応の project_root を決める
try:
    # スクリプトとして実行されたとき: __file__ が定義されている
    # p_01_simple_ma.py は scripts/ 配下なので、1つ親がプロジェクトルート
    project_root = Path(__file__).resolve().parents[1]
except NameError:
    # Notebook から実行されたとき: __file__ は無いので NameError
    # scripts/ をカレントにしている想定なので、その1つ上をルートとみなす
    project_root = Path.cwd().parent

# 評価結果からメトリクス用の DataFrame を作成
rows = []
n_samples = len(eval_df)  # 評価に使ったサンプル数
run_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # 実行時刻（ログ用）

for col, label, window in [
    ("pred_ma_3", "MA 3", 3),
    ("pred_ma_7", "MA 7", 7),
    ("pred_ma_14", "MA 14", 14),
]:
    m = mae(eval_df["value"], eval_df[col])
    r = rmse(eval_df["value"], eval_df[col])
    rows.append(
        {
            # 今は疑似データなので SIMULATED。将来 他のデータ などに置き換えやすいように列を用意
            "symbol": "SIMULATED",
            "target_col": "value",  # どのカラムを予測しているか（目的変数名）
            "window": window,       # 移動平均の窓長
            "mae": m,
            "rmse": r,
            "n_samples": n_samples,
            "run_at": run_at,
        }
    )

metrics_df = pd.DataFrame(rows)

# プロジェクトルート直下の data/ ディレクトリを指す Path を組み立て
data_dir = project_root / "data"

# data/ が無ければ作る。すでにあれば何もしない
data_dir.mkdir(parents=True, exist_ok=True)

# 書き出し先 CSV ファイル
csv_path = data_dir / "ma_baseline_metrics.csv"

# 既存ファイルがあれば追記（append）、なければ新規作成
if csv_path.exists():
    metrics_df.to_csv(
        csv_path,
        mode="a",        # 追記モード（append）
        header=False,    # すでにヘッダ行があるので、追加時は列名を書かない
        index=False,     # インデックス列は不要なので出力しない
    )
else:
    metrics_df.to_csv(
        csv_path,
        mode="w",        # 新規作成（明示的に上書きモード）
        header=True,     # 1回目は列名をヘッダとして書く
        index=False,     # 同上、インデックスは不要
    )

print(f"saved metrics to: {csv_path}")
# Notebook のときだけリッチ表示したいので、display があれば使う
try:
    display(metrics_df)
except NameError:
    # スクリプト実行時（display がない環境）は普通に print だけ
    print(metrics_df)


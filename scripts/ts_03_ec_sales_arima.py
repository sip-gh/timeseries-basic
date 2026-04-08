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
# ##### p_03_ec_sales_arima: EC日次売上の時系列予測（ARIMA → SARIMA）
#
# ###### 目的
# - 擬似的なEC小売の日次売上データを使い、ARIMA モデルによる7日先予測を実装する
# - p_02（サイン波シミュレーション）で習得した定常性検定→次数選択→残差診断のフローを、
#   より実務に近い「ビジネス系時系列」に移植する
# - 後で Kaggle などの実データに差し替えて再利用できるテンプレとして仕上げる
#
# ###### データ概要
# - 期間：2022-01-01 〜 2023-12-31（730日分）
# - 構造：ベーストレンド＋週次季節性（週末+30）＋年間季節性（12月+40）＋正規ノイズ（σ=10）
#
# ###### 結論（先に読む用）
# - ADF 検定：元系列は非定常 → 1階差分で定常化（d=1）
# - ACF/PACF から候補を絞り、AIC/BIC 比較で ARIMA(1,1,2) を採用
# - 7日先予測区間幅は約 70〜85（残差 σ≈18 の水準として妥当）
# - 残差 ACF にラグ7周期のスパイクあり → 週次季節性が残っていることを確認
# - SARIMA(1,1,2)(1,0,1,7) に拡張すると、ラグ7周期のスパイクが消え、AIC/BIC も大きく改善

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

# 日本語フォントを指定（グラフの日本語が文字化けしないように）
matplotlib.rcParams["font.family"] = "Hiragino Sans"  # Mac標準フォント


# %%
# 日次売上データを「約2年分」作る想定
n_days = 730  # 365日 × 2年 くらい

# 2022-01-01 から n_days 日ぶんの DatetimeIndex を作る
date_index = pd.date_range(
    start="2022-01-01",  # 開始日
    periods=n_days,      # 生成する日数
    freq="D",            # 日次（Daily）
)

date_index[:5]


# %%
# 乱数生成器（random number generator）を作る
# RNG = Random Number Generator の略
# default_rng(seed=42) で「毎回同じ乱数列」が出るように固定している
# こうしておくと、ノートを再実行しても同じ擬似データが再現できる
rng = np.random.default_rng(seed=42)

# ベーストレンド
# - 初期売上を 100
# - 1日あたり 0.05 ずつ増えていく（2年で約 100 + 0.05*730 ≒ 136）
#   → 「売上が少しずつ成長しているECサイト」のイメージ
trend = 100 + 0.05 * np.arange(n_days)

trend[:5]


# %%
# 曜日情報（0=月曜, 1=火曜, ..., 6=日曜）
dow = date_index.dayofweek

# 週次季節性：
# - 平日 (dow 0〜4)   : +0
# - 土日 (dow 5, 6) : +30
# 30 という数字は「週末は平日の +30 件くらい売上が乗る」イメージの仮定
weekly_seasonality = np.where(dow >= 5, 30, 0)

weekly_seasonality[:7], dow[:7]


# %%
# 月情報（1〜12）
month = date_index.month

# 年間季節性：
# - 12月だけ +40 （ボーナス月で売上がぐっと上がる、というイメージ）
# - それ以外の月は +0
# 40 もまた、「どれくらい12月が強いか」という仮のパラメータ
yearly_seasonality = np.where(month == 12, 40, 0)

yearly_seasonality[:31]


# %%
# ノイズ（ランダムなブレ）
# - 平均 0
# - 標準偏差 10
# → 大まかには「±20 くらいは普通にブレる」イメージ
noise = rng.normal(
    loc=0.0,   # 平均 (mean)
    scale=10.0,  # 標準偏差 (standard deviation)
    size=n_days,
)

noise[:5]


# %%
# トレンド＋季節性＋ノイズを足し合わせる
sales = trend + weekly_seasonality + yearly_seasonality + noise

# np.clip で下限・上限をはさむ
# - a_min=0   : 最小値（0未満は 0 にする）  → 売上がマイナスはおかしいので 0 にクリップ
# - a_max=None: 上限は指定しない（= クリップしない）
# a_min の a は argument（引数）の a と思ってOK
sales = np.clip(
    sales,        # 対象の配列
    a_min=0,      # 最小値（ここでは0）[web:120][web:139][web:142]
    a_max=None,   # 上限なし
)

sales[:10]


# %%
# pandas の DataFrame にまとめる
# index : 日付
# column: 売上 (sales)
df = pd.DataFrame(
    {"sales": sales},
    index=date_index,
)

df.head()


# %%
# 日次売上の全体推移をプロット
plt.figure(figsize=(12, 4))
plt.plot(df.index, df["sales"], color="steelblue")
plt.title("日次売上の推移（擬似データ）")
plt.xlabel("日付")
plt.ylabel("売上")
plt.tight_layout()
plt.show()


# %%
# 曜日カラムを追加（0=月曜, ..., 6=日曜）
df["dow"] = df.index.dayofweek

# 曜日ごとの平均売上を計算
dow_mean = df.groupby("dow")["sales"].mean()

plt.figure(figsize=(6, 4))
dow_mean.plot(kind="bar", color="orange")
plt.title("曜日別の平均売上")
plt.xlabel("曜日 (0=月曜, 6=日曜)")
plt.ylabel("平均売上")
plt.tight_layout()
plt.show()


# %%

# 月カラムを追加（1〜12）
df["month"] = df.index.month

# 月別の平均売上
month_mean = df.groupby("month")["sales"].mean()

plt.figure(figsize=(6, 4))
month_mean.plot(kind="bar", color="seagreen")
plt.title("月別の平均売上")
plt.xlabel("月")
plt.ylabel("平均売上")
plt.tight_layout()
plt.show()


# %%
# 元系列（売上）
series = df["sales"].copy()

# 1階差分（今日 − 昨日）でトレンドをざっくり落とす
df["diff_1"] = series.diff()

df[["sales", "diff_1"]].head()


# %%
from statsmodels.tsa.stattools import adfuller

def run_adf(series, label):
    """
    ADF検定の結果を見やすく表示するユーティリティ関数
    series: pandas Series（NaN は内部で dropna）
    label : どの系列かを表示するラベル
    """
    result = adfuller(series.dropna())  # autolag="AIC" がデフォルト

    adf_stat  = result[0]  # 検定統計量
    p_value   = result[1]  # p値
    n_lags    = result[2]  # 使用ラグ数
    n_obs     = result[3]  # 有効観測数
    crit_vals = result[4]  # 臨界値（1%, 5%, 10%）

    print(f"=== {label} ===")
    print(f"  ADF 統計量 : {adf_stat:.4f}")
    print(f"  p 値       : {p_value:.4f}")
    print(f"  使用ラグ数  : {n_lags}")
    print(f"  有効観測数  : {n_obs}")
    print("  臨界値:")
    for key, val in crit_vals.items():
        print(f"    {key}: {val:.4f}")

    # 5%水準での判定
    if adf_stat < crit_vals["5%"]:
        print("  判定       : 定常（帰無仮説を5%で棄却）")
    else:
        print("  判定       : 非定常（棄却できず）")
    print()

# 元の売上
run_adf(df["sales"], "元の売上(sales)")
# 1階差分
run_adf(df["diff_1"], "1階差分(diff_1)")


# %%
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# 図のサイズを少し大きめに
fig, axes = plt.subplots(2, 2, figsize=(14, 8))

# 上段左：元系列の ACF
plot_acf(df["sales"].dropna(), lags=40, ax=axes[0, 0])
axes[0, 0].set_title("ACF - 元の売上")

# 上段右：元系列の PACF
plot_pacf(df["sales"].dropna(), lags=40, ax=axes[0, 1])
axes[0, 1].set_title("PACF - 元の売上")

# 下段左：1階差分の ACF
plot_acf(df["diff_1"].dropna(), lags=40, ax=axes[1, 0])
axes[1, 0].set_title("ACF - 1階差分")

# 下段右：1階差分の PACF
plot_pacf(df["diff_1"].dropna(), lags=40, ax=axes[1, 1])
axes[1, 1].set_title("PACF - 1階差分")

plt.tight_layout()
plt.show()


# %%
from statsmodels.tsa.arima.model import ARIMA

candidates = [
    (0, 1, 1),
    (1, 1, 0),
    (1, 1, 1),
    (1, 1, 2),
    (0, 1, 2),
]

results = []

# tmp
order = (1, 1, 1)
model = ARIMA(df["sales"], order=order)
result = model.fit()
result.summary()


# %%
for order in candidates:
    print(f"Fitting ARIMA{order}...")
    model = ARIMA(df["sales"], order=order)
    fitted = model.fit()

    results.append(
        {
            "order": str(order),
            "AIC": round(fitted.aic, 2),
            "BIC": round(fitted.bic, 2),
        }
    )

# 結果を AIC 昇順で並べる
aic_df = pd.DataFrame(results).sort_values("AIC")
aic_df


# %%
# AIC最小だったモデル
best_order = (1, 1, 2)

# モデル定義（元の sales 系列を渡せば d=1 は内部で差分を取ってくれる）
best_model = ARIMA(df["sales"], order=best_order)

# パラメータ推定
best_result = best_model.fit()

print(best_result.summary())


# %%
# 何日先まで予測するか
n_forecast = 7

# steps=7 で「1日先〜7日先」までの予測をまとめて取る
forecast = best_result.get_forecast(steps=n_forecast)

# 予測値（平均）
forecast_mean = forecast.predicted_mean

# 95% 予測区間（lower / upper の2列を持つ DataFrame）
forecast_ci = forecast.conf_int(alpha=0.05)

forecast_mean, forecast_ci.head()


# %%
# プロットする実データの期間（直近60日くらいを見る）
history_window = 60
history = df["sales"].iloc[-history_window:]

fig, ax = plt.subplots(figsize=(12, 5))

# 実データ（直近）
history.plot(ax=ax, label="実データ（直近）", color="steelblue")

# 予測値
forecast_mean.plot(ax=ax, label="7日先までの予測", color="red")

# 予測区間（帯）
ax.fill_between(
    forecast_ci.index,
    forecast_ci.iloc[:, 0],  # 下限
    forecast_ci.iloc[:, 1],  # 上限
    color="red",
    alpha=0.2,
    label="95%予測区間",
)

ax.set_title(f"ARIMA{best_order} による 7日先予測")
ax.set_xlabel("日付")
ax.set_ylabel("売上")
ax.legend()
plt.tight_layout()
plt.show()


# %%
# 残差（実データ - モデルの当てはまり）
resid = best_result.resid

print("残差の基本統計量:")
print(resid.describe())


# %%
from statsmodels.graphics.tsaplots import plot_acf
import scipy.stats as stats

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# 時系列
resid.plot(ax=axes[0, 0])
axes[0, 0].axhline(0, color="red", linestyle="--")
axes[0, 0].set_title("残差の時系列")

# ヒストグラム
axes[0, 1].hist(resid, bins=30, color="steelblue", alpha=0.7)
axes[0, 1].set_title("残差のヒストグラム")

# ACF
plot_acf(resid, lags=40, ax=axes[1, 0])
axes[1, 0].set_title("残差のACF")

# Q-Qプロット（正規分布との比較）
stats.probplot(resid, plot=axes[1, 1])
axes[1, 1].set_title("残差のQ-Qプロット")

plt.tight_layout()
plt.show()


# %%
steps_list = [1, 7, 30]
rows = []

for n in steps_list:
    # get_forecast(steps=n) で「1〜n日先」の予測をまとめて取得する
    # 戻り値は PredictionResults オブジェクト
    fc = best_result.get_forecast(steps=n)

    # conf_int(alpha=0.05) で 95% 予測区間を DataFrame で返す
    # 列は ["lower sales", "upper sales"] の2列
    ci = fc.conf_int(alpha=0.05)

    # iloc[-1] で「n日先（最後のステップ）」の行だけ取り出す
    last_ci = ci.iloc[-1]

    # 上限 - 下限 = 区間幅
    width = last_ci.iloc[1] - last_ci.iloc[0]

    # predicted_mean.iloc[-1] で「n日先の点予測値」を取得
    point = fc.predicted_mean.iloc[-1]

    rows.append({
        "steps先": n,
        "点予測": round(point, 1),
        "下限(95%)": round(last_ci.iloc[0], 1),
        "上限(95%)": round(last_ci.iloc[1], 1),
        "区間幅": round(width, 1),
    })

ci_df = pd.DataFrame(rows)
ci_df


# %%
# sigma = 残差の標準偏差（1ステップのノイズ感）
sigma = resid.std()
print(f"残差の標準偏差 σ = {sigma:.2f}")

# 理論上は n ステップ先の幅 ≒ 1.96 * σ * √n * 2（上下合計）
# 実際の ARIMA は MA/AR 項の影響で少しずれるが、オーダー感の確認に使う
print("\n理論値との比較:")
for n in steps_list:
    theory_width = 2 * 1.96 * sigma * (n ** 0.5)
    actual_width = ci_df.loc[ci_df["steps先"] == n, "区間幅"].values[0]
    print(f"  {n:2d}日先：理論値 {theory_width:.1f}  実際 {actual_width:.1f}")


# %%
# SARIMA は ARIMA に「季節成分」を追加したモデル
# order=(p,d,q) が通常成分、seasonal_order=(P,D,Q,m) が季節成分
# m=7 は「7日周期（週次）」を指定している
from statsmodels.tsa.statespace.sarimax import SARIMAX

sarima_model = SARIMAX(
    df["sales"],
    order=(1, 1, 2),          # 通常の ARIMA 部分（p,d,q）
    seasonal_order=(1, 0, 1, 7),  # 季節成分（P,D,Q,m）
    # P=1：季節 AR 1次（7日前の値が効く）
    # D=0：季節差分なし（元系列の季節成分は穏やか）
    # Q=1：季節 MA 1次
    # m=7：季節周期 = 7日
)
sarima_result = sarima_model.fit(disp=False)  # disp=False で最適化ログを非表示

print(sarima_result.summary())


# %%
resid_sarima = sarima_result.resid

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

# ARIMA(1,1,2) の残差 ACF（左）
plot_acf(resid.dropna(), lags=40, ax=axes[0])
axes[0].set_title("残差 ACF：ARIMA(1,1,2)")

# SARIMA(1,1,2)(1,0,1,7) の残差 ACF（右）
plot_acf(resid_sarima.dropna(), lags=40, ax=axes[1])
axes[1].set_title("残差 ACF：SARIMA(1,1,2)(1,0,1,7)")

plt.tight_layout()
plt.show()


# %%
print("モデル比較:")
print(f"  ARIMA(1,1,2)          AIC={best_result.aic:.2f}  BIC={best_result.bic:.2f}")
print(f"  SARIMA(1,1,2)(1,0,1,7) AIC={sarima_result.aic:.2f}  BIC={sarima_result.bic:.2f}")


# %%

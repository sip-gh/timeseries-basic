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
# ### p_02_stationarity: ARIMA に向けた定常性と次数推定
#
# - シミュレーションデータで ADF / ACF / PACF を確認
# - ARIMA(0,1,1) をフィットし、残差診断とウォークフォワード検証まで行う
# - 後で実データに差し替えて再利用する前提

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Hiragino Sans'  # Mac標準フォント
from statsmodels.tsa.stattools import adfuller


# %%
n_periods = 200
start_date="2020-01-01"
date_index = pd.date_range(start=start_date, periods=n_periods, freq="D")

x = np.linspace(0, 2 * np.pi, n_periods)
signal = np.sin(x)
noise = 0.3 * np.random.randn(n_periods)
values = signal + noise

df = pd.DataFrame({"value": values}, index=date_index)
df.head()


# %%
fig, axes = plt.subplots(2, 1, figsize=(12, 6))

axes[0].plot(df.index, df["value"], color="steelblue")
axes[0].set_title("Original (non-stationary)")
axes[0].set_ylabel("value")

df["diff_1"] = df["value"].diff()
axes[1].plot(df.index, df["diff_1"], color="tomato")
axes[1].set_title("1st Difference")
axes[1].set_ylabel("diff value")

plt.tight_layout()


# %%
# セル4: ADF 検定（拡張版）

def run_adf(series, label):
    result = adfuller(series.dropna())  # autolag="AIC" がデフォルト

    adf_stat   = result[0]  # ADF 統計量
    p_value    = result[1]  # p 値
    n_lags     = result[2]  # AIC で自動選択されたラグ数
    n_obs      = result[3]  # 有効観測数
    crit_vals  = result[4]  # 臨界値（1%, 5%, 10%）
    aic        = result[5]  # ラグ選択に使った AIC

    print(f"=== {label} ===")
    print(f"  ADF 統計量     : {adf_stat:.4f}")
    print(f"  p 値           : {p_value:.4f}")
    print(f"  使用ラグ数      : {n_lags}  （AIC で自動選択）")
    print(f"  有効観測数      : {n_obs}")
    print(f"  AIC            : {aic:.4f}")
    print(f"  臨界値  1%     : {crit_vals['1%']:.4f}")
    print(f"  臨界値  5%     : {crit_vals['5%']:.4f}")
    print(f"  臨界値 10%     : {crit_vals['10%']:.4f}")

    # ADF統計量が臨界値より小さければ（より負なら）棄却できる
    reject_5pct = adf_stat < crit_vals["5%"]
    conclusion  = "定常（5%水準で棄却）" if reject_5pct else "非定常（棄却できず）"
    print(f"  判定           : {conclusion}")
    print()

run_adf(df["value"], "元データ")
run_adf(df["diff_1"], "1階差分")


# %%
# Dickey-Fuller 分布 vs t 分布の可視化
# （シミュレーションで近似的に再現する）

from scipy import stats

fig, ax = plt.subplots(figsize=(10, 4))

x = np.linspace(-6, 4, 500)

# 通常の t 分布（自由度 100）
ax.plot(x, stats.t.pdf(x, df=100), label="t-distribution (df=100)", linestyle="--")

# DF 分布の近似（正確な分布は複雑なので、既知の臨界値を目印として示す）
# ADF の 5% 臨界値はおよそ -2.86（定数項ありモデルの場合）
ax.axvline(x=-2.86, color="red", linestyle=":", label="ADF 5% critical value ≈ -2.86")
ax.axvline(x=-1.96, color="gray", linestyle=":", label="t 5% critical value ≈ -1.96")

ax.set_title("t-distribution vs ADF critical region")
ax.set_xlabel("test statistic")
ax.legend()
plt.tight_layout()
plt.show()


# %%
# セル5: ACF / PACF で自己相関の構造を視覚化する

# statsmodels の ACF / PACF プロット関数を読み込む
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# 2行 × 2列のサブプロットを作成（上段: 元データ、下段: 1階差分）
fig, axes = plt.subplots(2, 2, figsize=(14, 8))

# --- 上段左: 元データの ACF ---
# ACF（自己相関関数）: 時点 t と t-k の相関を k=0〜40 まで表示
# 非定常だとラグが増えてもゆっくりしか相関が落ちない（棒が長く続く）
plot_acf(df["value"].dropna(), lags=40, ax=axes[0, 0], title="ACF - 元データ")

# --- 上段右: 元データの PACF ---
# PACF（偏自己相関関数）: 間にあるラグの影響を除いた「直接の相関」だけを表示
# AR モデルの次数 p のヒントになる（どこで急に落ちるか）
plot_pacf(df["value"].dropna(), lags=40, ax=axes[0, 1], title="PACF - 元データ")

# --- 下段左: 1階差分の ACF ---
# 差分をとって定常化したあと、残っている自己相関を確認する
# MA モデルの次数 q のヒントになる（どのラグまで有意か）
plot_acf(df["diff_1"].dropna(), lags=40, ax=axes[1, 0], title="ACF - 1階差分")

# --- 下段右: 1階差分の PACF ---
# 差分系列での直接の自己相関を確認する
# AR 部分の次数 p を改めて確認するために使う
plot_pacf(df["diff_1"].dropna(), lags=40, ax=axes[1, 1], title="PACF - 1階差分")

plt.tight_layout()
plt.show()


# %%
# 信頼区間の値を数値で確認する
import statsmodels.api as sm
import numpy as np

acf_values, confint = sm.tsa.acf(df["value"].dropna(), nlags=40, alpha=0.05)

# confint は各ラグの [下限, 上限] の配列
print(confint[:5])  # ラグ0〜4の信頼区間を表示


# %%
from statsmodels.tsa.arima.model import ARIMA

model = ARIMA(df["value"], order=(1, 1, 1))
result = model.fit()

print(result.summary())


# %%
candidates = [(0,1,1), (1,1,1), (1,1,0), (2,1,1), (1,1,2)]
results = []
for order in candidates:
    m = ARIMA(df["value"], order=order).fit()
    results.append({"order": str(order), "AIC": round(m.aic, 3), "BIC": round(m.bic, 3)})

    df_aic = pd.DataFrame(results).sort_values("AIC")
    print(df_aic)


# %%
# セル7: ARIMA(0,1,1) で予測する

best_model = ARIMA(df["value"], order=(0, 1, 1))
best_result = best_model.fit()

# 30ステップ先を予測
forecast = best_result.get_forecast(steps=30)
forecast_mean = forecast.predicted_mean          # 予測値
forecast_ci   = forecast.conf_int(alpha=0.05)   # 95%予測区間

print(forecast_mean.head())
print(forecast_ci.head())


# %%
fig, ax = plt.subplots(figsize=(12, 5))
df["value"].iloc[-60:].plot(ax=ax, label="実データ")
forecast_mean.plot(ax=ax, label="予測", color="red")

# fill_between: 2つのY値の間を塗りつぶす関数
# 予測区間（上限〜下限の帯）をグラフに重ねて表示するために使う
ax.fill_between(
    forecast_ci.index,          # X軸: 予測日付のインデックス
    forecast_ci["lower value"], # 下側の境界線: 95%区間の下限
    forecast_ci["upper value"], # 上側の境界線: 95%区間の上限
    alpha=0.2,                  # 透明度（0=完全透明, 1=不透明）。0.2で薄く重ねる
    color="red",                # 塗りつぶし色
    label="95%予測区間"          # 凡例に表示するラベル
)

ax.legend()
ax.set_title("ARIMA(0,1,1) 予測")


# %%
# セル9: 残差を取り出して分布・自己相関を確認する

# モデルの残差 = 実際の値 - モデルの予測値
# 良いモデルなら残差は「ホワイトノイズ」になるはず
# ホワイトノイズ = 平均ゼロ・自己相関なし・正規分布に近い
residuals = best_result.resid

print("残差の基本統計:")
print(residuals.describe())


# %%
# セル10: 残差を4つの角度から可視化する

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# --- 左上: 残差の時系列プロット ---
# 時間方向にパターン（トレンド・周期）が残っていないか確認
# → ランダムに上下しているなら OK
residuals.plot(ax=axes[0, 0], title="残差 時系列")
axes[0, 0].axhline(0, color="red", linestyle="--")  # ゼロ基準線

# --- 右上: 残差のヒストグラム ---
# 正規分布に近い形（左右対称の山型）かを確認
residuals.plot(kind="hist", bins=30, ax=axes[0, 1], title="残差 ヒストグラム")

# --- 左下: 残差の ACF ---
# 自己相関が残っていないか確認
# → 全ラグが信頼区間内なら「残差はホワイトノイズ」= モデルが構造を全部拾えた
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(residuals, lags=40, ax=axes[1, 0], title="残差 ACF")

# --- 右下: Q-Q プロット ---
# 残差が正規分布にどれだけ近いかを確認
# → 点が対角線に乗っていれば正規分布に近い
import scipy.stats as stats
stats.probplot(residuals, plot=axes[1, 1])
axes[1, 1].set_title("残差 Q-Q プロット")

plt.tight_layout()
plt.show()


# %%
# セル11: ARIMA(0,1,1) のウォークフォワード検証（1ステップ先）

from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

series = df["value"].copy()  # どのデータでも value 列に統一しておく

start_train = 150            # ここより前を学習期間として使う（データ数に応じて調整）

y_true = []  # 実績値
y_pred = []  # 1ステップ先予測値

for t in range(start_train, len(series) - 1):
    # 0〜t まででモデルを学習（未来を見ない）
    train_data = series.iloc[:t+1]
    model = ARIMA(train_data, order=(0, 1, 1))
    result = model.fit()

    # t+1 時点の 1ステップ先予測
    forecast = result.get_forecast(steps=1)
    y_hat = forecast.predicted_mean.iloc[0]

    # 実際の値（答え合わせ用）
    y_actual = series.iloc[t+1]

    y_pred.append(y_hat)
    y_true.append(y_actual)

# 誤差指標を計算
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)                 # 古い sklearn なので自分で sqrt をとる
mae  = mean_absolute_error(y_true, y_pred)

print(f"ウォークフォワード検証サンプル数: {len(y_true)}")
print(f"MSE : {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE : {mae:.4f}")


# %%

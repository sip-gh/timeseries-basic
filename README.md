# Time Series Basics with Python

This repository contains small, self-contained examples for learning classical time series forecasting in Python (ARIMA, SARIMA, etc.).

## Project structure

- `scripts/` – Step-by-step scripts for each topic (stationarity check, simple MA, ARIMA on sales data, etc.).
- `main.py` – Simple entry point to run selected examples from the command line.
- `pyproject.toml`, `uv.lock` – Project configuration and locked dependencies managed by `uv`.
- `README.md` – Project overview and usage.

## Setup

```bash
uv sync
uv run python main.py
```

## Usage

- Read the scripts under `scripts/` in order (e.g. `p_01_...`, `p_02_...`) to follow the learning flow.
- Modify the scripts or copy them as a starting point for your own time series experiments.

---

## Store Sales Demand Forecast (Kaggle)

This project also includes a small workflow for the Kaggle competition **“Store Sales – Time Series Forecasting”**, implemented as scripts under `scripts/`.
The goal is to go from a single-series baseline to a multi-series LightGBM model with time-series cross-validation.

- `scripts/demand_01_eda.py`
  Basic EDA on the training data. Plots and inspects a representative series (Store 1, GROCERY I) to understand overall trends and seasonality.

- `scripts/demand_02_baseline.py`
  Builds baseline models (Naive – previous day, and 7-day moving average) for the same representative series.
  Evaluates RMSE and SMAPE, and visualizes the last 90 days to get an intuition for the error level.

- `scripts/demand_03_lgbm_cv.py`
  Extends the model to all stores and product families.
  Uses calendar features and lag/moving-average features of sales as inputs to a LightGBM regressor.
  Runs a 3-fold expanding-window cross-validation (year-based, 2015–2017) and compares RMSE to a simple linear regression baseline.

### Baseline model (single series)

For the representative series (Store 1, GROCERY I), we evaluate a Naive model and a 7-day moving average.
You can reproduce the following metrics with `scripts/demand_02_baseline.py`.

| model     | RMSE   | SMAPE  |
|----------|--------|--------|
| naive_1d | 918.78 | 0.3623 |
| ma_7d    | 671.25 | 0.2363 |

### Model comparison and time-series CV

Using the 2013–2017 training data, we run a year-based 3-fold expanding-window cross-validation, where each fold trains on all past data and tests on the next year (test years: 2015, 2016, 2017).

| model              | RMSE (mean over folds) |
|--------------------|------------------------|
| naive_lag1         | 507.1                  |
| Linear Regression  | 348.1                  |
| LightGBM (v1 base) | 302.2                  |

Compared to the Naive baseline (previous day’s sales), linear regression significantly reduces RMSE, and LightGBM further improves RMSE by about 13% over linear regression.
For LightGBM, a simple setting (`n_estimators=200, num_leaves=31`) works best; increasing tree count or leaf count does not noticeably improve generalization.

### Features used

We use the following features from the Kaggle Store Sales dataset:

- Basic features
  - `store_nbr` : store ID
  - `family` : product family
  - `onpromotion` : number of items on promotion

- Calendar features
  - `year` : 2013–2017
  - `month` : 1–12
  - `day` : 1–31
  - `dow` : day of week (0=Mon, …, 6=Sun)
  - `weekofyear` : week number within the year
  - `is_weekend` : weekend flag
  - `is_month_start` : month start
  - `is_month_end` : month end
  - `is_quarter_end` : quarter end
  - `is_year_end` : year end

- Lag features (per store × family)
  - `lag_1` : sales 1 day ago
  - `lag_7` : sales 7 days ago
  - `lag_14` : sales 14 days ago

- Rolling mean features (per store × family, excluding today)
  - `rolling_mean_7` : mean sales over the past 7 days
  - `rolling_mean_14` : mean sales over the past 14 days
  - `rolling_mean_28` : mean sales over the past 28 days

According to LightGBM feature importance (gain), `rolling_mean_7` and the autoregressive lags (`lag_1`, `lag_7`, `lag_14`) contribute the most, indicating that the model heavily exploits the autocorrelation structure of sales.

### LightGBM and time-series CV

We first build simple baselines (Naive and 7-day moving average) for a single series in `demand_02_baseline.py` to understand error levels.
Then `demand_03_lgbm_cv.py` extends to all store–family combinations with calendar and lag/rolling features, and evaluates generalization via year-based 3-fold expanding-window cross-validation.

### Time-series CV configuration

The training data (2013-01-29–2017-08-15) is split into three folds using a year-based expanding window to avoid look-ahead bias.

- Fold 1
  - train: 2013-01-29 – 2014-12-31
  - test : 2015-01-01 – 2015-12-31

- Fold 2
  - train: 2013-01-29 – 2015-12-31
  - test : 2016-01-01 – 2016-12-31

- Fold 3
  - train: 2013-01-29 – 2016-12-31
  - test : 2017-01-01 – 2017-08-15

This setup always trains on past data and evaluates on future periods, closely matching a realistic deployment scenario for sales forecasting.

Additionally, we computed a SHAP value–based feature importance bar plot once to cross-check the LightGBM gain-based importance.

---

# Python時系列分析の基本

このリポジトリは、ARIMA・SARIMAなどの古典的な時系列予測モデルをPythonで学ぶための、小さなサンプル集です。

## 構成

- `scripts/` … 各トピックごとのスクリプト（定常性チェック、単純移動平均、売上データへのARIMAなど）
- `main.py` … コマンドラインから一部サンプルを実行するためのエントリーポイント
- `pyproject.toml`, `uv.lock` … `uv` による依存関係管理とプロジェクト設定
- `README.md` … プロジェクト概要と使い方

## セットアップ

```bash
uv sync
uv run python main.py
```

## 使い方

- `scripts/` 以下のスクリプト（`p_01_...`, `p_02_...` など）を順番に読み進めて、時系列分析の流れを確認してください。
- スクリプトを自分用にコピー・改造して、独自の時系列実験の叩き台として使えます。

---

## Store Sales需要予測（Kaggle）

Kaggleコンペ **「Store Sales - Time Series Forecasting」** の学習データを用いて、
単一系列のベースラインから、多系列LightGBMモデル＋時系列クロスバリデーションまでを `scripts/` 以下のスクリプトとして整理しています。

- `scripts/demand_01_eda.py`
  学習データの基本的なEDA。代表系列（Store 1, GROCERY I）の時系列プロットなどを通して、トレンドや季節性の有無を確認します。

- `scripts/demand_02_baseline.py`
  同じ代表系列を対象に、Naive法（1日前の売上＝今日の予測）と7日移動平均のベースラインモデルを構築します。
  RMSE / SMAPE を算出し、直近90日の予測と実績のプロットで精度感を評価します。

- `scripts/demand_03_lgbm_cv.py`
  対象を全店舗・全ファミリーに拡張し、カレンダー特徴量と売上のラグ・移動平均を特徴量とした LightGBM 回帰モデルを構築します。
  2015〜2017年をテスト対象とした年ベースのexpanding window 3-foldクロスバリデーションを行い、線形回帰とのRMSE比較を実施します。

### ベースラインモデル（単一系列）

代表系列（Store 1, GROCERY I）に対して、Naive法と7日移動平均のベースラインを評価しました。
以下の結果は `scripts/demand_02_baseline.py` で再現可能です。

| model     | RMSE   | SMAPE  |
|----------|--------|--------|
| naive_1d | 918.78 | 0.3623 |
| ma_7d    | 671.25 | 0.2363 |

### モデル比較と時系列CVの設定

2013〜2017年の学習データを用いて、「過去の全期間で学習し、次の1年をテストする」年ベースの3-foldクロスバリデーションを行いました（テスト年は 2015, 2016, 2017）。

| model              | RMSE (mean over folds) |
|--------------------|------------------------|
| naive_lag1         | 507.1                  |
| Linear Regression  | 348.1                  |
| LightGBM (v1 base) | 302.2                  |

Naive（1日前の売上そのまま）に比べて、Linear Regression で RMSE を大きく改善し、さらに LightGBM で Linear に対してもおよそ 13％程度 RMSE を改善できました。
LightGBM については、`n_estimators=200, num_leaves=31` のシンプルなベース設定が最も良く、木の本数や葉の数を増やしても汎化性能はほとんど向上しませんでした。

### 使用した特徴量

本モデルでは、Kaggle Store Sales データのうち、以下の特徴量を使用しています。

- 基本特徴量
  - `store_nbr` : 店舗ID
  - `family` : 商品ファミリー
  - `onpromotion` : プロモーション中商品数

- カレンダー特徴量
  - `year` : 年（2013〜2017）
  - `month` : 月（1〜12）
  - `day` : 日（1〜31）
  - `dow` : 曜日（0=月曜, …, 6=日曜）
  - `weekofyear` : 年内の週番号
  - `is_weekend` : 週末フラグ
  - `is_month_start` : 月初フラグ
  - `is_month_end` : 月末フラグ
  - `is_quarter_end` : 四半期末フラグ
  - `is_year_end` : 年末フラグ

- 時系列ラグ特徴量（店舗×ファミリーごとに算出）
  - `lag_1` : 1日前の売上
  - `lag_7` : 7日前の売上
  - `lag_14` : 14日前の売上

- 時系列移動平均特徴量（店舗×ファミリーごとに算出、当日を除く）
  - `rolling_mean_7` : 過去7日間の平均売上
  - `rolling_mean_14` : 過去14日間の平均売上
  - `rolling_mean_28` : 過去28日間の平均売上

LightGBM の Feature Importance（gain）からは、`rolling_mean_7` と `lag_1`, `lag_7`, `lag_14` といった自己回帰系の特徴量の寄与が特に大きく、売上の自己相関構造をモデルが強く利用していることが分かりました。

### LightGBM と時系列クロスバリデーション

本プロジェクトでは、まず `demand_02_baseline.py` で代表系列（Store 1, GROCERY I）を対象に、Naive法（1日前の売上）と7日移動平均によるシンプルなベースラインモデルを構築し、RMSE/SMAPEで精度感を把握しました。
そのうえで `demand_03_lgbm_cv.py` では、全店舗・全ファミリーに拡張し、カレンダー特徴量と売上のラグ・移動平均を入力とした LightGBM 回帰モデルを作成し、年ごとのexpanding window 3-foldクロスバリデーションで汎化性能を評価しています。

### 時系列クロスバリデーションの設定

学習データ（2013-01-29〜2017-08-15）は、将来情報を使わないように「年ごとのexpanding window」で3分割しました。

- Fold 1
  - train: 2013-01-29 〜 2014-12-31
  - test : 2015-01-01 〜 2015-12-31

- Fold 2
  - train: 2013-01-29 〜 2015-12-31
  - test : 2016-01-01 〜 2016-12-31

- Fold 3
  - train: 2013-01-29 〜 2016-12-31
  - test : 2017-01-01 〜 2017-08-15

この設定により、常に「過去データで学習し、翌年以降のデータで評価する」形になっており、売上予測の実運用に近い条件で汎化性能を評価できています。

また、LightGBM の gain ベースの特徴量重要度を補足的に確認するため、SHAP 値に基づく特徴量重要度の棒グラフも 1 回だけ作成しました。

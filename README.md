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

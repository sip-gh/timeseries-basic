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

本プロジェクトでは、Kaggle「Store Sales - Time Series Forecasting」の学習データを用いて、売上予測のベースライン構築から多系列LightGBMモデル＋時系列クロスバリデーションまでを、`notebooks_local/` 以下のノートブックとして整理しています。

- `notebooks_local/demand_01_eda.ipynb`
  データ概要の確認と、代表系列（Store 1, GROCERY I）の時系列プロットによる基本的な傾向の把握。

- `notebooks_local/demand_02_baseline.ipynb`
  同じ代表系列を対象に、Naive法（1日前の売上）と7日移動平均によるベースラインモデルを構築し、RMSE/SMAPEおよび直近90日のプロットで精度感を評価。

- `notebooks_local/demand_03_lgbm_cv.ipynb`
  全店舗・全ファミリーに拡張し、カレンダー特徴量と売上のラグ・移動平均を入力とした LightGBM 回帰モデルを作成。2015〜2017年をテスト対象とした年ベースのexpanding window 3foldクロスバリデーションで、線形回帰とのRMSE比較を行っています。

### ベースラインモデル（単一系列）

代表系列（Store 1, GROCERY I）に対して、Naive法と7日移動平均のベースラインを評価しました。

| model     | RMSE   | SMAPE  |
|----------|--------|--------|
| naive_1d | 918.78 | 0.3623 |
| ma_7d    | 671.25 | 0.2363 |

Naive法は「1日前の売上=今日の予測」、7日移動平均は「直近7日間の平均売上」を予測としています。

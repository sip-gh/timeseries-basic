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


# Pythonで学ぶ時系列分析の基本

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

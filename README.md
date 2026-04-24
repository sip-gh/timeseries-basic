# Store Sales 需要予測 MVP

Kaggle「Store Sales – Time Series Forecasting」を題材に、
**Naive ベースライン → LightGBM + 時系列CV → SHAP 解釈**
までを一気通貫で実装した需要予測ポートフォリオです。

## 成果サマリ

| モデル | RMSE（3-fold 平均） | Naive 比 |
|--------|---------------------|---------|
| Naive (lag1) | 507.1 | ― |
| Linear Regression | 348.1 | -31% |
| **LightGBM v1** | **302.2** | **-40%** |

- TimeSeriesSplit（expanding window 3-fold）で未来リークなし評価
- SHAP・Feature Importance でラグ特徴量の寄与を確認済み
- MLflow で Naive vs LightGBM の run を記録

## 技術スタック
Python 3.11 / LightGBM / SHAP / MLflow / pandas / uv

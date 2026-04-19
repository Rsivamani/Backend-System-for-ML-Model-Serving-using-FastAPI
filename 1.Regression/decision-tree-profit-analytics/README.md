# 🌳 Decision Tree Regression — Profit Prediction API
### Production-Grade ML Service with Automated Model Selection

[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-F7931E)](https://scikit-learn.org)

## 📌 Problem Statement
Predict startup profit from operational expenditures using non-parametric regression — comparing interpretable decision trees against ensemble and kernel methods.

## 💡 Solution Overview
FastAPI service training **4 regression models**, comparing R²/MAE/RMSE, auto-selecting the best, and persisting via joblib.

## 🔬 Model Selection

> ⚠️ **No model pre-selected. Best is chosen objectively by test R².**

### Sample Comparison Table

| Model | train_R² | test_R² | MAE | RMSE | overfit_gap |
|-------|----------|---------|-----|------|-------------|
| LinearRegression | 0.9321 | 0.9347 | 7502 | 9128 | -0.0026 |
| **DecisionTreeRegressor** | 1.0000 | 0.8201 | 11200 | 15300 | 0.1799 |
| RandomForestRegressor | 0.9850 | 0.9502 | 6800 | 8100 | 0.0348 |
| SVR | 0.9100 | 0.8950 | 8600 | 11100 | 0.0150 |

> Note: Decision Tree often overfits (gap=0.18). Best model is auto-selected at runtime.

### ✅ Selection: Highest test R² with acceptable overfit_gap wins

> **⚠️ Note:** Model selection is data-dependent. The best performer may differ on other datasets.

## 🚀 How to Run
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
curl -X POST http://localhost:8000/train/
```

## 📡 Endpoints
| POST | `/train` | Auto-model selection |
| POST | `/predict` | Predict profit |
| GET | `/health` | Liveness |

## 🔮 Future Improvements
- Tree pruning analysis endpoint
- Feature importance (`/feature-importance`)
- Gradient Boosting comparison (XGBoost/LightGBM)
- MLflow experiment tracking

# 📊 Business Intelligence Profit Prediction API
### Production-Grade ML Service with Automated Model Selection

[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-F7931E)](https://scikit-learn.org)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker)](https://docker.com)

---

## 📌 Problem Statement

Investors and business analysts need a reliable tool to predict startup profitability based on operational spend. This API predicts **annual profit** from R&D, Administration, and Marketing expenditures across US states.

---

## 💡 Solution Overview

A FastAPI service that trains **4 regression models** simultaneously, compares performance objectively, auto-selects the winner, and serves real-time predictions through a typed REST API.

---

## 🏗️ Architecture

### MVC Pattern

| Layer | Component | Role |
|-------|-----------|------|
| **Model** | `services/training_service.py` | Multi-model Pipeline, joblib persistence |
| **View** | `schemas/`, JSON responses | Pydantic v2 request/response |
| **Controller** | `routes/`, `controllers/` | HTTP routing + orchestration |

### Data Pipeline

```
50_Startups.csv
  │
  ├─► dropna()
  ├─► Features: R&D Spend, Administration, Marketing Spend (numeric)
  │             State (categorical → OneHotEncoded)
  ├─► Target: Profit
  │
  ├─► ColumnTransformer: StandardScaler + OneHotEncoder(drop="first")
  │
  ├─► Train 4 models → compare R², MAE, RMSE, overfit_gap
  │
  └─► Best model → joblib.dump()
```

---

## 🔬 Model Selection

> ⚠️ **No model is assumed to be best.** All 4 are trained and objectively evaluated.

### Candidates

| # | Model | Characteristics |
|---|-------|----------------|
| 1 | **Linear Regression** | Baseline, assumes linear feature relationships |
| 2 | **Decision Tree Regressor** | Non-parametric, interpretable, may overfit |
| 3 | **Random Forest Regressor** | Ensemble, reduces overfitting via bagging |
| 4 | **SVR (RBF kernel)** | Handles non-linearity, kernel trick |

### Metrics Used

| Metric | Symbol | Interpretation |
|--------|--------|----------------|
| Test R² | `test_r2` | Proportion of variance explained on hold-out |
| MAE | `mae` | Mean prediction error (in USD) |
| RMSE | `rmse` | Root mean squared error (penalises outliers) |
| Overfit Gap | `overfit_gap` | train_R² − test_R² (lower = better generalisation) |

### Sample Comparison Table (returned by `/train`)

| Model | train_R² | test_R² | MAE | RMSE | overfit_gap |
|-------|----------|---------|-----|------|-------------|
| LinearRegression | 0.9321 | 0.9347 | 7502 | 9128 | -0.0026 |
| DecisionTreeRegressor | 1.0000 | 0.8201 | 11200 | 15300 | 0.1799 |
| **RandomForestRegressor** | **0.9850** | **0.9502** | **6800** | **8100** | **0.0348** |
| SVR | 0.9100 | 0.8950 | 8600 | 11100 | 0.0150 |

### ✅ Selected Model: **Random Forest Regressor**
> **Why:** Highest test R² (0.9502) with a small overfitting gap (0.0348), outperforming all alternatives on unseen data. The ensemble approach effectively handles multicollinearity between expenditure features.

> **⚠️ Important Note:** Model selection is data-dependent. The chosen model performed best on this dataset and **may vary with different data**.

---

## 🗂️ Folder Structure

```
business-intelligence-profit-api/
├── app/
│   ├── main.py
│   ├── routes/routes.py
│   ├── controllers/controllers.py
│   ├── services/
│   │   ├── training_service.py   # 4-model comparison + auto-selection
│   │   └── prediction_service.py
│   ├── models/
│   ├── schemas/schemas.py
│   └── utils/config.py
├── data/raw/50_Startups.csv
├── tests/test_api.py
├── requirements.txt
├── Dockerfile
└── README.md
```

## 🛠️ Tech Stack
FastAPI · scikit-learn (Pipeline, ColumnTransformer) · Pydantic v2 · pandas · joblib · pytest · Docker

## 🚀 How to Run

```bash
pip install -r requirements.txt
# Ensure 50_Startups.csv is in data/raw/
uvicorn app.main:app --reload
# → http://127.0.0.1:8000/docs
```

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Liveness probe |
| POST | `/train` | Multi-model training + auto-selection |
| POST | `/predict` | Predict profit |

### `POST /predict/` Sample

**Request:**
```json
{"rd_spend": 165349.20, "administration": 136897.80, "marketing_spend": 471784.10, "state": "New York"}
```

**Response:**
```json
{"predicted_profit": 192261.83, "currency": "USD", "state": "New York"}
```

### `POST /train/` Sample Response (abbreviated)
```json
{
  "best_model": "RandomForestRegressor",
  "best_metrics": {"test_r2": 0.9502, "mae": 6800.0, "rmse": 8100.0},
  "selection_reason": "RandomForestRegressor achieved the highest test R² (0.9502)...",
  "comparison_table": [...]
}
```

## 🔮 Future Improvements
- Feature importance endpoint (`/feature-importance`)
- Cross-validation (k-fold) for more robust estimates
- Hyperparameter tuning via GridSearchCV
- MLflow experiment tracking
- Drift detection and automatic retraining

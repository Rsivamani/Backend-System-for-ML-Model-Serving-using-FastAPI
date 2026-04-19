# ⚡ Business Intelligence Profit Prediction — SVR API
### Production-Grade ML Service with Automated Model Selection

[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-F7931E)](https://scikit-learn.org)

## 📌 Problem Statement
Predict startup annual profit from R&D, Administration, and Marketing spend — enabling data-driven investment decisions.

## 💡 Solution Overview
FastAPI service training **4 regression models**, auto-selecting the best by **test R²** and overfitting gap analysis.

## 🏗️ Architecture

| MVC Layer | Component |
|-----------|-----------|
| **Model** | 4-model Pipeline + joblib |
| **View** | Pydantic v2 typed responses |
| **Controller** | FastAPI routes + controllers |

## 🔬 Model Selection

> ⚠️ **No model pre-selected. Best is chosen objectively by test R².**

### Candidates

| Model | Notes |
|-------|-------|
| Linear Regression | Baseline with linear assumptions |
| Decision Tree Regressor | Non-parametric, may overfit |
| Random Forest Regressor | Ensemble bagging |
| **SVR (RBF kernel)** | Non-linear, kernel trick |

### Sample Comparison

| Model | train_R² | test_R² | MAE | overfit_gap |
|-------|----------|---------|-----|-------------|
| LinearRegression | 0.9321 | 0.9347 | 7502 | -0.0026 |
| DecisionTreeRegressor | 1.0000 | 0.8201 | 11200 | 0.1799 |
| RandomForestRegressor | 0.9850 | 0.9502 | 6800 | 0.0348 |
| **SVR** | 0.9100 | 0.8950 | 8600 | 0.0150 |

### ✅ Best model auto-selected at runtime by highest test R²

> **⚠️ Note:** Model selection is data-dependent and may vary with different datasets.

## 🚀 How to Run
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## 📡 Endpoints
| POST | `/train` | Compare 4 models, save best |
| POST | `/predict` | Predict profit |
| GET | `/health` | Liveness |

## 🔮 Future Improvements
- Hyperparameter tuning (C, epsilon, gamma for SVR)
- Cross-validation
- MLflow tracking

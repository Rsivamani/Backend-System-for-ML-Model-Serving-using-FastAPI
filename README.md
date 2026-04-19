# 🤖 ML Projects with FastAPI 

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-F7931E?logo=scikit-learn)](https://scikit-learn.org)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker)](https://docker.com)

> **7 machine learning projects** transformed from Jupyter notebooks into **FastAPI REST APIs** with **automated multi-model selection**, MVC architecture, Pydantic v2 validation, Docker support, and comprehensive unit tests.

---

## 📋 Project Index

| Project | Directory | Dataset | Candidates |
|---------|-----------|---------|------------|
| [HR Salary Prediction API](1.Regression/hr-salary-prediction-api/) | `hr-salary-prediction-api` | Salary_Data.csv | LR, DT, RF, SVR |
| [Startup Profit Forecasting Service](1.Regression/startup-profit-forecasting-service/) | `startup-profit-forecasting-service` | 50_Startups.csv | LR, DT, RF, SVR |
| [Support Vector Profit Analysis](1.Regression/support-vector-profit-analysis/) | `support-vector-profit-analysis` | 50_Startups.csv | LR, DT, RF, SVR |
| [Decision Tree Profit Analytics](1.Regression/decision-tree-profit-analytics/) | `decision-tree-profit-analytics` | 50_Startups.csv | LR, DT, RF, SVR |

### 🎯 Classification Projects (4 models compared per project)

| Project | Directory | Dataset | Candidates |
|---------|-----------|---------|------------|
| [Ad Purchase Prediction Classifier](2.classification/social-ads-purchase-classifier/) | `social-ads-purchase-classifier` | Social_Network_Ads.csv | Logistic, DT, RF, SVC |
| [Customer Selection Random Forest](2.classification/customer-targeting-random-forest/) | `customer-targeting-random-forest` | Social_Network_Ads.csv | Logistic, DT, RF, SVC |
| [Purchase Intent SVM Classifier](2.classification/svm-purchase-intent-prediction/) | `svm-purchase-intent-prediction` | Social_Network_Ads.csv | Logistic, DT, RF, SVC |

---

## 🧠 Model Selection Philosophy

> **"No algorithm is universally best. Let the data decide."**

Every project trains **all candidate models** and selects the winner objectively:

### Regression Selection
```
Train: LinearRegression, DecisionTreeRegressor, RandomForestRegressor, SVR
Metric: Highest test R²
Guard:  overfit_gap = train_R² − test_R² (lower is safer)
```

### Classification Selection
```
Train: LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, SVC
Metric: Highest F1 Score  (robust to class imbalance vs accuracy)
Guard:  overfit_gap = train_acc − test_acc
```

---

## 🏗️ Architecture (All 7 Projects)

```
Client (HTTP)
     │
     ▼
┌─────────────────────────┐
│  FastAPI (app/main.py)  │  ← VIEW: Pydantic schemas, JSON responses
└──────────┬──────────────┘
           │
     ┌─────▼─────┐
     │  Routes   │  ← CONTROLLER: /health, /train, /predict
     └─────┬─────┘
           │
   ┌───────▼────────┐
   │  Controllers   │  ← CONTROLLER: Business orchestration
   └───────┬────────┘
           │
    ┌──────▼──────┐
    │  Services   │  ← MODEL: Multi-model training + auto-selection
    └──────┬──────┘
           │
    ┌──────▼──────┐
    │joblib Model │  ← Best model persisted here
    └─────────────┘
```

### MVC Mapping
| Pattern | Project Component |
|---------|-----------------|
| **Model** | sklearn Pipeline (Preprocessor → Best Estimator) |
| **View** | Pydantic v2 schemas + FastAPI JSON responses |
| **Controller** | FastAPI routes + Controller classes |

---

## ⚡ Universal API Contract

Every project exposes the same 3 endpoints:

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | `{"status": "ok"}` |
| `POST` | `/train` | Train all models, auto-select best, return comparison table |
| `POST` | `/predict` | Predict using the best saved model |

### `/train` Response Structure (all projects)
```json
{
  "status": "success",
  "best_model": "RandomForestClassifier",
  "best_metrics": {
    "f1_score": 0.8642,
    "accuracy": 0.9083,
    "precision": 0.875,
    "recall": 0.8537
  },
  "selection_reason": "RandomForestClassifier achieved the highest F1 (0.8642)...",
  "note": "Model selection is data-dependent. May vary with different data.",
  "comparison_table": [
    {"model": "LogisticRegression", "accuracy": 0.8583, "f1_score": 0.7436, ...},
    {"model": "DecisionTreeClassifier", "accuracy": 0.8917, "f1_score": 0.8000, ...},
    {"model": "RandomForestClassifier", "accuracy": 0.9083, "f1_score": 0.8642, ...},
    {"model": "SVC", "accuracy": 0.9000, "f1_score": 0.8430, ...}
  ]
}
```

---

## 🚀 Quick Start

```bash
# Any project
cd "path/to/project"
python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload

# 1. Train (auto-selects best model)
curl -X POST http://localhost:8000/train/

# 2. Predict
curl -X POST http://localhost:8000/predict/ \
  -H "Content-Type: application/json" \
  -d '{"years_experience": 5.0}'   # or {"age":35,...} for classification

# Docs
open http://127.0.0.1:8000/docs
```

---

## 🐳 Docker

```bash
docker build -t ml-api .
docker run -p 8000:8000 -v $(pwd)/data:/app/data ml-api
```

---

## 🛠️ Tech Stack

| Layer | Technology | Version |
|-------|-----------|---------|
| API | FastAPI + Uvicorn | 0.111+ |
| ML | scikit-learn | 1.4+ |
| Validation | Pydantic v2 | 2.7+ |
| Config | pydantic-settings | 2.2+ |
| Persistence | joblib | 1.4+ |
| Data | pandas + numpy | 2.2+ / 1.26+ |


---

## 📊 Model Performance Summary

### Regression (50_Startups / Salary datasets)

| Algorithm | Typical test R² | Overfit Risk |
|-----------|----------------|-------------|
| Linear Regression | 0.93–0.96 | Low |
| Decision Tree Regressor | 0.82–0.90 | **High** |
| Random Forest Regressor | 0.93–0.95 | Medium |
| SVR (RBF) | 0.89–0.95 | Low |

### Classification (Social Network Ads)

| Algorithm | Typical F1 | Overfit Risk |
|-----------|-----------|-------------|
| Logistic Regression | 0.74 | Low |
| Decision Tree | 0.80 | **High** |
| Random Forest | **0.86** | Medium |
| SVC (RBF) | 0.84 | Low |

---





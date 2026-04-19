# 🎯 HR Salary Prediction API
### Production-Grade ML Service with Automated Model Selection

[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📌 Problem Statement

HR departments spend significant time benchmarking employee salaries without a data-driven baseline. This API provides a **real-time, evidence-based salary estimate** based on years of professional experience — enabling faster, fairer compensation decisions.

---

## 💡 Solution Overview

A **production-grade FastAPI service** that:
1. Trains **4 regression models** on the dataset
2. **Auto-selects the best** based on test R² and overfitting analysis
3. Persists only the winning model via `joblib`
4. Serves predictions through a clean REST API with Pydantic v2 validation

---

## 🏗️ Architecture

### MVC Pattern

```
┌─────────────────────────────────────────────────────┐
│                   CLIENT (HTTP)                      │
└─────────────────────────┬───────────────────────────┘
                          │
              ┌───────────▼───────────┐
              │   FastAPI (main.py)   │  ◄── VIEW
              │  Pydantic Schemas     │      (Request/Response validation)
              └───────────┬───────────┘
                          │
              ┌───────────▼───────────┐
              │  Routes + Controllers │  ◄── CONTROLLER
              │  (Business Logic)     │      (Orchestration)
              └───────────┬───────────┘
                          │
              ┌───────────▼───────────┐
              │      Services         │  ◄── MODEL
              │  (ML Pipeline Logic)  │      (Training, Prediction)
              └───────────┬───────────┘
                          │
              ┌───────────▼───────────┐
              │  joblib Model (.pkl)  │  ◄── Persisted Artefact
              └───────────────────────┘
```

| MVC Layer | Project Component | Responsibility |
|-----------|------------------|----------------|
| **View** | `schemas/`, FastAPI JSON responses | Request validation, response serialisation |
| **Controller** | `routes/`, `controllers/` | Route handling, business orchestration |
| **Model** | `services/`, `models/` | ML pipeline, training, prediction |

### Data Pipeline Flow

```
Salary_Data.csv
    │
    ├─► dropna() — remove nulls
    ├─► Feature: YearsExperience
    ├─► Target: Salary
    │
    ├─► train_test_split (70/30)
    │
    ├─► For each candidate model:
    │     StandardScaler → Estimator → Pipeline.fit()
    │     Evaluate: train_R², test_R², MAE, MSE, RMSE, overfit_gap
    │
    ├─► Select: highest test_R² + acceptable overfit_gap
    │
    └─► joblib.dump(best_pipeline) → app/models/
```

---

## 🔬 Model Selection (Critical Section)

> ⚠️ **No model is assumed to be best.** All candidates are trained and evaluated. The winner is selected objectively by metrics.

### Candidates Trained

| # | Model | Type | Notes |
|---|-------|------|-------|
| 1 | **Linear Regression** | Parametric | Baseline — assumes linearity |
| 2 | **Decision Tree Regressor** | Non-parametric | Handles non-linearity, prone to overfit |
| 3 | **Random Forest Regressor** | Ensemble | Reduces DT variance via bagging |
| 4 | **SVR (RBF kernel)** | Kernel-based | Effective in high-dimensional space |

### Evaluation Metrics

| Metric | Meaning |
|--------|---------|
| **test_R²** | Explained variance on unseen data (higher = better) |
| **MAE** | Mean absolute error in salary (lower = better) |
| **RMSE** | Root mean squared error (penalises large errors) |
| **overfit_gap** | `train_R² − test_R²` (lower = better generalisation) |

### Sample Comparison Table (from `/train` response)

| Model | train_R² | test_R² | MAE | RMSE | overfit_gap |
|-------|----------|---------|-----|------|-------------|
| LinearRegression | 0.9450 | 0.9569 | 3426 | 5592 | -0.0119 |
| DecisionTreeRegressor | 0.9980 | 0.8800 | 4200 | 7100 | 0.1180 |
| RandomForestRegressor | 0.9870 | 0.9310 | 3800 | 6100 | 0.0560 |
| SVR | 0.9500 | 0.9480 | 3550 | 5800 | 0.0020 |

### ✅ Selected Model: **Linear Regression**
> **Why:** Highest test R² (0.9569) with near-zero overfitting gap (−0.0119), confirming excellent generalisation. The salary-experience relationship is fundamentally linear in this dataset — more complex models overfit.

> **⚠️ Important Note:** Model selection is **data-dependent**. The chosen model performed best on this specific dataset. Results may differ with larger datasets, additional features, or different data distributions.

---

## 🗂️ Folder Structure

```
hr-salary-prediction-api/
│
├── app/
│   ├── main.py                     # FastAPI app with lifespan hooks
│   ├── routes/
│   │   └── routes.py               # /health, /train, /predict endpoints
│   ├── controllers/
│   │   ├── training_controller.py  # Orchestrates training pipeline
│   │   └── prediction_controller.py
│   ├── services/
│   │   ├── training_service.py     # ← Multi-model training + auto-selection
│   │   └── prediction_service.py  # Lazy model loading + inference
│   ├── models/                     # joblib artefact storage
│   ├── schemas/
│   │   ├── request_schema.py       # Pydantic v2 (years_experience ≥ 0)
│   │   └── response_schema.py      # Typed response
│   └── utils/
│       └── config.py               # pydantic-settings (.env aware)
│
├── data/
│   ├── raw/Salary_Data.csv
│   └── processed/
│
├── notebooks/                      # Original exploration (reference only)
├── tests/
│   └── test_api.py                 # pytest with TestClient
│
├── requirements.txt
├── Dockerfile
├── .gitignore
├── config.yaml
└── README.md
```

---

## 🛠️ Tech Stack

| Layer | Technology | Version |
|-------|-----------|---------|
| API Framework | FastAPI + Uvicorn | 0.111+ |
| ML Library | scikit-learn | 1.4+ |
| Data Validation | Pydantic v2 | 2.7+ |
| Configuration | pydantic-settings | 2.2+ |
| Model Persistence | joblib | 1.4+ |
| Data Processing | pandas, numpy | 2.2+, 1.26+ |
| Testing | pytest + httpx | 8.0+ |
| Container | Docker | — |
| Language | Python | 3.11 |

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Liveness probe |
| `POST` | `/train` | Run multi-model selection, save best |
| `POST` | `/predict` | Predict salary from experience |
| `GET` | `/docs` | Interactive Swagger UI |

### Request / Response Examples

#### `POST /train/`
```bash
curl -X POST http://localhost:8000/train/
```
```json
{
  "status": "success",
  "message": "Best model selected and saved: LinearRegression",
  "best_model": "LinearRegression",
  "best_metrics": {
    "test_r2": 0.9569,
    "mae": 3426.88,
    "rmse": 5592.04
  },
  "selection_reason": "LinearRegression achieved the highest test R² (0.9569) with acceptable generalisation.",
  "comparison_table": [
    {"model": "LinearRegression", "train_r2": 0.945, "test_r2": 0.9569, "mae": 3426.88, "overfit_gap": -0.0119},
    {"model": "DecisionTreeRegressor", "train_r2": 0.998, "test_r2": 0.88, "mae": 4200.0, "overfit_gap": 0.118},
    {"model": "RandomForestRegressor", "train_r2": 0.987, "test_r2": 0.931, "mae": 3800.0, "overfit_gap": 0.056},
    {"model": "SVR", "train_r2": 0.950, "test_r2": 0.948, "mae": 3550.0, "overfit_gap": 0.002}
  ]
}
```

#### `POST /predict/`
```bash
curl -X POST http://localhost:8000/predict/ \
  -H "Content-Type: application/json" \
  -d '{"years_experience": 5.0}'
```
```json
{
  "years_experience": 5.0,
  "predicted_salary": 72000.50,
  "currency": "USD"
}
```

---

## 🚀 How to Run

### 1. Local Setup
```bash
# Clone and navigate
git clone https://github.com/your-username/ml-fastapi-portfolio
cd "1.Regression/1.SLR_HR Salary Prediction"

# Create venv
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Start API
uvicorn app.main:app --reload
```

### 2. Access Swagger UI
```
http://127.0.0.1:8000/docs
```

### 3. Train the Model
```bash
curl -X POST http://localhost:8000/train/
```

### 4. Predict
```bash
curl -X POST http://localhost:8000/predict/ \
  -H "Content-Type: application/json" \
  -d '{"years_experience": 5.0}'
```

### 5. Docker
```bash
docker build -t hr-salary-api .
docker run -p 8000:8000 -v $(pwd)/data:/app/data hr-salary-api
```

### 6. Tests
```bash
pytest tests/ -v
```

---

## 🔮 Future Improvements

- [ ] Cross-validation (k-fold) instead of single train/test split
- [ ] Polynomial features for capturing non-linear salary growth
- [ ] Confidence intervals on predictions
- [ ] `/model-info` endpoint — returns currently loaded model metadata
- [ ] MLflow integration for experiment tracking
- [ ] Automated retraining on data drift detection
- [ ] API key authentication
- [ ] Prometheus metrics for monitoring

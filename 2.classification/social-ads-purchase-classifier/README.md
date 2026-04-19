# 🎯 Social Network Ads — Purchase Prediction API
### Production-Grade ML Classification Service with Automated Model Selection

[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-F7931E)](https://scikit-learn.org)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker)](https://docker.com)

---

## 📌 Problem Statement

Digital marketers need to identify which users are most likely to purchase a product after seeing a social media advertisement. Accurate purchase prediction enables **targeted campaign spending** — reducing wasted ad budget and increasing ROI.

---

## 💡 Solution Overview

A FastAPI service that:
1. Trains **4 classifier models** on the Social Network Ads dataset
2. **Compares Accuracy, Precision, Recall, and F1** on a hold-out set
3. **Auto-selects by F1 Score** (robust to class imbalance)
4. Serves real-time predictions with **purchase probability and confidence score**

---

## 🏗️ Architecture

### MVC Pattern

| Layer | Component | Role |
|-------|-----------|------|
| **Model** | `services/training_service.py` | 4-model Pipeline training + auto-selection |
| **View** | `schemas/`, JSON responses | Pydantic v2 validated request/response |
| **Controller** | `routes/`, `controllers/` | HTTP routing + orchestration |

### Data Pipeline

```
Social_Network_Ads.csv
  │
  ├─► dropna() → drop("User ID")
  ├─► Features: Age, EstimatedSalary (numeric) + Gender (categorical)
  ├─► Target: Purchased (binary: 0/1)
  │
  ├─► ColumnTransformer:
  │     StandardScaler(Age, EstimatedSalary)
  │     OneHotEncoder(Gender, drop="first")
  │
  ├─► Train 4 classifiers → compare Accuracy/Precision/Recall/F1
  │
  └─► Best model (highest F1) → joblib.dump()
```

---

## 🔬 Model Selection

> ⚠️ **No model is assumed to be best.** All classifiers are trained and evaluated. F1 Score is used as the primary selection metric because it balances precision and recall, making it more robust than raw accuracy for potentially imbalanced datasets.

### Candidates Evaluated

| # | Model | Key Characteristic |
|---|-------|-------------------|
| 1 | **Logistic Regression** | Baseline linear classifier, probabilistic output |
| 2 | **Decision Tree Classifier** | Interpretable, non-linear boundaries |
| 3 | **Random Forest Classifier** | Ensemble — reduces DT variance |
| 4 | **SVC (RBF kernel)** | Optimal decision boundary in high-dim space |

### Metrics Used

| Metric | Meaning | Why It Matters |
|--------|---------|---------------|
| **Accuracy** | % correct predictions | Overall performance |
| **Precision** | Of predicted positives, % actually positive | Cost of false alarms |
| **Recall** | Of actual positives, % caught | Cost of missed buyers |
| **F1 Score** | Harmonic mean of Precision + Recall | Balanced, primary selection metric |
| **overfit_gap** | train_acc − test_acc | Overfitting indicator |

### Sample Comparison Table (from `/train` response)

| Model | train_acc | accuracy | precision | recall | f1_score | overfit_gap |
|-------|-----------|---------|-----------|--------|----------|-------------|
| LogisticRegression | 0.8643 | 0.8583 | 0.7838 | 0.7073 | 0.7436 | 0.0060 |
| DecisionTreeClassifier | 1.0000 | 0.8917 | 0.8205 | 0.7805 | 0.8000 | 0.1083 |
| **RandomForestClassifier** | **0.9857** | **0.9083** | **0.8750** | **0.8537** | **0.8642** | **0.0774** |
| SVC | 0.9357 | 0.9000 | 0.8571 | 0.8293 | 0.8430 | 0.0357 |

### ✅ Selected Model: **Random Forest Classifier**
> **Why:** Highest F1 Score (0.8642) across all classifiers, with strong precision (0.875) and recall (0.854). The ensemble approach handled the feature interactions between Age, Salary, and Gender better than individual models.

> **⚠️ Important Note:** Model selection is data-dependent. The chosen model performed best on this dataset and **may vary with different data distributions or feature sets**.

---

## 🗂️ Folder Structure

```
social-network-purchase-prediction/
├── app/
│   ├── main.py
│   ├── routes/routes.py          # /health /train /predict
│   ├── controllers/controllers.py
│   ├── services/
│   │   ├── training_service.py   # 4-model comparison pipeline
│   │   └── prediction_service.py # Inference with probability
│   ├── models/                   # Best model saved here
│   ├── schemas/schemas.py        # Pydantic v2 schemas
│   └── utils/config.py
├── data/raw/Social_Network_Ads.csv
├── tests/test_api.py
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| API | FastAPI + Uvicorn |
| ML | scikit-learn (Pipeline, ColumnTransformer) |
| Validation | Pydantic v2 |
| Config | pydantic-settings |
| Persistence | joblib |
| Testing | pytest + httpx |
| Container | Docker |

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Liveness probe |
| `POST` | `/train` | Train 4 models, auto-select best |
| `POST` | `/predict` | Classify + return purchase probability |
| `GET` | `/docs` | Swagger UI |

### `POST /train/` — Response
```json
{
  "status": "success",
  "best_model": "RandomForestClassifier",
  "best_metrics": {
    "accuracy": 0.9083,
    "precision": 0.875,
    "recall": 0.8537,
    "f1_score": 0.8642,
    "overfit_gap": 0.0774
  },
  "selection_reason": "RandomForestClassifier achieved the highest F1 score (0.8642)...",
  "note": "Model selection is data-dependent. Results may vary with different data.",
  "comparison_table": [...]
}
```

### `POST /predict/` — Request & Response
```bash
curl -X POST http://localhost:8000/predict/ \
  -H "Content-Type: application/json" \
  -d '{"age": 35, "estimated_salary": 60000, "gender": "Male"}'
```
```json
{
  "age": 35,
  "estimated_salary": 60000.0,
  "gender": "Male",
  "purchased": 0,
  "purchase_probability": 0.23,
  "confidence": 0.77,
  "prediction_label": "Will NOT Purchase"
}
```

---

## 🚀 How to Run

```bash
# Install
pip install -r requirements.txt

# Place dataset
# data/raw/Social_Network_Ads.csv

# Start API
uvicorn app.main:app --reload

# Train (select best model)
curl -X POST http://localhost:8000/train/

# Predict
curl -X POST http://localhost:8000/predict/ \
  -H "Content-Type: application/json" \
  -d '{"age": 40, "estimated_salary": 80000, "gender": "Female"}'

# Swagger UI
open http://127.0.0.1:8000/docs

# Tests
pytest tests/ -v

# Docker
docker build -t social-ads-clf-api .
docker run -p 8000:8000 -v $(pwd)/data:/app/data social-ads-clf-api
```

---

## 🔮 Future Improvements

- [ ] Feature importance endpoint (`/feature-importance`)
- [ ] ROC-AUC and confusion matrix endpoint (`/evaluate`)
- [ ] Cross-validation (stratified k-fold)
- [ ] Hyperparameter tuning (GridSearchCV for RF n_estimators, max_depth)
- [ ] SHAP explainability integration
- [ ] MLflow experiment tracking
- [ ] Retraining pipeline on data drift
- [ ] API key authentication

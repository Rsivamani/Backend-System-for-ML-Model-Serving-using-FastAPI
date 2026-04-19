# 🤖 Social Network Ads — Purchase Prediction API (SVM)
### Production-Grade ML Service with Automated Model Selection

[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-F7931E)](https://scikit-learn.org)

## 📌 Problem Statement
Predict product purchase intent from demographic signals — enabling precision targeting and smarter ad spend allocation.

## 💡 Solution Overview
FastAPI service evaluating **4 classifiers**, auto-selecting the best by **F1 Score** to handle precision-recall trade-offs, serving predictions with purchase probability.

## 🏗️ Architecture

| MVC Layer | Component |
|-----------|-----------|
| **Model** | 4-classifier Pipeline + joblib persistence |
| **View** | Pydantic v2 validated JSON |
| **Controller** | FastAPI routes + Controller classes |

## 🔬 Model Selection

> ⚠️ **No model is pre-selected. All 4 classifiers compete on F1 Score.**

### Candidates Evaluated

| Model | Characteristic |
|-------|---------------|
| Logistic Regression | Linear, probabilistic |
| Decision Tree Classifier | Tree splits on feature thresholds |
| Random Forest Classifier | Bagged ensemble |
| **SVC (RBF kernel)** | Max-margin hyperplane |

### Sample Comparison Table

| Model | accuracy | precision | recall | f1_score | overfit_gap |
|-------|---------|-----------|--------|----------|-------------|
| LogisticRegression | 0.8583 | 0.7838 | 0.7073 | 0.7436 | 0.0060 |
| DecisionTreeClassifier | 0.8917 | 0.8205 | 0.7805 | 0.8000 | 0.1083 |
| RandomForestClassifier | 0.9083 | 0.875 | 0.8537 | 0.8642 | 0.0774 |
| **SVC** | **0.9000** | **0.8571** | **0.8293** | **0.8430** | **0.0357** |

### ✅ Selected Model (data-dependent)
The model with the **highest F1 Score** and **lowest overfit_gap** is automatically selected at training time.

> **⚠️ Note:** Model selection is data-dependent. The chosen model performed best on this dataset and may vary with different data distributions.

## 🛠️ Tech Stack
FastAPI · scikit-learn (SVC + ensemble) · Pydantic v2 · joblib · pytest · Docker

## 🚀 How to Run
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
curl -X POST http://localhost:8000/train/
```
Swagger: http://127.0.0.1:8000/docs

## 📡 API Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /health | Liveness |
| POST | /train | Compare 4 models, auto-select best |
| POST | /predict | Predict purchase + probability |

## 📥 Sample

**Request:** `POST /predict/`
```json
{"age": 40, "estimated_salary": 80000, "gender": "Female"}
```

**Response:**
```json
{
  "purchased": 1, "purchase_probability": 0.85,
  "confidence": 0.85, "prediction_label": "Will Purchase"
}
```

## 🔮 Future Improvements
- ROC-AUC endpoint
- Kernel comparison (linear, poly, rbf)
- SHAP explainability
- Hyperparameter tuning (GridSearchCV for C, gamma)
- MLflow integration

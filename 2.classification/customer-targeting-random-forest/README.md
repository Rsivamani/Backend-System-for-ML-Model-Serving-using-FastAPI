# 🌲 Social Network Ads — Purchase Prediction API (Random Forest)
### Production-Grade ML Service with Automated Model Selection

[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-F7931E)](https://scikit-learn.org)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker)](https://docker.com)

## 📌 Problem Statement
Predict whether a social network user will purchase a product — enabling targeted ad campaigns and reducing wasted spend.

## 💡 Solution Overview
FastAPI service training **4 classifiers** (Logistic Regression, Decision Tree, Random Forest, SVC), auto-selecting the best by **F1 Score**, and serving predictions with purchase probability.

## 🏗️ Architecture

| MVC Layer | Component |
|-----------|-----------|
| **Model** | 4-classifier Pipeline training + auto-selection |
| **View** | Pydantic v2 schemas + JSON responses |
| **Controller** | FastAPI routes + Controller classes |

## Data Flow
```
Social_Network_Ads.csv → dropna → ColumnTransformer → 4 classifiers → best F1 → joblib
```

## 🔬 Model Selection

> ⚠️ **No model is assumed best. All 4 are trained and the best is selected by F1 Score.**

### Candidates

| Model | Type |
|-------|------|
| Logistic Regression | Linear baseline |
| Decision Tree Classifier | Non-parametric, interpretable |
| **Random Forest Classifier** | Ensemble (n_estimators=100) |
| SVC (RBF kernel) | Maximum margin classifier |

### Sample Comparison Table

| Model | accuracy | precision | recall | f1_score | overfit_gap |
|-------|---------|-----------|--------|----------|-------------|
| LogisticRegression | 0.8583 | 0.7838 | 0.7073 | 0.7436 | 0.0060 |
| DecisionTreeClassifier | 0.8917 | 0.8205 | 0.7805 | 0.8000 | 0.1083 |
| **RandomForestClassifier** | **0.9083** | **0.875** | **0.8537** | **0.8642** | 0.0774 |
| SVC | 0.9000 | 0.8571 | 0.8293 | 0.8430 | 0.0357 |

### ✅ Selected: **Random Forest Classifier**
Best F1 (0.8642) — strongest balance of precision and recall. Ensemble reduces individual tree variance.

> **⚠️ Note:** Model selection is data-dependent. The chosen model performed best on this dataset and may vary with different data.

## 🛠️ Tech Stack
FastAPI · scikit-learn · Pydantic v2 · pandas · joblib · pytest · Docker

## 🚀 How to Run
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
# POST /train/ → then POST /predict/
```
Docs: http://127.0.0.1:8000/docs

## 📡 API Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /health | Liveness |
| POST | /train | Train 4 models, auto-select |
| POST | /predict | Classify with probability |

## Sample Predict
```json
POST /predict/
{"age": 35, "estimated_salary": 60000, "gender": "Male"}
→
{"purchased": 0, "purchase_probability": 0.23, "prediction_label": "Will NOT Purchase"}
```

## 🔮 Future Improvements
- Feature importance endpoint
- Cross-validation (stratified k-fold)
- SHAP explainability
- Hyperparameter tuning (GridSearchCV)
- MLflow tracking

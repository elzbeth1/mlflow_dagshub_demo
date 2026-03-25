# mlflow_dagshub_demo
Demo for ML FLow and Dagshub

# 🚀 MLflow-Based Anomaly Detection System

## 📌 Project Overview

This project builds an end-to-end **Machine Learning pipeline** for anomaly detection / credit risk classification using **MLflow for experiment tracking and model management**, integrated with **DagsHub for remote tracking**.

---

## 🎯 Objective

To identify high-risk customers (anomalies) using machine learning models trained on imbalanced data.

---

## 🧠 Key Features

* ✅ Handles imbalanced data using **SMOTE / SMOTETomek**
* ✅ Trains multiple models (Logistic Regression, Random Forest, XGBoost)
* ✅ Tracks experiments using **MLflow**
* ✅ Logs parameters, metrics, and models
* ✅ Uses **MLflow Model Registry** for versioning
* ✅ Promotes best model to **Production**
* ✅ Integrated with **DagsHub** for remote experiment tracking

---

## ⚙️ Tech Stack

* Python 🐍
* Scikit-learn
* XGBoost
* MLflow
* DagsHub
* Pandas, NumPy

---

## 🔄 ML Workflow

### 1. Data Preprocessing

* Missing value handling
* Encoding categorical variables
* Feature scaling

---

### 2. Handling Imbalanced Data

* Applied:

  * SMOTE
  * SMOTETomek

---

### 3. Model Training

Trained multiple models:

* Logistic Regression
* Random Forest
* XGBoost

---

### 4. Experiment Tracking (MLflow)

Tracked:

* Hyperparameters
* Metrics:

  * Accuracy
  * Recall
  * F1-score

```python
with mlflow.start_run(run_name=model_name):
    mlflow.log_params(params)
    mlflow.log_metric("accuracy", accuracy)
```

---

### 5. Model Comparison

* Compared models using:

  * F1 Score
  * Recall (important for imbalance)

---

### 6. Model Logging

```python
mlflow.xgboost.log_model(model, model_name)
```

---

### 7. Model Registry

```python
mlflow.register_model(model_uri, model_name)
```

---

### 8. Model Versioning

* Maintained multiple versions of models
* Enabled reproducibility

---

### 9. Model Deployment (Production Stage)

```python
model_uri = "models:/XGB-Smote/Production"
model = mlflow.xgboost.load_model(model_uri)
```

---

### 10. Remote Tracking with DagsHub

* Stored experiments remotely
* Enabled collaboration
* Centralized experiment tracking

---

## 📊 Results

* Improved model performance using resampling techniques
* Achieved better recall for minority class
* Successfully tracked and compared multiple models

---

## 📂 Project Structure

```
├── data/
├── notebooks/
│   └── ML Flow Model Registry.ipynb
├── src/
├── mlartifacts/
├── mlflow.db
└── README.md
```

---

## 🚀 How to Run

```bash
# Clone repo
git clone https://github.com/your-username/mlflow_dagshub_demo.git

# Activate environment
conda activate ds_env

# Run notebook
jupyter notebook
```

---

## 🔥 Future Improvements

* Deploy model using Streamlit / FastAPI
* Add CI/CD pipeline
* Automate retraining
* Add monitoring

---

## 💡 Key Learnings

* Importance of experiment tracking
* Handling imbalanced datasets
* Model lifecycle management
* Real-world MLOps workflow

---

## 🙌 Acknowledgements

This project demonstrates practical implementation of:

* MLflow
* Model Registry
* Experiment tracking
* MLOps fundamentals

---

## ⭐ If you like this project

Give it a ⭐ on GitHub!


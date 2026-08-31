# retail-sales-regression-random-forest-xgboost-deployment
Production-oriented retail sales regression using Random Forest and tuned XGBoost, with Scikit-learn pipelines and GridSearchCV for reproducible preprocessing and model optimization. Served via Flask/Gunicorn, containerized with Docker, surfaced through Streamlit, and deployed on Hugging Face Spaces.

🛒 SuperKart Sales Prediction

An end-to-end Machine Learning Deployment Case Study for predicting retail outlet sales using XGBoost, Flask, Docker, Streamlit, and Hugging Face Spaces.

Overview

SuperKart is a retail chain operating supermarkets and food marts across different tier cities.

Accurate sales forecasting can help retail organizations make better decisions around:

* Inventory planning
* Product procurement
* Regional sales strategies
* Store-level planning
* Future sales expectations

The objective of this project is to build and deploy a machine learning solution that predicts sales using historical product and store information.

The project covers the complete workflow from data preprocessing and model development to API deployment and an interactive frontend.

⸻

Architecture

                    ┌──────────────────────┐
                    │     Retail Data      │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Data Preprocessing   │
                    │  Scikit-learn        │
                    │  ColumnTransformer   │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Model Training     │
                    │ Random Forest / XGB  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │    GridSearchCV      │
                    │ Hyperparameter Tune  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Tuned XGBoost      │
                    └──────────┬───────────┘
                               │
                            Joblib
                               │
                               ▼
                    ┌──────────────────────┐
                    │      Flask API       │
                    │      Gunicorn        │
                    └──────────┬───────────┘
                               │
                            Docker
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Hugging Face Space   │
                    └──────────▲───────────┘
                               │
                         HTTP Requests
                               │
                    ┌──────────┴───────────┐
                    │    Streamlit UI      │
                    └──────────────────────┘

⸻

Technology Stack

Python

Python is used across the complete solution—from data analysis and model training to backend API development and frontend integration.

Pandas & NumPy

Used for:

* Data manipulation
* Feature preparation
* DataFrame construction
* Preparing prediction requests and responses

Scikit-learn

Scikit-learn provides the preprocessing, pipeline, evaluation, and hyperparameter-tuning infrastructure.

The preprocessing pipeline handles different feature types independently.

Numerical Features

Missing Value Imputation
        ↓
StandardScaler

SimpleImputer handles missing values, while StandardScaler standardizes numerical variables.

Nominal Categorical Features

Missing Value Imputation
        ↓
OneHotEncoder

OneHotEncoder converts non-ordered categories into machine-readable features while supporting previously unseen categories.

Ordinal Features

Missing Value Imputation
        ↓
OrdinalEncoder

OrdinalEncoder preserves the natural ordering of values such as store size and city tier.

Why use a Pipeline?

The preprocessing logic and machine learning estimator are bundled into one scikit-learn pipeline.

This is important because the same transformations used during model training must also be applied during inference.

Instead of separately preprocessing incoming production data, the application can simply execute:

Raw Input
   ↓
Saved Pipeline
   ↓
Preprocessing
   ↓
XGBoost
   ↓
Prediction

This reduces the possibility of training-serving inconsistencies.

⸻

Machine Learning Models

Two ensemble regression approaches were explored.

Random Forest Regressor

Random Forest provides a strong bagging-based baseline for structured data by combining predictions from multiple decision trees.

The base model showed a larger difference between training and testing performance, so hyperparameter tuning was applied.

XGBoost Regressor

XGBoost uses gradient boosting, where models are built sequentially to reduce errors made by earlier trees.

It was selected as the final model because it provided:

* Strong overall prediction performance
* Better train/test generalization
* Effective modeling of nonlinear relationships
* Stable performance across the evaluated regression metrics

⸻

Hyperparameter Tuning

GridSearchCV was used with 5-fold cross-validation to search for stronger model configurations.

For XGBoost, the search included parameters related to:

* Number of estimators
* Tree depth
* Column sampling
* Learning rate
* L2 regularization

R² was used as the GridSearchCV scoring metric.

⸻

Model Evaluation

Regression performance was evaluated using:

Metric	Purpose
R²	Measures the proportion of sales variance explained by the model
Adjusted R²	Adjusts R² based on the number of predictors
RMSE	Penalizes larger prediction errors
MAE	Measures average absolute prediction error
MAPE	Expresses average error relative to actual sales

Final Tuned XGBoost

Test R²   : 0.9207
Test MAPE : 4.59%

The tuned model explains approximately 92.07% of the variance in unseen product-store sales, while its average percentage prediction error is below 5%.

⸻

Model Serialization — Joblib

After selecting the tuned XGBoost pipeline, the complete pipeline is serialized using joblib.

superkart_sales_prediction_model.joblib

Joblib allows the trained preprocessing and model pipeline to be loaded directly by the prediction service without retraining the model.

⸻

Backend — Flask REST API

A Flask application wraps the serialized machine learning model.

The backend:

1. Loads the saved Joblib model.
2. Accepts prediction inputs.
3. Converts input data into the expected format.
4. Runs the saved preprocessing/model pipeline.
5. Returns the predicted sales value as JSON.

Prediction Endpoint

POST /v1/sales

Flask was chosen because it provides a lightweight way to expose a Python machine learning model through an HTTP API.

⸻

Gunicorn

Gunicorn is used to run the Flask backend.

The Docker configuration launches Gunicorn using four worker processes and exposes the API on port 7860.

This separates the Flask application code from the HTTP-serving process and allows multiple worker processes to handle requests.

⸻

Frontend — Streamlit

Streamlit provides the interactive user interface.

Users can provide information such as:

* Product weight
* Product allocated area
* Product MRP
* Product type
* Sugar content
* Store
* Store type
* Store size
* City tier
* Store establishment information

The Streamlit application sends these inputs to the backend API and displays the resulting sales prediction.

Streamlit was used because it allows a Python-based ML model to be exposed through an interactive UI without requiring a separate JavaScript frontend.

⸻

Frontend–Backend Communication

The architecture separates the user interface from the prediction service.

Streamlit
    │
    │ HTTP request
    ▼
Flask REST API
    │
    ▼
Saved ML Pipeline
    │
    ▼
XGBoost Prediction
    │
    ▼
JSON Response
    │
    ▼
Streamlit UI

The Python requests library is used for HTTP communication.

This separation means the same prediction API could later support another frontend or application without changing the ML model.

⸻

Docker

Both application components are packaged using Docker.

Docker provides:

* Reproducible runtime environments
* Explicit dependency management
* Consistent Python versions
* Easier application deployment
* Separation between frontend and backend services

The backend container includes:

app.py
superkart_sales_prediction_model.joblib
requirements.txt
Dockerfile

The backend uses a python:3.9-slim base image and launches the Flask application through Gunicorn.

The Streamlit frontend is also containerized using Python 3.9.

⸻

Deployment — Hugging Face Spaces

Hugging Face Spaces is used to host the application.

The backend is configured as a Docker Space, while the frontend provides the interactive Streamlit experience.

The huggingface_hub Python library is used as part of the deployment workflow for repository/Space creation and file uploads.

⸻

End-to-End Workflow

1. Load retail data
        ↓
2. Explore and prepare data
        ↓
3. Build preprocessing pipeline
        ↓
4. Train Random Forest
        ↓
5. Train XGBoost
        ↓
6. Compare model performance
        ↓
7. Tune hyperparameters with GridSearchCV
        ↓
8. Select tuned XGBoost
        ↓
9. Serialize pipeline with Joblib
        ↓
10. Build Flask prediction API
        ↓
11. Serve API using Gunicorn
        ↓
12. Containerize with Docker
        ↓
13. Build Streamlit frontend
        ↓
14. Connect frontend to REST API
        ↓
15. Deploy using Hugging Face Spaces

⸻

Why This Project Matters

This case study goes beyond training a regression model.

It demonstrates the engineering steps needed to move a machine learning solution from a notebook toward a usable application:

Data → Feature Engineering → ML Pipeline → Model Selection → Hyperparameter Tuning → Serialization → REST API → Containerization → UI → Deployment

The key takeaway is that model accuracy is only one part of an ML system. Reproducible preprocessing, reliable inference, API design, environment consistency, and user accessibility are equally important when operationalizing machine learning.

⸻

Tech Stack

Python · Pandas · NumPy · Scikit-learn · XGBoost · GridSearchCV · Joblib · Flask · Gunicorn · REST API · Requests · Streamlit · Docker · Hugging Face Spaces

⸻

Project Type

Machine Learning | Regression | Retail Sales Forecasting | Model Deployment | MLOps Case Study

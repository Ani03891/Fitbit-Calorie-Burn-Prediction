# Fitbit Calorie Burn Prediction & Workout Pattern Clustering

## Project Overview

This project predicts calories burned during workouts using Machine Learning models and discovers workout patterns using Unsupervised Learning techniques. The project includes data preprocessing, feature engineering, regression models, clustering, and model comparison.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Jupyter Notebook

---

## Project Structure

```
Fitbit_Calorie_Prediction/
│
├── data/
│   └── Fitbit_dataset.csv
│
├── notebooks/
│   ├── hyperparameter_Tuning.ipynb
│   └── unsupervised_ln.ipynb
│
├── output/
│   ├── cluster_summary.csv
│   ├── feature_importance.csv
│   └── model_comparison.csv
│
├── models/
│   (ignored by Git)
│
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Machine Learning Models

- Linear Regression
- Ridge Regression
- Lasso Regression
- Decision Tree Regressor
- Random Forest Regressor
- Support Vector Regressor
- K-Nearest Neighbors Regressor
- XGBoost Regressor

---

## Unsupervised Learning

- K-Means Clustering
- Workout Pattern Analysis
- Cluster Summary

---

## Evaluation Metrics

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

---

## Features

- Data Cleaning
- Outlier Handling using IQR
- One-Hot Encoding
- Feature Scaling
- Model Comparison
- Feature Importance Analysis
- Hyperparameter Tuning
- Workout Clustering

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Run

```bash
python main.py
```

---

## Author

**Anitha Preethi**
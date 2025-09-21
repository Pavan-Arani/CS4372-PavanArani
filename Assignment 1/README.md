# Linear Regression Analysis – Wine Quality Dataset
## Overview
This project applies linear regression models to the Wine Quality (Red Wine) dataset from the UCI Machine Learning Repository.  

The models that were compared:
1. SGDRegressor (Scikit-learn) – stochastic gradient descent regression with hyperparameter tuning.  
2. Ordinary Least Squares (OLS) (Statsmodels) – traditional regression analysis with interpretability.  

The goal is to look at the model performance differences between the 2 models and compare results between them.

## Dataset
- Source: [Wine Quality Dataset (Red Wine)](https://archive.ics.uci.edu/dataset/186/wine+quality)  
- Samples: 1,599 wines.  
- Features: 11 properties (acidity, alcohol, pH, etc.).
- Target: Wine quality score (0–10).  

## Requirements
Install dependencies with:
```bash
pip install pandas scikit-learn statsmodels matplotlib seaborn
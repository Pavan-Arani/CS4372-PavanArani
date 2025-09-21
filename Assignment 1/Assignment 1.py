import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm

url = "https://raw.githubusercontent.com/Pavan-Arani/CS4372-PavanArani/refs/heads/main/Assignment%201/winequality-red.csv"
df = pd.read_csv(url, sep = ";")  # file uses semicolons

# Check for missing values and nulls
print(df.isnull().sum())

# Features and target
X = df.drop("quality", axis=1)
y = df["quality"]

# Correlation heatmap
plt.figure(figsize = (10,8))
sns.heatmap(df.corr(), annot = True, cmap = "coolwarm")
plt.show()

# Shows relation between the attributes and their means.
print(df.describe())
df.hist(figsize=(12, 12))
plt.show()

# Data has no missing values, but features need standardization.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size = 0.2, random_state = 42
)

# SGDRegressor Model
sgd = SGDRegressor(
    loss = "squared_error",
    penalty = "l2",                   # L2 regularization
    alpha = 0.001,                    # Regularization strength
    max_iter = 1000,                  # Max iterations
    learning_rate = "invscaling",     # Learning rate schedule
    eta0 = 0.01,                      # Initial learning rate
    random_state = 42
)

sgd.fit(X_train, y_train)
y_pred_sgd = sgd.predict(X_test)

# Output results
print("SGD Regressor Results")
print("---------------------")
print("SGD R^2 Score:", r2_score(y_test, y_pred_sgd))
print("SGD Mean Squared Error:", mean_squared_error(y_test, y_pred_sgd))

# OLS Model
X_train_const = sm.add_constant(X_train)
ols_model = sm.OLS(y_train, X_train_const).fit()

# Output results
print("\nOLS Regression Results")
print("----------------------")
print(ols_model.summary())

# Plot predicted vs actual values for SGD
plt.scatter(y_test, y_pred_sgd, alpha=0.5)
plt.xlabel("Actual Quality")
plt.ylabel("Predicted Quality (SGD)")
plt.title("SGD: Predicted vs Actual")
plt.show()
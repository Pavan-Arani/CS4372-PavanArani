import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm

url = "https://raw.githubusercontent.com/Pavan-Arani/CS4372-PavanArani/refs/heads/main/Assignment%201/winequality-red.csv"
df = pd.read_csv(url, sep=";")  # file uses semicolons

print(df.head())
print(df.info())

# Check nulls
print(df.isnull().sum())

# Features and target
X = df.drop("quality", axis=1)
y = df["quality"]

# Summary stats
print(df.describe())

# Correlation heatmap
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.show()

# Data has no missing values, but features need standardization.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# SGDRegressor Model
sgd = SGDRegressor(
    loss="squared_error",
    penalty="l2",
    alpha=0.001,
    max_iter=1000,
    learning_rate="invscaling",
    eta0=0.01,
    random_state=42
)

sgd.fit(X_train, y_train)
y_pred_sgd = sgd.predict(X_test)

print("SGD R²:", r2_score(y_test, y_pred_sgd))
print("SGD MSE:", mean_squared_error(y_test, y_pred_sgd))


# OLS Model
X_train_const = sm.add_constant(X_train)
ols_model = sm.OLS(y_train, X_train_const).fit()
print(ols_model.summary())


plt.scatter(y_test, y_pred_sgd, alpha=0.5)
plt.xlabel("Actual Quality")
plt.ylabel("Predicted Quality (SGD)")
plt.title("SGD: Predicted vs Actual")
plt.show()
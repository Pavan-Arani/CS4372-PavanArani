import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                            precision_recall_curve, roc_curve, auc)
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42
plt.style.use('seaborn-v0_8')

# Load data from public URL
url = "https://raw.githubusercontent.com/Pavan-Arani/CS4372-PavanArani/refs/heads/main/Assignment%201/winequality-red.csv"
df = pd.read_csv(url, sep=';')

# Check for missing values and inconsistencies
print("--- DATA QUALITY CHECK ---")
print("Missing values:", df.isnull().sum().sum())
print("Duplicates:", df.duplicated().sum())

# Examine target variable and convert to binary classification
df['quality_binary'] = (df['quality'] >= 7).astype(int)
print("\nTarget distribution:")
print(df['quality_binary'].value_counts())

# Check for normal distribution
print("\n--- DISTRIBUTION ANALYSIS ---")
features = [col for col in df.columns if col not in ['quality', 'quality_binary']]
for col in features:
    print(f"{col}: Skewness={df[col].skew():.2f}")

# Correlation analysis
print("\n--- CORRELATION WITH TARGET ---")
corr_matrix = df[features + ['quality_binary']].corr()
target_corr = corr_matrix['quality_binary'].abs().sort_values(ascending=False)
print(target_corr)

# Plot correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.show()

# Select important features (correlation > 0.05 with target)
important_features = target_corr[target_corr > 0.05].index.tolist()
important_features.remove('quality_binary')
print(f"\nSelected features: {important_features}")

# Prepare data
X = df[important_features]
y = df['quality_binary']

# Split and standardize
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(f"\nTraining samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

# Model Building

print("\n" + "="*80)
print("Building the Models")

# 1. Decision Tree
print("\n1. Decision Tree Classifier")

dt_params = {
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 5, 10],
    'criterion': ['gini', 'entropy']
}
dt = GridSearchCV(DecisionTreeClassifier(random_state=RANDOM_STATE), 
                  dt_params, cv=5, scoring='f1')
dt.fit(X_train, y_train)
print(f"Best params: {dt.best_params_}")
print(f"Best CV F1: {dt.best_score_:.4f}")

# 2. Random Forest
print("\n2. Random Forest Classifier")
rf_params = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5],
    'max_features': ['sqrt', 'log2']
}
rf = GridSearchCV(RandomForestClassifier(random_state=RANDOM_STATE), 
                  rf_params, cv=5, scoring='f1')
rf.fit(X_train, y_train)
print(f"Best params: {rf.best_params_}")
print(f"Best CV F1: {rf.best_score_:.4f}")

# 3. AdaBoost
print("\n3. AdaBoost Classifier")
ada_params = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.1, 0.5, 1.0]
}
ada = GridSearchCV(AdaBoostClassifier(random_state=RANDOM_STATE), 
                   ada_params, cv=5, scoring='f1')
ada.fit(X_train, y_train)
print(f"Best params: {ada.best_params_}")
print(f"Best CV F1: {ada.best_score_:.4f}")

# 4. XGBoost
print("\n4. XGBoost Classifier")
xgb_params = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3]
}

xgb = GridSearchCV(XGBClassifier(random_state=RANDOM_STATE, eval_metric='logloss'), 
                   xgb_params, cv=5, scoring='f1')
xgb.fit(X_train, y_train)
print(f"Best params: {xgb.best_params_}")
print(f"Best CV F1: {xgb.best_score_:.4f}")

# Tree Visualization

print("\n" + "="*80)
print("Tree Visualizations")

# Visualize Decision Tree
plt.figure(figsize=(20, 10))
plot_tree(dt.best_estimator_, feature_names=important_features,
          class_names=['Bad', 'Good'], filled=True, rounded=True)
plt.title('Decision Tree Visualization')
plt.show()

# Visualize one tree from Random Forest
plt.figure(figsize=(20, 10))
plot_tree(rf.best_estimator_.estimators_[0], feature_names=important_features,
          class_names=['Bad', 'Good'], filled=True, rounded=True)
plt.title('Random Forest - Single Tree Visualization')
plt.show()

# Result Analysis

print("\n" + "="*80)
print("Model Eval")

models = {
    'Decision Tree': dt.best_estimator_,
    'Random Forest': rf.best_estimator_,
    'AdaBoost': ada.best_estimator_,
    'XGBoost': xgb.best_estimator_
}

results = []

for name, model in models.items():
    print(f"\n{name}:")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    print(classification_report(y_test, y_pred, target_names=['Bad', 'Good']))
    
    # Store for plotting
    results.append({
        'name': name,
        'y_pred': y_pred,
        'y_proba': y_proba,
        'model': model
    })

# Confusion Matrices
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for idx, r in enumerate(results):
    cm = confusion_matrix(y_test, r['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
    axes[idx].set_title(f"{r['name']}\nConfusion Matrix")
    axes[idx].set_ylabel('True')
    axes[idx].set_xlabel('Predicted')
plt.tight_layout()
plt.show()

# ROC Curves
plt.figure(figsize=(10, 8))
for r in results:
    fpr, tpr, _ = roc_curve(y_test, r['y_proba'])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f"{r['name']} (AUC={roc_auc:.3f})")
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# Precision-Recall Curves
plt.figure(figsize=(10, 8))
for r in results:
    precision, recall, _ = precision_recall_curve(y_test, r['y_proba'])
    plt.plot(recall, precision, lw=2, label=r['name'])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves')
plt.legend()
plt.grid(alpha=0.3)
plt.show()
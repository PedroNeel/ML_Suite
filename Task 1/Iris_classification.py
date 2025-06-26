# Iris_classification.py (with auto-download)
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import os
import urllib.request

# Download dataset if missing
if not os.path.exists('iris.csv'):
    print("Downloading Iris dataset...")
    iris_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    urllib.request.urlretrieve(iris_url, "iris.csv")

# Load and preprocess data
iris = pd.read_csv('iris.csv', header=None)
iris.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

print("Data loaded successfully!")
print("First 5 rows:")
print(iris.head())

print("\nMissing values:")
print(iris.isnull().sum())

# Handle missing values
for col in iris.columns[:-1]:
    if iris[col].isnull().sum() > 0:
        iris[col].fillna(iris[col].mean(), inplace=True)

# Encode labels
le = LabelEncoder()
iris['species'] = le.fit_transform(iris['species'])

# Train-test split
X = iris.drop('species', axis=1)
y = iris['species']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Model training
param_grid = {'max_depth': [2, 3, 4, 5], 'min_samples_split': [2, 5, 10]}
model = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print(f"\nBest Parameters: {model.best_params_}")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision (Macro): {precision_score(y_test, y_pred, average='macro'):.4f}")
print(f"Recall (Macro): {recall_score(y_test, y_pred, average='macro'):.4f}")

# Visualization
plt.figure(figsize=(10, 6))
cm = confusion_matrix(y_test, y_pred)
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()

# Use class names for ticks
tick_marks = np.arange(len(le.classes_))
plt.xticks(tick_marks, le.classes_, rotation=45)
plt.yticks(tick_marks, le.classes_)

# Add text annotations
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig('confusion_matrix.png')
print("\nConfusion matrix saved as 'confusion_matrix.png'")

# Feature importance
plt.figure(figsize=(10, 6))
importances = model.best_estimator_.feature_importances_
features = X.columns
sorted_idx = np.argsort(importances)
plt.barh(range(len(sorted_idx)), importances[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), features[sorted_idx])
plt.xlabel('Feature Importance')
plt.title('Feature Importances')
plt.savefig('feature_importances.png')
print("Feature importances saved as 'feature_importances.png'")
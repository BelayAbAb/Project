import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report

# 4.2.1 Data Preparation
# Load your dataset
df = pd.read_csv(r'C:\Users\User\Desktop\10Acadamy\Week 8\Resource\Data\creditcard.csv')  

# Feature and target separation
X = df.drop(columns='Class')  # Replace 'Class' with the actual target variable name
y = df['Class']

# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4.2.2 Model Selection
# Initialize the model
model = RandomForestClassifier(random_state=42)

# Hyperparameter tuning (optional)
param_grid = {
    'max_depth': [10, 15, 18],
    'class_weight': [{0: 1, 1: 10}, {0: 1, 1: 5}, {0: 1, 1: 2}],
}
cv = StratifiedKFold(n_splits=5)
grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='f1', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_

# 4.2.3 Model Training and Evaluation
f1_scores = []

for fold, (train_index, test_index) in enumerate(cv.split(X_train, y_train)):
    X_fold_train, X_fold_test = X_train.iloc[train_index], X_train.iloc[test_index]
    y_fold_train, y_fold_test = y_train.iloc[train_index], y_train.iloc[test_index]

    best_model.fit(X_fold_train, y_fold_train)
    y_pred = best_model.predict(X_fold_test)
    f1 = f1_score(y_fold_test, y_pred)
    f1_scores.append(f1)
    print(f"Fold {fold + 1}: F1 Score = {f1:.4f}")

print(f"Mean F1 Score: {np.mean(f1_scores):.4f}")

# Evaluate on test set
y_proba = best_model.predict_proba(X_test)
threshold = 0.3
y_pred = (y_proba[:, 1] >= threshold).astype(int)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Normal (0)', 'Fraud (1)'],
            yticklabels=['Normal (0)', 'Fraud (1)'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('confusion_matrix.jpg')  # Save the confusion matrix as JPG
plt.show()

# Classification report
class_report = classification_report(y_test, y_pred)
print("\nClassification Report:")
print(class_report)

# Save classification report as text
with open('classification_report.txt', 'w') as f:
    f.write(class_report)


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import mlflow

# Define the path to the data
data_path = r"C:\Users\User\Desktop\10Acadamy\Week 8\Project\report"
file_name = "Processed_Fraud_Data.csv"
file_path = os.path.join(data_path, file_name)

# Load datasets with dtype specification
dtype_spec = {
    'user_id': 'int64',
    'signup_time': 'str',
    'purchase_time': 'str',
    'purchase_value': 'float64',
    'device_id': 'str',
    'age': 'int64',
    'ip_address': 'float64',
    'class': 'int64',
    'lower_bound_ip_address': 'float64',
    'upper_bound_ip_address': 'float64',
    'country': 'str',
    'transaction_frequency': 'int64',
    'velocity': 'float64',
    'signup_hour': 'int64',
    'signup_day': 'int64'
}

# Check if the file exists
if not os.path.isfile(file_path):
    raise FileNotFoundError(f"File not found: {file_path}")

# Load data
fraud_data = pd.read_csv(file_path, dtype=dtype_spec, low_memory=False)

# Convert timestamps to datetime
fraud_data['signup_time'] = pd.to_datetime(fraud_data['signup_time'], errors='coerce')
fraud_data['purchase_time'] = pd.to_datetime(fraud_data['purchase_time'], errors='coerce')

# Feature Engineering
# Transaction frequency and velocity
fraud_data['transaction_frequency'] = fraud_data.groupby('user_id')['purchase_value'].transform('count')
fraud_data['velocity'] = fraud_data['purchase_value'] / fraud_data['transaction_frequency']

# Time-based features
fraud_data['signup_hour'] = fraud_data['signup_time'].dt.hour
fraud_data['signup_day'] = fraud_data['signup_time'].dt.dayofweek

# Normalization and Scaling
numerical_features = ['purchase_value', 'age', 'transaction_frequency', 'velocity']
scaler = MinMaxScaler()
fraud_data[numerical_features] = scaler.fit_transform(fraud_data[numerical_features])

# Handling High Cardinality Categorical Features
# Define the maximum number of unique categories to keep
max_categories = 20

# Function to limit categories
def limit_categories(df, column, max_categories):
    counts = df[column].value_counts()
    top_categories = counts.nlargest(max_categories).index
    df[column] = np.where(df[column].isin(top_categories), df[column], 'Other')

# Apply to categorical features
limit_categories(fraud_data, 'device_id', max_categories)
limit_categories(fraud_data, 'country', max_categories)

# Encoding Categorical Features
categorical_features = ['device_id', 'country']
fraud_data = pd.get_dummies(fraud_data, columns=categorical_features, drop_first=True)

# Feature and Target Separation
X = fraud_data.drop(columns=['class'])
y = fraud_data['class']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Selection and Training
# Example models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier()
}

# Use MLflow for tracking
mlflow.start_run()

for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Evaluation
    report = classification_report(y_test, y_pred, output_dict=True)
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    
    print(f"{model_name} Classification Report:\n", report)
    print(f"{model_name} ROC AUC Score: {roc_auc:.2f}")
    
    # Log metrics with MLflow
    mlflow.log_param("model_name", model_name)
    mlflow.log_metric("roc_auc", roc_auc)

mlflow.end_run()

# Save modified data to CSV
output_data_path = os.path.join(data_path, "Modified_Fraud_Data.csv")
fraud_data.to_csv(output_data_path, index=False)

print("Data processing complete. Modified data saved and models trained.")

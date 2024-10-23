import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout  
from tensorflow.keras.regularizers import l2
import warnings

# Set up the seaborn palette
palette = ['#00777F', '#5BABF5', '#AADEFE', '#EAAC9F', '#8AA0AF']
sns.set_theme(context='notebook', palette=palette, style='darkgrid')

# Suppress FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Load dataset
df = pd.read_csv(r"C:\Users\User\Desktop\10Acadamy\Week 8\Resource\Data\creditcard.csv")
print(df.head())

# Data Exploration
print(f"The dataset has {df.shape[0]} rows and {df.shape[1]} columns.")
status_counts = df.Class.value_counts()
plt.figure(figsize=(7, 7))
plt.pie(status_counts, labels=status_counts.index, autopct='%1.1f%%',
        startangle=140, colors=palette, shadow=True)
plt.title('Distribution of a Target Variable')
plt.axis('equal')  
plt.tight_layout()
plt.show()

features = df.columns[:-1]

fig, axes = plt.subplots(10, 3, figsize=(15, 40))  
axes = axes.flatten()

for i, feature in enumerate(features):
    sns.histplot(df[feature], ax=axes[i], kde=False, bins=30)
    axes[i].set_title(f'Histogram of {feature}')
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel('Frequency')

for i in range(len(features), len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.show()

# Data Transformation
df_transformed = df.copy()

def log_transform_skewed(column):
    transformed = np.where(column >= 0, np.log1p(column), -np.log1p(-column))
    return transformed

skewness_before = df.skew()

for col in features:
    if abs(df[col].skew()) > 0.75:
        df_transformed[col] = log_transform_skewed(df[col])

skewness_after = df_transformed.skew()
skewness_comparison = pd.DataFrame({
    'Skewness Before': skewness_before,
    'Skewness After': skewness_after
})

print(skewness_comparison)

# Histograms after transformation
fig, axes = plt.subplots(10, 3, figsize=(15, 40))  
axes = axes.flatten()

for i, feature in enumerate(features):
    sns.histplot(df_transformed[feature], ax=axes[i], kde=False, bins=30)
    axes[i].set_title(f'{feature} after Transformation')
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel('Frequency')

for i in range(len(features), len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.show()

# Scaling and Outlier Detection
X = df_transformed[features]
y = df_transformed.Class

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Isolation Forest
iso_forest = IsolationForest(contamination=0.05, random_state=101)  
iso_preds = iso_forest.fit_predict(X_scaled)
iso_preds = [1 if x == -1 else 0 for x in iso_preds]

print(classification_report(y, iso_preds))
roc_auc = roc_auc_score(y, iso_preds)
print("ROC AUC Score: ", roc_auc)

# Confusion Matrix
cm = confusion_matrix(y, iso_preds)
custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', ['#CFEEF0', '#00777F'])
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, cmap=custom_cmap, fmt='g')
plt.title('Confusion Matrix ')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.xticks([0.5, 1.5], ['Non-Fraud (0)', 'Fraud (1)'])
plt.yticks([0.5, 1.5], ['Non-Fraud (0)', 'Fraud (1)'])
plt.show()

# Add other outlier detection methods (OneClassSVM, LOF, DBSCAN) as needed
# Follow the same structure for the remaining parts of the original code.

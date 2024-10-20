import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the processed data
merged_data = pd.read_csv(r'C:\Users\User\Desktop\10Acadamy\Week 8\Resource\Data\merged_processed_data.csv')

# Debug: Print column names to check for 'class'
print(merged_data.columns)

# 1. Age Distribution
plt.figure(figsize=(10, 6))
sns.histplot(merged_data['age'], bins=30, kde=True)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.savefig(r'C:\Users\User\Desktop\10Acadamy\Week 8\Project\report\Age_Distribution.jpg')
plt.close()

# 2. Correlation Matrix
# Select only numeric columns
numeric_cols = merged_data.select_dtypes(include='number')
correlation_matrix = numeric_cols.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Matrix')
plt.savefig(r'C:\Users\User\Desktop\10Acadamy\Week 8\Project\report\Correlation_Matrix.jpg')
plt.close()

# 3. Count of Fraud Classes
# Use 'class' instead of 'fraud_class'
if 'class' in merged_data.columns:
    fraud_counts = merged_data['class'].value_counts()

    plt.figure(figsize=(10, 6))
    sns.barplot(x=fraud_counts.index, y=fraud_counts.values)
    plt.title('Count of Fraud Classes')
    plt.xlabel('Fraud Class')
    plt.ylabel('Count')
    plt.savefig(r'C:\Users\User\Desktop\10Acadamy\Week 8\Project\report\Count_of_Fraud_Classes.jpg')
    plt.close()
else:
    print("Column 'class' does not exist in the DataFrame.")

# 4. Fraud Counts by Country
if 'country' in merged_data.columns:
    fraud_counts_by_country = merged_data['country'].value_counts()

    plt.figure(figsize=(10, 6))
    sns.barplot(x=fraud_counts_by_country.index, y=fraud_counts_by_country.values)
    plt.title('Fraud Counts by Country')
    plt.xlabel('Country')
    plt.ylabel('Count')
    plt.savefig(r'C:\Users\User\Desktop\10Acadamy\Week 8\Project\report\Fraud_Counts_by_Country.jpg')
    plt.close()
else:
    print("Column 'country' does not exist in the DataFrame.")

# 5. Purchase Value by Fraud Class
if 'class' in merged_data.columns:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='class', y='purchase_value', data=merged_data)
    plt.title('Purchase Value by Fraud Class')
    plt.xlabel('Fraud Class')
    plt.ylabel('Purchase Value')
    plt.savefig(r'C:\Users\User\Desktop\10Acadamy\Week 8\Project\report\Purchase_Value_by_Fraud_Class.jpg')
    plt.close()
else:
    print("Column 'class' does not exist in the DataFrame.")

# 6. Purchase Value Distribution
plt.figure(figsize=(10, 6))
sns.histplot(merged_data['purchase_value'], bins=30, kde=True)
plt.title('Purchase Value Distribution')
plt.xlabel('Purchase Value')
plt.ylabel('Frequency')
plt.savefig(r'C:\Users\User\Desktop\10Acadamy\Week 8\Project\report\Purchase_Value_Distribution.jpg')
plt.close()

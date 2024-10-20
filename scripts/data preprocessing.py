import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Load the datasets
fraud_data = pd.read_csv(r'C:\Users\User\Desktop\10Acadamy\Week 8\Resource\Data\Fraud_Data.csv')
ip_data = pd.read_csv(r'C:\Users\User\Desktop\10Acadamy\Week 8\Resource\Data\IpAddress_to_Country.csv')
creditcard_data = pd.read_csv(r'C:\Users\User\Desktop\10Acadamy\Week 8\Resource\Data\creditcard.csv')

# 1. Handle Missing Values
# Impute missing values in Fraud Data
imputer_fraud = SimpleImputer(strategy='mean')
fraud_data['purchase_value'] = imputer_fraud.fit_transform(fraud_data[['purchase_value']])
fraud_data.dropna(inplace=True)  # Drop rows with any remaining missing values

# Impute missing values in Credit Card Data
creditcard_data.fillna(creditcard_data.mean(), inplace=True)

# 2. Data Cleaning
# Remove duplicates in both datasets
fraud_data.drop_duplicates(inplace=True)
creditcard_data.drop_duplicates(inplace=True)

# Ensure correct data types in Fraud Data
fraud_data['signup_time'] = pd.to_datetime(fraud_data['signup_time'])  # Removed format
fraud_data['purchase_time'] = pd.to_datetime(fraud_data['purchase_time'])  # Removed format

# 3. Merge Datasets for Geolocation Analysis
# Convert IP addresses to integer format
def ip_to_int(ip):
    if isinstance(ip, str):  # Check if the IP is a string
        return sum([int(part) << (8 * (3 - idx)) for idx, part in enumerate(ip.split('.'))])
    return np.nan  # Return NaN for non-string values

fraud_data['ip_address'] = fraud_data['ip_address'].apply(ip_to_int)
merged_data = fraud_data.merge(ip_data, left_on='ip_address', right_on='lower_bound_ip_address', how='left')

# 4. Feature Engineering
# Create transaction frequency and velocity features
transaction_frequency = fraud_data.groupby('user_id').size().reset_index(name='transaction_frequency')
merged_data = merged_data.merge(transaction_frequency, on='user_id', how='left')
merged_data['velocity'] = merged_data['purchase_value'] / (merged_data['purchase_time'].dt.hour + 1)

# 5. Normalization and Scaling
scaler = StandardScaler()
merged_data[['purchase_value', 'transaction_frequency', 'velocity']] = scaler.fit_transform(
    merged_data[['purchase_value', 'transaction_frequency', 'velocity']]
)

# Normalize Amount in Credit Card Data
creditcard_data['Amount'] = scaler.fit_transform(creditcard_data[['Amount']])

# 6. Encode Categorical Features
# One-hot encoding for gender in Fraud Data
encoder = OneHotEncoder(sparse=False)
encoded_gender = encoder.fit_transform(merged_data[['sex']])
encoded_gender_df = pd.DataFrame(encoded_gender, columns=encoder.get_feature_names_out(['sex']))
merged_data = pd.concat([merged_data.reset_index(drop=True), encoded_gender_df.reset_index(drop=True)], axis=1)

# 7. Save the merged data
merged_data.to_csv(r'C:\Users\User\Desktop\10Acadamy\Week 8\Resource\Data\merged_processed_data.csv', index=False)

print("Data preprocessing complete and saved to merged_processed_data.csv")

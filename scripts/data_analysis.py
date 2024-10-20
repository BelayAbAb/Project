import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# Sample Data Creation
data = {
    'user_id': [22058, 333320, 1359, 150084, 221365, 159135, 50116, 360585, 
                159045, 182338, 199700, 73884, 79203, 299320, 82931, 
                31383, 78986],
    'signup_time': ['24/02/2015 22:55', '07/06/2015 20:39', '01/01/2015 18:52', 
                    '28/04/2015 21:13', '21/07/2015 07:09', '21/05/2015 06:03', 
                    '01/08/2015 22:40', '06/04/2015 07:35', '21/04/2015 23:38', 
                    '25/01/2015 17:49', '11/07/2015 18:26', '29/05/2015 16:22', 
                    '16/06/2015 21:19', '03/03/2015 19:17', '16/02/2015 02:50', 
                    '01/02/2015 01:06', '15/05/2015 03:52'],
    'purchase_time': ['18/04/2015 02:47', '08/06/2015 01:38', '01/01/2015 18:52', 
                      '04/05/2015 13:54', '09/09/2015 18:40', '09/07/2015 08:05', 
                      '27/08/2015 03:37', '25/05/2015 17:21', '02/06/2015 14:01', 
                      '23/03/2015 23:05', '28/10/2015 21:59', '16/06/2015 05:45', 
                      '21/06/2015 03:29', '05/04/2015 12:32', '16/04/2015 00:56', 
                      '24/03/2015 10:17', '11/08/2015 02:29'],
    'purchase_value': [0.172413793, 0.048275862, 0.04137931, 0.24137931,
                       0.206896552, 0.227586207, 0.013793103, 0.124137931,
                       0.144827586, 0.365517241, 0.027586207, 0.337931034,
                       0.062068966, 0.282758621, 0.04137931, 0.337931034,
                       0.331034483],
    'device_id': ['QVPSPJUOCKZAR', 'EOGFQPIZPYXFZ', 'YSSKYOSJHPPLJ', 
                  'ATGTXKYKUDUQN', 'NAUITBZFJKHWW', 'ALEYXFXINSXLZ', 
                  'IWKVZHJOCLPUR', 'HPUCUYLMJBYFW', 'ILXYDOZIHOOHT', 
                  'NRFFPPHZYFUVC', 'TEPSJVVXGNTYR', 'ZTZZJUCRDOCJZ', 
                  'IBPNKSMCKUZWD', 'RMKQNVEWGTWPC', 'XKIFNYUZMBWFU', 
                  'UNUAVQXNWFBMO', 'TGHVAWBMZRDHH'],
    'age': [39, 53, 53, 41, 45, 18, 19, 34, 43, 31, 35, 32, 33, 38, 24, 24, 23],
    'ip_address': [732758368.8, 350311387.9, 2621473820, 3840542444,
                   415583117.5, 2809315200, 3987484329, 1692458728,
                   3719094257, 341674739.6, 1819008578, 4038284553,
                   4161540927, 3178510015, 4203487754, 995732779,
                   3503883392],
    'class': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'ip_integer': [732758368, 350311387, 2621473820, 3840542443,
                   415583117, 2809315199, 3987484328, 1692458727,
                   3719094257, 341674739, 1819008577, 4038284553,
                   4161540926, 3178510014, 4203487753, 995732779,
                   3503883391],
    'lower_bound_ip_address': [0] * 17,
    'upper_bound_ip_address': [0] * 17,
    'country': [None] * 17,  # Placeholder for country
    'transaction_frequency': [0] * 17,  # Placeholder
    'velocity': [0.172413793, 0.048275862, 0.04137931, 0.24137931,
                 0.206896552, 0.227586207, 0.013793103, 0.124137931,
                 0.144827586, 0.365517241, 0.027586207, 0.337931034,
                 0.062068966, 0.282758621, 0.04137931, 0.337931034,
                 0.331034483],
    'signup_hour': [22, 20, 18, 21, 7, 6, 22, 7, 23, 17, 18, 16, 21, 19, 2, 1, 3],
    'signup_day': [1, 6, 3, 1, 1, 3, 5, 0, 1, 6, 5, 4, 1, 1, 0, 6, 4],
    'source_Direct': [False, False, False, False, False, False, False, False, False, False, False, True, False, True, False, False, True],
    'source_SEO': [True, False, True, True, False, True, False, False, True, False, False, False, True, False, False, True, True],
    'browser_FireFox': [False] * 17,
    'browser_IE': [False] * 17,
    'browser_Opera': [False] * 17,
    'browser_Safari': [False] * 17,
    'sex_M': [True, False, True, True, True, True, False, True, False, False, False, False, True, True, False, True, True],
}

fraud_data = pd.DataFrame(data)

# Data Preprocessing
# Convert time columns to datetime
fraud_data['signup_time'] = pd.to_datetime(fraud_data['signup_time'], format='%d/%m/%Y %H:%M')
fraud_data['purchase_time'] = pd.to_datetime(fraud_data['purchase_time'], format='%d/%m/%Y %H:%M')

# Calculate transaction frequency
fraud_data['transaction_frequency'] = fraud_data.groupby('user_id')['purchase_time'].transform('count')

# Normalization and Scaling
scaler = MinMaxScaler()
fraud_data[['purchase_value', 'age']] = scaler.fit_transform(fraud_data[['purchase_value', 'age']])

# Exploratory Data Analysis (EDA)

# 1. Age Distribution
plt.figure(figsize=(10, 6))
sns.histplot(fraud_data['age'], bins=30, kde=True)
plt.title('Age Distribution')
plt.savefig(r'C:\Users\User\Desktop\10Acadamy\Week 8\Project\report\Age_Distribution.jpg')
plt.close()

# 2. Correlation Matrix
numeric_data = fraud_data.select_dtypes(include=[np.number])  # Select only numeric columns
numeric_data = numeric_data.drop(columns=['user_id'], errors='ignore')  # Exclude user_id
plt.figure(figsize=(12, 8))
sns.heatmap(numeric_data.corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Matrix')
plt.savefig(r'C:\Users\User\Desktop\10Acadamy\Week 8\Project\report\Correlation_Matrix.jpg')
plt.close()

# 3. Count of Fraud Classes
plt.figure(figsize=(10, 6))
sns.countplot(x='class', data=fraud_data)
plt.title('Count of Fraud Classes')
plt.savefig(r'C:\Users\User\Desktop\10Acadamy\Week 8\Project\report\Count_of_Fraud_Classes.jpg')
plt.close()

# 4. Fraud Counts by Country
fraud_data['country'] = ['US', 'US', 'UK', 'UK', 'US', 'CA', 'CA', 
                          'US', 'US', 'UK', 'CA', 'US', 'UK', 'US', 
                          'US', 'CA', 'UK']
fraud_counts_by_country = fraud_data['country'].value_counts()
plt.figure(figsize=(10, 6))
fraud_counts_by_country.plot(kind='bar')
plt.title('Fraud Counts by Country')
plt.savefig(r'C:\Users\User\Desktop\10Acadamy\Week 8\Project\report\Fraud_Counts_by_Country.jpg')
plt.close()

# 5. Purchase Value by Fraud Class
plt.figure(figsize=(10, 6))
sns.boxplot(x='class', y='purchase_value', data=fraud_data)
plt.title('Purchase Value by Fraud Class')
plt.savefig(r'C:\Users\User\Desktop\10Acadamy\Week 8\Project\report\Purchase_Value_by_Fraud_Class.jpg')
plt.close()

# 6. Purchase Value Distribution
plt.figure(figsize=(10, 6))
sns.histplot(fraud_data['purchase_value'], bins=30, kde=True)
plt.title('Purchase Value Distribution')
plt.savefig(r'C:\Users\User\Desktop\10Acadamy\Week 8\Project\report\Purchase_Value_Distribution.jpg')
plt.close()

# 7. Purchase Value vs Age
plt.figure(figsize=(10, 6))
sns.scatterplot(x='age', y='purchase_value', hue='class', data=fraud_data)
plt.title('Purchase Value vs Age')
plt.savefig(r'C:\Users\User\Desktop\10Acadamy\Week 8\Project\report\Purchase_Value_vs_Age.jpg')
plt.close()

# Save the modified data to a CSV file
output_data_path = r'C:\Users\User\Desktop\10Acadamy\Week 8\Project\reportProcessed_Fraud_Data.csv'
fraud_data.to_csv(output_data_path, index=False)


print("Modified data saved to CSV files and visualizations saved as JPG.")

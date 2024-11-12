import pandas as pd
from sklearn.preprocessing import StandardScaler

# 讀取資料
file_path = 'supermarket_sales - Sheet1.csv'
data = pd.read_csv(file_path)

# 日期和時間的處理
data['Date'] = pd.to_datetime(data['Date'])
data['Time'] = pd.to_datetime(data['Time']).dt.time
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['Day'] = data['Date'].dt.day
data['Day_of_Week'] = data['Date'].dt.dayofweek

# 刪除不需要的欄位
data = data.drop(['Date', 'Invoice ID'], axis=1)

# 一熱編碼處理類別變數
data_encoded = pd.get_dummies(data, columns=['Branch', 'City', 'Customer type', 'Gender', 'Product line', 'Payment'], drop_first=True)

# 數值欄位標準化
scaler = StandardScaler()
numerical_cols = ['Unit price', 'Quantity', 'Tax 5%', 'Total', 'cogs', 'gross margin percentage', 'gross income', 'Rating']
data_encoded[numerical_cols] = scaler.fit_transform(data_encoded[numerical_cols])

# 1. 分類分析資料
classification_data = data_encoded.drop(['Time'], axis=1)  # 假設分類分析不需要 'Time' 欄位
classification_data.to_csv('classification_data.csv', index=False)

# 2. 數值特徵統計分析（已處理過的數據）
numerical_summary = data_encoded.describe()
numerical_summary.to_csv('numerical_summary.csv', index=False)

# 3. 時間序列資料準備
time_series_data = data.groupby(['Year', 'Month', 'Day']).agg({'Total': 'mean'}).reset_index()
time_series_data.to_csv('time_series_data.csv', index=False)


import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules

# 1. 資料讀取與初步清理
# --------------------------------------------

# 讀取 CSV 檔案
data = pd.read_csv('supermarket_sales - Sheet1.csv')

# 檢查缺失值
missing_values = data.isnull().sum()
print("缺失值檢查：\n", missing_values)

# 移除重複值
data_cleaned = data.drop_duplicates()

# 2. 資料格式轉換
# --------------------------------------------

# 將 'Date' 欄位轉換為日期格式，並提取年、月、日
data_cleaned['Date'] = pd.to_datetime(data_cleaned['Date'], format='%m/%d/%Y')
data_cleaned['Year'] = data_cleaned['Date'].dt.year
data_cleaned['Month'] = data_cleaned['Date'].dt.month
data_cleaned['Day'] = data_cleaned['Date'].dt.day

# 將 'Time' 欄位轉換為時間格式，並提取小時、分鐘
# 檢查時間格式，嘗試多種格式進行轉換
try:
    data_cleaned['Time'] = pd.to_datetime(data_cleaned['Time'], format='%H:%M:%S')
except:
    data_cleaned['Time'] = pd.to_datetime(data_cleaned['Time'], format='%H:%M')

data_cleaned['Hour'] = data_cleaned['Time'].dt.hour
data_cleaned['Minute'] = data_cleaned['Time'].dt.minute

# 3. 創建消費特徵
# --------------------------------------------

# 計算總消費金額
data_cleaned['Total Spend'] = data_cleaned['Unit price'] * data_cleaned['Quantity']

# 4. 特徵工程
# --------------------------------------------

## 4.1 創建客戶標識

# 由於缺少客戶 ID，我們創建一個新的客戶標識
data_cleaned['Customer ID'] = data_cleaned['Customer type'] + '_' + data_cleaned['Gender'] + '_' + data_cleaned['Branch']

## 4.2 RFM 分析特徵

# 計算參考日期（資料集中最新的日期）
reference_date = data_cleaned['Date'].max()

# 計算 RFM 特徵
rfm = data_cleaned.groupby('Customer ID').agg({
    'Date': lambda x: (reference_date - x.max()).days,
    'Invoice ID': 'nunique',
    'Total Spend': 'sum'
}).rename(columns={'Date': 'Recency', 'Invoice ID': 'Frequency', 'Total Spend': 'Monetary'})

# Reset index to make 'Customer ID' a column instead of an index
rfm = rfm.reset_index()

# 處理 RFM 特徵中的缺失值
rfm[['Recency', 'Frequency', 'Monetary']] = rfm[['Recency', 'Frequency', 'Monetary']].fillna(0)

# 進行標準化
scaler = StandardScaler()
rfm[['Recency_scaled', 'Frequency_scaled', 'Monetary_scaled']] = scaler.fit_transform(
    rfm[['Recency', 'Frequency', 'Monetary']]
)

## 4.3 客戶分群

# 使用 K-Means 進行客戶分群
kmeans = KMeans(n_clusters=4, random_state=42)
rfm['Customer Segment'] = kmeans.fit_predict(rfm[['Recency_scaled', 'Frequency_scaled', 'Monetary_scaled']])

# 將包含 'Customer Segment' 的 rfm 合併回 data_cleaned
data_combined = pd.merge(data_cleaned, rfm[['Customer ID', 'Recency', 'Frequency', 'Monetary', 'Customer Segment']], on='Customer ID', how='left')

## 4.4 產品流行度特徵

# 在進行 One-Hot 編碼之前，計算產品流行度
product_popularity = data_combined.groupby('Product line')['Quantity'].sum().reset_index()
product_popularity.rename(columns={'Quantity': 'Total Quantity Sold'}, inplace=True)

# 合併回主資料集
data_combined = pd.merge(data_combined, product_popularity, on='Product line', how='left')

## 4.5 時間相關特徵

# 提取星期幾和是否為週末
data_combined['DayOfWeek'] = data_combined['Date'].dt.dayofweek
data_combined['IsWeekend'] = data_combined['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)

## 4.6 價格敏感度特徵

# 計算每個客戶的平均單價
customer_price_sensitivity = data_combined.groupby('Customer ID')['Unit price'].mean().reset_index()
customer_price_sensitivity.rename(columns={'Unit price': 'Avg Unit Price'}, inplace=True)

# 合併回主資料集
data_combined = pd.merge(data_combined, customer_price_sensitivity, on='Customer ID', how='left')

## 4.7 客戶忠誠度特徵

# 將會員身份轉換為二元變數
data_combined['IsMember'] = data_combined['Customer type'].apply(lambda x: 1 if x == 'Member' else 0)

## 4.8 交互特徵

# 創建價格和數量的交互項
data_combined['Price_Quantity_Interaction'] = data_combined['Unit price'] * data_combined['Quantity']

## 4.9 類別變數的 One-Hot 編碼

# 對類別變數進行 One-Hot 編碼
data_combined = pd.get_dummies(data_combined, columns=['Product line', 'Gender', 'Payment', 'City', 'Branch'])

# 5. 關聯規則挖掘（可選）
# --------------------------------------------

# 注意：在 One-Hot 編碼之前進行關聯規則挖掘

# 構建交易矩陣
basket = data_cleaned.groupby(['Invoice ID', 'Product line'])['Quantity'].sum().unstack().fillna(0)
basket = basket.applymap(lambda x: 1 if x > 0 else 0)

# 使用 Apriori 算法
frequent_itemsets = apriori(basket, min_support=0.01, use_colnames=True)
rules = association_rules(frequent_itemsets, num_itemsets=1, metric="lift", min_threshold=1)

# 6. 資料整合與清理
# --------------------------------------------

# 填補缺失值
data_combined.fillna(0, inplace=True)

# 刪除不需要的欄位
# 確認要刪除的欄位是否存在
columns_to_drop = ['Customer type', 'Date', 'Time', 'gross income', 'Tax 5%', 'cogs',
                   'gross margin percentage', 'Invoice ID']
existing_columns = data_combined.columns
columns_to_drop = [col for col in columns_to_drop if col in existing_columns]

final_data = data_combined.drop(columns_to_drop, axis=1)

# 7. 檢視處理後的資料
# --------------------------------------------

print("處理後的資料集：")
print(final_data.head())

# 8. 儲存處理後的資料
# --------------------------------------------

# 將最終資料集儲存為 CSV 檔案
final_data.to_csv('processed_supermarket_sales.csv', index=False)
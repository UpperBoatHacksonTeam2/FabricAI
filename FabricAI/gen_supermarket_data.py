import pandas as pd
import random

# 隨機生成模擬數據
def generate_random_data(num_records=100):
    data = []
    for _ in range(num_records):
        record = {
            "Invoice ID": f"{random.randint(100, 999)}-{random.randint(10, 99)}-{random.randint(1000, 9999)}",
            "Branch": random.choice(['A', 'B', 'C']),
            "City": random.choice(['Yangon', 'Naypyitaw', 'Mandalay']),
            "Customer type": random.choice(['Member', 'Normal']),
            "Gender": random.choice(['Male', 'Female']),
            "Product line": random.choice(['Health and beauty', 'Electronic accessories', 'Home and lifestyle', 'Sports and travel']),
            "Unit price": round(random.uniform(10, 100), 2),
            "Quantity": random.randint(1, 10),
            "Tax 5%": 0,  # 稅額稍後計算
            "Total": 0,   # 總價稍後計算
            "Date": f"{random.randint(1, 12)}/{random.randint(1, 28)}/2019",
            "Time": f"{random.randint(10, 23)}:{random.randint(10, 59)}",
            "Payment": random.choice(['Cash', 'Credit card', 'Ewallet']),
            "cogs": 0,   # 銷售成本稍後計算
            "gross margin percentage": 4.761904762,  # 假設為固定值
            "gross income": 0,  # 毛利潤稍後計算
            "Rating": round(random.uniform(4, 10), 1)
        }
        
        # 計算 "Tax 5%", "Total", "cogs", 和 "gross income"
        record["Tax 5%"] = round(record["Unit price"] * record["Quantity"] * 0.05, 2)
        record["Total"] = round(record["Unit price"] * record["Quantity"] + record["Tax 5%"], 2)
        record["cogs"] = round(record["Unit price"] * record["Quantity"], 2)
        record["gross income"] = round(record["Total"] - record["cogs"], 2)

        data.append(record)
    
    return pd.DataFrame(data)

# 生成 100 筆數據
df = generate_random_data(100)

# 將數據保存為 CSV 文件
output_path = 'generated_supermarket_sales.csv'
df.to_csv(output_path, index=False)
print(f"數據已保存至 {output_path}")
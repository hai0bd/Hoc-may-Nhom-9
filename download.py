from sklearn.datasets import load_wine
import pandas as pd

wine_data = load_wine()

# Tạo DataFrame với dữ liệu và nhãn
df = pd.DataFrame(data=wine_data.data, columns=wine_data.feature_names)
df['target'] = wine_data.target  # Thêm cột nhãn vào DataFrame

df.to_csv('wine_dataset.csv', index=False)
print("Dataset đã được lưu thành 'wine_dataset.csv'")

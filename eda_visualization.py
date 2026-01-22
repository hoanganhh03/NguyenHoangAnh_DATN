import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------
# 1. CẤU HÌNH HIỂN THỊ
# ------------------------------
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (8, 5)


DATA_FILE = "HaNoi_housing_dataset.csv"
df = pd.read_csv(DATA_FILE)

print("Kích thước dataset:", df.shape)
print("\n5 dòng đầu:")
print(df.head())


print("\nINFO:")
df.info()

print("\nDESCRIBE:")
print(df.describe())

plt.figure(figsize=(10,4))
sns.heatmap(df.isnull(), cbar=False)
plt.title("Missing Values Heatmap")
plt.show()

sns.histplot(df["Giá_m2"], bins=30, kde=True)
plt.title("Phân bố Giá / m²")
plt.xlabel("Triệu VNĐ / m²")
plt.ylabel("Count")
plt.show()

# Log-transform giá
df["Gia_log"] = np.log(df["Giá_m2"])

sns.histplot(df["Gia_log"], bins=30, kde=True)
plt.title("Phân bố log(Giá / m²)")
plt.xlabel("log(Giá)")
plt.show()


num_cols = [
    "ViTriXa_km",
    "TuoiNha",
    "Số tầng",
    "Số phòng ngủ",
    "Diện tích_m2"
]

for col in num_cols:
    sns.histplot(df[col], bins=25, kde=True)
    plt.title(f"Phân bố {col}")
    plt.show()

# ------------------------------
# 7. BOXPLOT (NGOẠI LAI)
# ------------------------------
for col in ["Giá_m2", "Diện tích_m2", "TuoiNha"]:
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot {col}")
    plt.show()

# ------------------------------
# 8. SO SÁNH GIÁ THEO BIẾN PHÂN LOẠI
# ------------------------------
plt.figure(figsize=(10,5))
sns.boxplot(x="Quận", y="Giá_m2", data=df)
plt.xticks(rotation=30)
plt.title("Giá / m² theo Quận")
plt.show()

sns.boxplot(x="Loại hình nhà ở", y="Giá_m2", data=df)
plt.title("Giá / m² theo Loại hình nhà ở")
plt.show()

# ------------------------------
# 9. SCATTER – MỐI QUAN HỆ VỚI GIÁ
# ------------------------------
sns.scatterplot(x="Diện tích_m2", y="Giá_m2", data=df, alpha=0.4)
plt.title("Diện tích vs Giá / m²")
plt.show()

sns.scatterplot(x="ViTriXa_km", y="Giá_m2", data=df, alpha=0.4)
plt.title("Khoảng cách vs Giá / m²")
plt.show()

# ------------------------------
# 10. CORRELATION HEATMAP
# ------------------------------
corr_cols = [
    "Giá_m2",
    "ViTriXa_km",
    "TuoiNha",
    "Số tầng",
    "Số phòng ngủ",
    "Diện tích_m2"
]

corr = df[corr_cols].corr()

plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# ------------------------------
# 11. COUNT PLOT – BIẾN PHÂN LOẠI
# ------------------------------
sns.countplot(x="Quận", data=df)
plt.xticks(rotation=30)
plt.title("Số lượng nhà theo Quận")
plt.show()

sns.countplot(x="Loại hình nhà ở", data=df)
plt.title("Số lượng theo Loại hình nhà ở")
plt.show()



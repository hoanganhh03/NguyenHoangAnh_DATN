import pandas as pd
import numpy as np
import pickle

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

# Load data
df = pd.read_csv("HaNoi_housing_dataset_30000.csv")

X = df.drop("Giá_m2", axis=1)
y = np.log(df["Giá_m2"])

num_cols = ["ViTriXa_km", "TuoiNha", "Số tầng", "Số phòng ngủ", "Diện tích_m2"]
cat_cols = ["Quận", "Loại hình nhà ở", "Giấy tờ pháp lý"]

preprocess = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])

model = Pipeline([
    ("preprocess", preprocess),
    ("model", RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    ))
])

model.fit(X, y)

pickle.dump(model, open("model_pipeline.pkl", "wb"))

print("✅ Đã lưu model_pipeline.pkl")

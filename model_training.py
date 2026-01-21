import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
import xgboost as xgb 
import pickle
from datetime import datetime
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# --- CẤU HÌNH: CHỌN MÔ HÌNH CHẠY & TẢI DỮ LIỆU ---
# Thay đổi giá trị này để chọn mô hình muốn chạy: 'ALL', 'LR', 'RF', 'XGB'
MODEL_TO_RUN = 'ALL' 

DATA_FILE = 'HaNoi_housing_dataset.csv'
CURRENT_YEAR = datetime.now().year

try:
    # Đọc file CSV, index_col=False để tránh đọc cột STT.
    df = pd.read_csv(DATA_FILE, encoding='utf-8', index_col=False) 
    print(f"Đã tải dữ liệu thành công. Kích thước ban đầu: {df.shape}")
except UnicodeDecodeError:
    try:
    
        df = pd.read_csv(DATA_FILE, encoding='cp1252', index_col=False)
        print("Đã tải dữ liệu thành công với encoding cp1252.")
    except Exception as e:
        print(f"LỖI TẢI DỮ LIỆU: {e}")
        exit()
except FileNotFoundError:
    print(f"LỖI: Không tìm thấy file dữ liệu {DATA_FILE}. Vui lòng kiểm tra lại.")
    exit()


# Bước 1: Loại bỏ khoảng trắng thừa từ TẤT CẢ tên cột
df.columns = df.columns.str.strip()
print(f"Tên cột SAU KHI STRIP: {df.columns.tolist()}")

try:
    # Lấy tên cột gốc tại Index 0
    original_col_0 = df.columns[0]
    
    # Gán tên chuẩn cho cột QUẬN (Vị trí 0)
    df.rename(columns={original_col_0: 'Quận'}, inplace=True)
    
    # Gán tên chuẩn cho cột Giá/m2 (Vị trí cuối cùng - Index 8)
    df.rename(columns={df.columns[-1]: 'Gia_m2'}, inplace=True)
    
    #Ánh xạ các cột đặc trưng còn lại theo vị trí 9 cột (Bắt đầu từ Index 1)
    if df.shape[1] >= 9:
        df.rename(columns={df.columns[1]: 'LoaiHinhNhaO',
                           df.columns[2]: 'GiayToPhapLy',
                           df.columns[3]: 'ViTriXa_km',
                           df.columns[4]: 'TuoiNha',
                           df.columns[5]: 'SoTang',
                           df.columns[6]: 'SoPhongNgu',
                           df.columns[7]: 'DienTich'}, inplace=True, errors='ignore')
    
    print(f"Tên cột sau khi ánh xạ cứng (9 cột): {df.columns.tolist()}")

except Exception as e:
    print(f"\nLỖI ÁNH XẠ TÊN CỘT: {e}")
    print("Vui lòng kiểm tra file CSV và đảm bảo cấu trúc 9 cột .")
    exit()


# Lọc dữ liệu: Chỉ giữ lại 3 quận trong phạm vi đề tài
TARGET_DISTRICTS = ['Quận Cầu Giấy', 'Quận Thanh Xuân', 'Quận Hà Đông']

if 'Quận' not in df.columns:
    print("\nLỖI KHÔNG TÌM THẤY CỘT 'Quận' SAU ÁNH XẠ.")
    exit()

df = df[df['Quận'].isin(TARGET_DISTRICTS)].copy()
print(f"Kích thước sau khi lọc 3 quận: {df.shape}")

# Loại bỏ các cột không cần thiết (Không còn cột Huyện)
# Định nghĩa các cột số học và phân loại
NUMERICAL_COLS = ['ViTriXa_km', 'TuoiNha', 'SoTang', 'SoPhongNgu', 'DienTich', 'Gia_m2']
CATEGORICAL_COLS = ['Quận', 'LoaiHinhNhaO', 'GiayToPhapLy']


# LÀM SẠCH VÀ CHUẨN HÓA DỮ LIỆU
# Xử lý các cột số: Loại bỏ ký tự không phải số và chuyển sang float
for col in NUMERICAL_COLS:
    if col in df.columns: 
        if df[col].dtype == 'object':
            # Loại bỏ ký tự không phải số hoặc dấu chấm (ví dụ: $ , )
            df[col] = df[col].astype(str).str.replace(r'[^\d.]', '', regex=True)
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Xử lý NaN (giá trị thiếu) - Điền khuyết
for col in NUMERICAL_COLS:
    if col in df.columns: 
        df[col].fillna(df[col].median(), inplace=True)

for col in CATEGORICAL_COLS:
    if col in df.columns:
        df[col].fillna(df[col].mode()[0], inplace=True)

# *******************************************************************
# BƯỚC TỐI ƯU HÓA: LỌC NGOẠI LAI VÀ LOG TRANSFORM
# *******************************************************************

# Lọc Ngoại lai cho các cột đặc trưng (Feature Outlier Clipping)
# VitriXa_km: Max 15km
df = df[df['ViTriXa_km'] <= 15].copy()
# TuoiNha: Max 50 năm
df = df[df['TuoiNha'] <= 50].copy()
# SoPhongNgu: Max 5 phòng
df = df[df['SoPhongNgu'] <= 5].copy()


max_price = df['Gia_m2'].max()
if max_price > 5000: 
    df['Gia_m2'] = df['Gia_m2'] / 1000000 

# LỌC NGOẠI LAI CỘT MỤC TIÊU (Giá/m2)
df = df[(df['Gia_m2'] >= 10) & (df['Gia_m2'] <= 150)].copy()

# 2. LOG TRANSFORM (Chuyển đổi Logarit để ổn định mô hình)
df['Gia_m2_log'] = np.log(df['Gia_m2'][df['Gia_m2'] > 0])
df['Gia_m2_log'].fillna(df['Gia_m2_log'].median(), inplace=True)

print(f"Kích thước sau khi lọc ngoại lai và Log Transform: {df.shape}")

# Mã hóa One-Hot Encoding cho các biến phân loại
OHE_COLS = [c for c in CATEGORICAL_COLS if c in df.columns]
df_ohe = pd.get_dummies(df, columns=OHE_COLS, drop_first=False, prefix=OHE_COLS)

# --- 3. Xác định Features và Target ---
TARGET_COL_NEW = 'Gia_m2_log' 
    
FEATURES = [col for col in df_ohe.columns if col not in [TARGET_COL_NEW, 'Gia_m2']] 
X = df_ohe[FEATURES]
y = df_ohe[TARGET_COL_NEW]

# Cột cần chuẩn hóa (Tất cả các cột số còn lại TRỪ cột mục tiêu)
SCALER_COLS = [col for col in X.columns if col in NUMERICAL_COLS and col != 'Gia_m2'] 

# --- 4. Chia tập dữ liệu Train/Test ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)

# --- 5. Chuẩn hóa dữ liệu (Scaling) ---
scaler = StandardScaler()

X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

if SCALER_COLS: # Chỉ scale nếu có cột số
    X_train_scaled[SCALER_COLS] = scaler.fit_transform(X_train[SCALER_COLS])
    X_test_scaled[SCALER_COLS] = scaler.transform(X_test[SCALER_COLS])


# --- 6. Huấn luyện và Đánh giá các Mô hình ---
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42, max_depth=8),
    "XGBoost Regressor": xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42, max_depth=5, learning_rate=0.1) 
}

# --- LỌC MÔ HÌNH DỰA TRÊN MODEL_TO_RUN ---
if MODEL_TO_RUN != 'ALL':
    models = {k: v for k, v in models.items() if MODEL_TO_RUN.upper() in k.upper()}
    if not models:
         print(f"\nCẢNH BÁO: Không tìm thấy mô hình cho lựa chọn '{MODEL_TO_RUN}'. Đang chạy tất cả.")
         MODEL_TO_RUN = 'ALL' 

best_r2 = -np.inf
best_model_name = ""
results = {}

print(f"\n--- KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH ({MODEL_TO_RUN}) ---")

for name, model in models.items():
    if 'Linear' in name:
        X_train_data, X_test_data = X_train_scaled, X_test_scaled
    else:

        X_train_data, X_test_data = X_train, X_test
        
    model.fit(X_train_data, y_train)
    y_pred_log = model.predict(X_test_data)
    
    y_test_original = np.exp(y_test)
    y_pred_original = np.exp(y_pred_log)
    
    r2 = r2_score(y_test_original, y_pred_original)
    mae = mean_absolute_error(y_test_original, y_pred_original)
    rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_original))
    
    results[name] = {'R2': r2, 'MAE': mae, 'RMSE': rmse}
    
    print(f"\n[Model: {name}]")
    print(f"  R2 Score (Độ khớp): {r2:.4f}")
    print(f"  RMSE (Sai số tuyệt đối): {rmse:.3f} Triệu/m2") 
    print(f"  MAE (Sai số trung bình): {mae:.3f} Triệu/m2")
    
    if r2 > best_r2:
        best_r2 = r2
        best_model_name = name
        best_model = model



if MODEL_TO_RUN == 'ALL' and best_r2 > -np.inf:
    MODEL_FILENAME = 'best_regressor_model.pkl'
    SCALER_FILENAME = 'scaler.pkl'
    OHE_FEATURES_FILENAME = 'ohe_features.pkl'

    with open(MODEL_FILENAME, 'wb') as file:
        pickle.dump(best_model, file)

    with open(SCALER_FILENAME, 'wb') as file:
        pickle.dump(scaler, file)

    with open(OHE_FEATURES_FILENAME, 'wb') as file:
        # Lưu danh sách features đã được OHE (X.columns)
        ohe_features_list = X.columns.tolist()
        pickle.dump(ohe_features_list, file)

    print("\n-------------------------------------------------")
    print(f"Mô hình TỐI ƯU được chọn là: {best_model_name} (R2={best_r2:.4f})")
    print(f"Mô hình đã được lưu vào: {MODEL_FILENAME}")
    print("Bây giờ bạn có thể chạy app.py để triển khai ứng dụng web.")
elif MODEL_TO_RUN != 'ALL':
    print("\nĐã chạy xong mô hình riêng biệt. Không lưu file .pkl.")
else:
     print("\nCẢNH BÁO: Không có mô hình nào đạt R2 > -inf. Không lưu file .pkl.")
     

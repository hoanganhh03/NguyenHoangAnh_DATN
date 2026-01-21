import streamlit as st
import pandas as pd
import numpy as np
import pickle

def ai_explain(row):
    reasons = []

    if row["ViTriXa_km"] <= 5:
        reasons.append("📍 Gần trung tâm nên giá cao hơn")
    else:
        reasons.append("📍 Xa trung tâm nên giá thấp hơn")

    if row["Diện tích_m2"] >= 90:
        reasons.append("📐 Diện tích lớn")
    elif row["Diện tích_m2"] <= 50:
        reasons.append("📐 Diện tích nhỏ")

    if row["Số phòng ngủ"] >= 3:
        reasons.append("🛏️ Nhiều phòng ngủ")

    if row["TuoiNha"] <= 5:
        reasons.append("🏗️ Nhà mới")

    return " | ".join(reasons)


def price_alert(price):
    if price < 110:
        return "🟢 GIÁ THẤP – Có thể mua", "success"
    elif price < 140:
        return "🟡 GIÁ TRUNG BÌNH – Cân nhắc", "warning"
    else:
        return "🔴 GIÁ CAO – Không khuyến nghị", "error"


district_avg_price = {
    "Quận Hà Đông": 100,
    "Quận Cầu Giấy": 110,
    "Quận Thanh Xuân": 118,
    "Quận Đống Đa": 142,
    "Quận Hai Bà Trưng": 130,
    "Quận Long Biên": 100,
    "Quận Tây Hồ": 160
}

st.set_page_config(
    page_title="Dự đoán giá nhà chung cư",
    page_icon="🏠",
    layout="wide"
)

@st.cache_resource
def load_model():
    return pickle.load(open("model_pipeline.pkl", "rb"))

model = load_model()

st.title("🏠 Dự đoán giá nhà chung cư Hà Nội")

tab1, tab2 = st.tabs(["🔮 Dự đoán", "📂 Import CSV"])

# ================= TAB 1 =================
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        district = st.selectbox(
            "Quận",
            list(district_avg_price.keys())
        )
        house_type = st.selectbox(
            "Loại hình nhà ở",
            ["Chung cư", "Duplex", "Penthouse"]
        )
        legal = st.selectbox(
            "Giấy tờ pháp lý",
            ["Sổ hồng", "Đã có sổ"]
        )
        distance = st.number_input(
        "Khoảng cách tới trung tâm (km)",
        min_value=0.0,
        max_value=50.0,
        value=1.5,
        step=0.5
        )


    with col2:
        age = st.slider("Tuổi nhà", 0, 40, 10)
        floors = st.slider("Số tầng", 3, 40, 10)
        bedrooms = st.slider("Số phòng ngủ", 1, 5, 2)
        area = st.slider("Diện tích (m²)", 30, 200, 70)

    if st.button("🚀 Dự đoán ngay"):
        input_df = pd.DataFrame([{
            "Quận": district,
            "Loại hình nhà ở": house_type,
            "Giấy tờ pháp lý": legal,
            "ViTriXa_km": distance,
            "TuoiNha": age,
            "Số tầng": floors,
            "Số phòng ngủ": bedrooms,
            "Diện tích_m2": area
        }])

        log_price = model.predict(input_df)[0]
        price = np.exp(log_price)

        # Lưu state
        st.session_state["price"] = price
        st.session_state["input_df"] = input_df

# ===== HIỂN THỊ KẾT QUẢ =====
if "price" in st.session_state:
    price = st.session_state["price"]
    input_df = st.session_state["input_df"]

    st.markdown("## 💰 Kết quả dự đoán")
    st.metric(
        label="Giá nhà dự đoán (triệu VNĐ / m²)",
        value=f"{price:.2f}"
    )

    # ===== CẢNH BÁO =====
    alert_text, alert_type = price_alert(price)
    getattr(st, alert_type)(alert_text)

    # ===== SO SÁNH KHU VỰC =====
    avg_price = district_avg_price.get(input_df.iloc[0]["Quận"])
    if price < avg_price:
        st.success(f"📊 Thấp hơn, giá trung bình {avg_price} triệu/m²")
    else:
        st.error(f"📊 Cao hơn, giá trung bình {avg_price} triệu/m²")

    # =====  GIẢI THÍCH GIÁ =====
    st.markdown("## phân tích giá")
    st.info(ai_explain(input_df.iloc[0]))

# ================= TAB 2 =================
with tab2:
    st.subheader("📂 Dự đoán giá từ file CSV")

    uploaded_file = st.file_uploader("📎 Tải file CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        st.markdown("### 📄 Dữ liệu đầu vào")
        st.dataframe(df.head())

        preds_log = model.predict(df)
        df["Giá_dự_đoán_m2"] = np.exp(preds_log).round(2)

        st.markdown("### ✅ Kết quả dự đoán")
        st.dataframe(df.head())

        st.markdown("### 📊 Phân phối giá dự đoán")
        st.bar_chart(df["Giá_dự_đoán_m2"])

        st.markdown("### 📊 Giá trung bình theo quận")
        avg_by_district = df.groupby("Quận")["Giá_dự_đoán_m2"].mean()
        st.bar_chart(avg_by_district)

        csv = df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "⬇️ Tải file kết quả",
            csv,
            "ket_qua_du_doan.csv",
            "text/csv"
        )

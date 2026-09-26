from pathlib import Path
import pickle
from datetime import date, datetime
import os
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models/best_model_xgboost.pkl"
TRAIN_PATH = PROJECT_ROOT / "data/processed/train.csv"
APP_TIMEZONE = ZoneInfo("Asia/Ho_Chi_Minh")


@st.cache_resource
def load_model_bundle():
    with MODEL_PATH.open("rb") as model_file:
        return pickle.load(model_file)


@st.cache_data
def load_training_defaults():
    train_df = pd.read_csv(TRAIN_PATH)
    return {
        "windspeed": float(train_df["windspeed"].median()),
        "lag_1": float(train_df["lag_1"].median()),
        "lag_2": float(train_df["lag_2"].median()),
        "lag_24": float(train_df["lag_24"].median()),
        "lag_168": float(train_df["lag_168"].median()),
    }


def get_weather_api_key():
    """Read the OpenWeather key without placing it in source control."""
    try:
        return st.secrets.get("OPENWEATHER_API_KEY") or os.getenv("OPENWEATHER_API_KEY")
    except FileNotFoundError:
        return os.getenv("OPENWEATHER_API_KEY")


@st.cache_data(ttl=600, show_spinner=False)
def fetch_current_weather(city, api_key):
    response = requests.get(
        "https://api.openweathermap.org/data/2.5/weather",
        params={"q": city, "appid": api_key, "units": "metric", "lang": "vi"},
        timeout=10,
    )
    response.raise_for_status()
    payload = response.json()
    weather_id = int(payload["weather"][0]["id"])
    if weather_id in (800, 801):
        weather_class = 1
    elif 200 <= weather_id < 700:
        weather_class = 3
    else:
        weather_class = 2
    return {
        "temperature": float(payload["main"]["temp"]),
        "humidity": int(payload["main"]["humidity"]),
        "weather": weather_class,
        "description": payload["weather"][0]["description"].capitalize(),
        "city": payload["name"],
    }


def build_input_row(selected_date, hour, temperature_c, humidity_percent, weather, workingday, holiday, feature_columns):
    timestamp = pd.Timestamp(selected_date) + pd.Timedelta(hours=hour)
    hour = timestamp.hour
    month = timestamp.month
    year = timestamp.year
    day_of_week = timestamp.dayofweek
    season = {1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2, 7: 3, 8: 3, 9: 3, 10: 4, 11: 4, 12: 4}[month]
    normalized_temperature = temperature_c / 41.0
    normalized_humidity = humidity_percent / 100.0
    defaults = load_training_defaults()

    values = {
        "yr": int(year >= 2012),
        "mnth": month,
        "hr": hour,
        "holiday": holiday,
        "weekday": day_of_week,
        "workingday": workingday,
        "season": season,
        "temp": normalized_temperature,
        "atemp": normalized_temperature,
        "hum": normalized_humidity,
        "windspeed": defaults["windspeed"],
        "hour": hour,
        "day": timestamp.day,
        "month": month,
        "year": year,
        "day_of_week": day_of_week,
        "day_of_year": timestamp.dayofyear,
        "is_weekend": int(day_of_week >= 5),
        "is_workingday": workingday,
        "rush_hour": int(hour in [7, 8, 9, 16, 17, 18, 19]),
        "weathersit": weather,
        **{name: defaults[name] for name in ["lag_1", "lag_2", "lag_24", "lag_168"]},
    }
    return pd.DataFrame([[values[column] for column in feature_columns]], columns=feature_columns)


def build_hourly_predictions(selected_date, temperature_c, humidity_percent, weather, workingday, holiday, model, feature_columns):
    hourly_rows = pd.concat(
        [build_input_row(selected_date, hour, temperature_c, humidity_percent, weather, workingday, holiday, feature_columns) for hour in range(24)],
        ignore_index=True,
    )
    hourly_predictions = model.predict(hourly_rows)
    return pd.DataFrame({"Giờ": range(24), "Số lượng dự báo": [max(0.0, float(value)) for value in hourly_predictions]})


def build_next_24h_predictions(start_time, temperature_c, humidity_percent, weather, workingday, holiday, model, feature_columns):
    timestamps = [pd.Timestamp(start_time) + pd.Timedelta(hours=offset) for offset in range(24)]
    hourly_rows = pd.concat(
        [build_input_row(timestamp.date(), timestamp.hour, temperature_c, humidity_percent, weather, workingday, holiday, feature_columns) for timestamp in timestamps],
        ignore_index=True,
    )
    predictions = model.predict(hourly_rows)
    return pd.DataFrame({"Thời gian": timestamps, "Giờ": [timestamp.strftime("%H:%M") for timestamp in timestamps], "Số lượng dự báo": [max(0.0, float(value)) for value in predictions]})


st.set_page_config(page_title="Bike Demand Prediction", page_icon="🚲", layout="wide")
st.title("Bike Demand Prediction")
st.caption("Dự báo bằng mô hình XGBoost đã lưu và dữ liệu thời tiết từ OpenWeatherMap")

if not MODEL_PATH.exists() or not TRAIN_PATH.exists():
    st.error("Model or training data is missing. Run PHASE 11 first.")
    st.stop()

bundle = load_model_bundle()
model = bundle["model"]
feature_columns = bundle["feature_columns"]

input_column, result_column = st.columns([0.8, 1.2], gap="large")
with input_column:
    st.subheader("Thông tin đầu vào")
    current_time = datetime.now(APP_TIMEZONE).replace(second=0, microsecond=0)
    selected_date = st.date_input(
        "Ngày bắt đầu dự báo",
        value=current_time.date(),
        help="Mặc định là ngày hiện tại, nhưng bạn có thể chọn ngày khác.",
    )
    st.caption(f"Giờ bắt đầu được lấy tự động theo giờ hiện tại: {current_time:%H:%M}.")
    hour = current_time.hour
    city = st.text_input("Địa điểm lấy thời tiết", value="Hanoi")
    api_key = get_weather_api_key()
    if not api_key:
        st.error("Chưa cấu hình OPENWEATHER_API_KEY. Local: .streamlit/secrets.toml · Streamlit Cloud: Manage app → Settings → Secrets.")

    weather_data = st.session_state.get("weather_data", {})
    city_query = city.strip()
    should_refresh_weather = bool(api_key) and (
        not weather_data or weather_data.get("query") != city_query
    )
    if should_refresh_weather and city_query:
        try:
            weather_data = fetch_current_weather(city_query, api_key)
            weather_data["query"] = city_query
            st.session_state["weather_data"] = weather_data
        except requests.RequestException:
            st.warning(f"Không thể tự động lấy thời tiết cho {city}. Bạn có thể nhập thông số thủ công.")

    if st.button("Lấy thời tiết hiện tại"):
        if not api_key:
            st.error("Hãy thêm OPENWEATHER_API_KEY vào Secrets của Streamlit Cloud hoặc .streamlit/secrets.toml khi chạy local.")
        else:
            try:
                weather_data = fetch_current_weather(city_query, api_key)
                weather_data["query"] = city_query
                st.session_state["weather_data"] = weather_data
            except requests.RequestException:
                st.error(f"Không thể lấy dữ liệu thời tiết cho {city}. Hãy kiểm tra tên thành phố và API key.")

    weather_data = st.session_state.get("weather_data", weather_data)
    if weather_data:
        st.success(f"Thời tiết tại {weather_data['city']}: {weather_data['description']}")
    temperature_c = st.number_input("Nhiệt độ (°C)", min_value=-20.0, max_value=45.0, value=weather_data.get("temperature", 20.0), step=0.5)
    humidity_percent = st.slider("Độ ẩm (%)", min_value=0, max_value=100, value=weather_data.get("humidity", 60))
    weather = st.selectbox(
        "Điều kiện thời tiết",
        options=[1, 2, 3, 4],
        index=max(0, weather_data.get("weather", 1) - 1),
        format_func=lambda value: {1: "Trời quang / ít mây", 2: "Sương mù / nhiều mây", 3: "Mưa hoặc tuyết nhẹ", 4: "Mưa hoặc tuyết lớn"}[value],
    )
    workingday = int(st.checkbox("Ngày làm việc", value=True))
    holiday = int(st.checkbox("Ngày lễ", value=False))
    predict_clicked = st.button("Dự đoán nhu cầu", type="primary", use_container_width=True)

with result_column:
    st.subheader("Kết quả dự đoán")
    if predict_clicked:
        forecast_start = datetime.combine(selected_date, current_time.time(), tzinfo=APP_TIMEZONE)
        hourly_data = build_next_24h_predictions(forecast_start, temperature_c, humidity_percent, weather, workingday, holiday, model, feature_columns)
        prediction = hourly_data.iloc[0]["Số lượng dự báo"]
        st.metric("Nhu cầu tại thời điểm bắt đầu", f"{prediction:,.0f} lượt thuê")
        chart = go.Figure()
        chart.add_trace(go.Scatter(x=hourly_data["Thời gian"], y=hourly_data["Số lượng dự báo"], mode="lines+markers", name="Nhu cầu dự báo", line=dict(color="#ff4b4b", width=3), marker=dict(size=7)))
        chart.add_trace(go.Scatter(x=[hourly_data.iloc[0]["Thời gian"]], y=[prediction], mode="markers", name="Thời điểm hiện tại", marker=dict(color="#00c2a8", size=15, line=dict(color="white", width=2))))
        chart.update_layout(title=f"Nhu cầu dự báo trong 24 giờ tới - bắt đầu {forecast_start:%d/%m %H:%M}", xaxis_title="Thời gian (giờ)", yaxis_title="Số lượng lượt thuê", hovermode="x unified", height=500, margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(chart, use_container_width=True)
        st.caption("Biểu đồ thể hiện dự báo từ thời điểm hiện tại đến 24 giờ tiếp theo; điểm màu xanh là thời điểm bắt đầu.")
    else:
        st.info("Kiểm tra thông tin thời tiết ở bên trái rồi bấm Dự đoán nhu cầu để xem 24 giờ tiếp theo.")

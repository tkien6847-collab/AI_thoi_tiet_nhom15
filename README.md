# Bike Demand Prediction

Project Machine Learning dự báo nhu cầu thuê xe theo thời gian bằng dữ liệu Bike Sharing theo giờ.

## Mục tiêu

- Dự đoán tổng số lượt thuê xe (`cnt`).
- Tạo đặc trưng thời gian, thời tiết, lag và rolling.
- So sánh Linear Regression, Random Forest, XGBoost và GRU.
- Đánh giá theo đúng thứ tự thời gian và tránh data leakage.
- Chạy được trên CPU.

## Dataset

Dataset hiện tại là `data/raw/hour.csv`, gồm dữ liệu theo giờ từ `2011-01-01` đến `2012-12-31`.

Các cột chính:

- Thời gian: `dteday`, `hr`, `weekday`, `mnth`, `yr`.
- Lịch: `holiday`, `workingday`, `season`.
- Thời tiết: `weathersit`, `temp`, `atemp`, `hum`, `windspeed`.
- Target: `cnt`.
- `casual` và `registered` là các thành phần của `cnt`, không được dùng làm feature.

Nếu thay dataset, đặt file mới tại `data/raw/hour.csv` và giữ schema tương ứng.

## Cấu trúc

```text
bike-demand-prediction/
├── app/app.py
├── data/
│   ├── raw/hour.csv
│   └── processed/
├── models/
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Feature_Engineering.ipynb
│   ├── 03_Time_Series_Split.ipynb
│   ├── 04_Regression.ipynb
│   ├── 05_RandomForest.ipynb
│   ├── 06_XGBoost.ipynb
│   ├── 07_LSTM_GRU.ipynb
│   ├── 08_Model_Comparison.ipynb
│   ├── 09_Error_Analysis.ipynb
│   ├── 10_Ablation_Study.ipynb
│   └── 11_Model_Saving.ipynb
├── results/
│   ├── figures/
│   └── metrics/
├── requirements.txt
└── README.md
```

Chạy notebook từ thư mục gốc `D:\bike-demand-prediction` để các đường dẫn output nhất quán.

## Cài đặt

Yêu cầu Python 3.11.

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
py -3.11 -m pip install -r requirements.txt
```

Trong VS Code, chọn kernel Python 3.11 cho các notebook.

## Quy trình chạy

Chạy tuần tự các notebook:

1. `01_EDA.ipynb`: kiểm tra dữ liệu và trực quan hóa, không train model.
2. `02_Feature_Engineering.ipynb`: tạo feature thời gian, lag và rolling.
3. `03_Time_Series_Split.ipynb`: chia train/validation/test theo tỷ lệ 70/15/15.
4. `04_Regression.ipynb`: Linear Regression baseline.
5. `05_RandomForest.ipynb`: Random Forest và feature importance.
6. `06_XGBoost.ipynb`: XGBoost CPU-only.
7. `07_LSTM_GRU.ipynb`: GRU với sequence length 24 và EarlyStopping.
8. `08_Model_Comparison.ipynb`: so sánh các model.
9. `09_Error_Analysis.ipynb`: phân tích lỗi theo nhóm thời gian.
10. `10_Ablation_Study.ipynb`: đánh giá đóng góp của từng nhóm feature.
11. `11_Model_Saving.ipynb`: fit model cuối và lưu model tốt nhất.

## Chống data leakage

- Không dùng `train_test_split` hoặc shuffle.
- Split theo thứ tự thời gian.
- Rolling dùng `target.shift(1)` trước khi tính trung bình.
- Scaler của GRU chỉ fit trên train.
- Hyperparameter được chọn bằng validation.
- Test chỉ dùng cho đánh giá cuối.
- `casual`, `registered` và target `cnt` bị loại khỏi predictors.
- PHASE 8 chọn model bằng validation RMSE; test chỉ được dùng để báo cáo.

## Kết quả

Bảng PHASE 8 trên test trước khi fit lại model cuối:

| Model | MAE | RMSE | R² | Training Time |
|---|---:|---:|---:|---:|
| XGBoost | 35.91 | 58.95 | 0.9241 | 0.95 s |
| Random Forest | 37.36 | 62.73 | 0.9140 | 1.97 s |
| GRU | 48.27 | 69.09 | 0.8957 | 27.84 s |
| Linear Regression | 57.79 | 83.94 | 0.8461 | 0.05 s |

Feature set tốt nhất trong ablation là **time + weather + lag**. Model final được fit lại trên train + validation và đạt trên test:

- MAE: `29.15`
- RMSE: `47.75`
- R²: `0.9502`

Model final: `models/best_model_xgboost.pkl`.

## Chạy Streamlit

Sau khi chạy PHASE 11:

```powershell
py -3.11 -m streamlit run app/app.py
```

Mở `http://localhost:8501`.

App nhận ngày, giờ, nhiệt độ, độ ẩm, thời tiết và working day. Vì giao diện đơn giản không yêu cầu lag và windspeed, hai nhóm giá trị này dùng median từ training data. Với hệ thống production, nên truyền demand lịch sử thực tế để tạo lag thay vì dùng median fallback.

## Giới hạn hiện tại

- Dataset có một số khoảng trống timestamp. PHASE 2 hiện dùng `shift(1)`, `shift(2)`, `shift(24)` và `shift(168)` theo thứ tự bản ghi; vì vậy tên lag thể hiện số quan sát trước đó, không đảm bảo đúng số giờ lịch. Nếu cần forecast theo đúng 24/168 giờ, hãy resample theo hourly timestamp và xử lý missing timestamp trước khi tạo lag.
- Các metrics với lag là đánh giá one-step-ahead: demand quá khứ được giả định đã quan sát khi tạo lag cho bước kế tiếp. Forecast nhiều bước cần đánh giá recursive hoặc direct horizon riêng.
- App giới hạn ngày trong miền dataset 2011–2012. Lag và windspeed dùng median chỉ là fallback cho demo; production nên nhận lịch sử demand và biến thời tiết đầy đủ.

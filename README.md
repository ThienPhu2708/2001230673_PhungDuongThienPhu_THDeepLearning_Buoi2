# 2001230673_PhungDuongThienPhu_THDeepLearning_Buoi2
Bài tập thực hành DeepLearning buổi 2

# 🏠 Data Preprocessing & Deep Learning — HOMES Dataset

> **Bài thực hành:** Chuẩn hóa dữ liệu đúng cách để chống Data Leakage, kết hợp mô hình hồi quy với TensorFlow/Keras.

---

## 📌 Mô tả

Dự án này thực hành quy trình tiền xử lý dữ liệu chuẩn trong Machine Learning, áp dụng trên dataset giá nhà (`23_HOMES.csv`). Trọng tâm là đảm bảo **không xảy ra Data Leakage** trong quá trình scaling và encoding, sau đó xây dựng mô hình hồi quy bằng **TensorFlow/Keras** để dự đoán giá nhà.

---

## 🎯 Mục tiêu

- Hiểu và áp dụng đúng **Golden Rule**: `Split first → Fit on train → Transform val/test → Test only ONCE`
- Phân biệt và sử dụng đúng các kỹ thuật preprocessing:
  - **Min-Max Scaling** cho numeric features
  - **One-Hot Encoding** cho categorical nominal features
  - **Label/Ordinal Encoding** cho ordered features hoặc target labels
- Xây dựng pipeline ML hoàn chỉnh với `ColumnTransformer` + `TensorFlow/Keras`
- Nhận biết và tránh các lỗi preprocessing phổ biến

---

## 📂 Phạm vi

| Hạng mục | Chi tiết |
|---|---|
| Dataset | `23_HOMES.csv` — 41 mẫu dữ liệu nhà ở |
| Task | Regression — dự đoán giá nhà |
| Preprocessing | MinMaxScaler, OneHotEncoder (scikit-learn) |
| Model | Neural Network — TensorFlow/Keras Sequential |
| Kỹ thuật | Dropout, EarlyStopping, ColumnTransformer Pipeline |
| Môi trường | Google Colab / Python 3.x |

---

## 🛠️ Công nghệ sử dụng

![Python](https://img.shields.io/badge/Python-3.x-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-green)
![Pandas](https://img.shields.io/badge/Pandas-latest-lightgrey)

---

## 🔄 Quy trình thực hiện

```
Raw Data (23_HOMES.csv)
        │
        ▼
1. Khám phá dữ liệu (EDA)
        │
        ▼
2. Split FIRST — train_test_split()
        │
        ├──── X_train ──────────────────────────────┐
        │         │                                  │
        │         ▼                                  │
        │   3. Fit Preprocessor                      │
        │      - MinMaxScaler (numeric)              │
        │      - OneHotEncoder (categorical)         │
        │         │                                  │
        │         ▼                                  ▼
        │   4. Transform Train          4. Transform Test
        │         │                                  │
        └─────────┴──────────────────────────────────┘
                  │
                  ▼
        5. Build Keras Model
           Input → Dense(32) → Dropout → Dense(16) → Dropout → Dense(1)
                  │
                  ▼
        6. Train (validation_split=0.2, EarlyStopping)
                  │
                  ▼
        7. Evaluate TEST SET — chỉ 1 lần duy nhất
```

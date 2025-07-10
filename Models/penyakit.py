import joblib

# Ganti path ini sesuai lokasi file .pkl kamu
model_path = "D:/FAIZ/KULIAH/programming/DATATHON/aaaa/Models/logistic_regression_model.pkl"
encoder_path = "D:/FAIZ/KULIAH/programming/DATATHON/aaaa/Models/encoder.pkl"

# Muat model dan encoder
model = joblib.load(model_path)
encoder = joblib.load(encoder_path)

# ──────────── Informasi model ────────────
print("📌 Tipe model:", type(model).__name__)
print("\n📌 Parameter model:")
print(model.get_params())

# ──────────── Fitur input ────────────
if hasattr(model, "feature_names_in_"):
    print("\n📌 Nama fitur yang digunakan:")
    for f in model.feature_names_in_:
        print("-", f)
else:
    print("\n❗ Model tidak menyimpan nama fitur (feature_names_in_)")

# ──────────── Koefisien model ────────────
if hasattr(model, "coef_"):
    print("\n📌 Koefisien model:")
    print(model.coef_)
else:
    print("\n❗ Model tidak memiliki atribut 'coef_'")

# ──────────── Label klasifikasi ────────────
if hasattr(model, "classes_"):
    print("\n📌 Label kelas yang diprediksi:")
    print(encoder.inverse_transform(model.classes_))  # gunakan encoder agar readable
else:
    print("\n❗ Model tidak memiliki atribut 'classes_'")

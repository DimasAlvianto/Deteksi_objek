# Deteksi_objek
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile

# Setup halaman
st.set_page_config(page_title="Deteksi Objek", layout="centered")
st.title("📸 Deteksi Objek")

st.markdown("""
Upload gambar untuk mendeteksi objek secara otomatis menggunakan YOLOv8n.  
Model yang digunakan sudah pre-trained untuk mendeteksi objek umum seperti orang, mobil, hewan, dll.
""")

# Model YOLOv8n saja
MODEL_PATH = "yolov8n.pt"

# Fungsi deteksi gambar
def detect_image(image_path, model_path):
    model = YOLO(model_path)
    results = model(image_path)
    for r in results:
        return r.plot()

# Upload gambar
st.subheader("🖼️ Upload Gambar")
uploaded_image = st.file_uploader("Unggah gambar (jpg/png)", type=["jpg", "jpeg", "png"])

if uploaded_image:
    # Simpan sementara gambar
    temp_image = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    temp_image.write(uploaded_image.read())
    st.image(temp_image.name, caption="Gambar Asli", use_column_width=True)

    # Deteksi otomatis
    with st.spinner("🔎 Mendeteksi objek..."):
        result_img = detect_image(temp_image.name, MODEL_PATH)
        st.image(result_img, caption="Hasil Deteksi", use_column_width=True)

    st.success("✅ Deteksi selesai! Objek pada gambar sudah diberi kotak dan label.")

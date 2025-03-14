
import streamlit as st
import joblib
import tensorflow as tf
import numpy as np
import requests
import io
from PIL import Image

st.set_page_config(page_title="AI Project", layout="centered")

# โหลดโมเดลจาก Hugging Face URL
@st.cache_resource
def load_ml_model():
    url = 'https://huggingface.co/datasets/NoahINT/AI-Model/resolve/main/heart_disease_model.pkl'
    response = requests.get(url)
    return joblib.load(io.BytesIO(response.content))

@st.cache_resource
def load_scaler():
    url = 'https://huggingface.co/datasets/NoahINT/AI-Model/resolve/main/heart_disease_scaler.pkl'
    response = requests.get(url)
    return joblib.load(io.BytesIO(response.content))

@st.cache_resource
def load_cnn_model():
    url = 'https://huggingface.co/datasets/NoahINT/AI-Model/resolve/main/cnn_cifar10_model.h5'
    model_path = '/tmp/cnn_cifar10_model.h5'
    with open(model_path, 'wb') as f:
        f.write(requests.get(url).content)
    return tf.keras.models.load_model(model_path)

# โหลดโมเดล
ml_model = load_ml_model()
scaler = load_scaler()
cnn_model = load_cnn_model()

# เมนู
menu = ["หน้าหลัก", "ทำนายโรคหัวใจ (ML)", "จำแนกรูปภาพ (CNN)", "เกี่ยวกับโครงการ"]
choice = st.sidebar.selectbox("เลือกเมนู", menu)

# หน้าหลัก
if choice == "หน้าหลัก":
    st.title("โครงการ AI: ทำนายโรคหัวใจ และ จำแนกรูปภาพ")
    st.write("เว็บนี้ใช้ Machine Learning และ Deep Learning เพื่อวิเคราะห์ข้อมูลและรูปภาพ")

# หน้า ML
elif choice == "ทำนายโรคหัวใจ (ML)":
    st.header("ทำนายโรคหัวใจ (Machine Learning)")
    age = st.number_input("อายุ", min_value=1, max_value=120, value=30)
    sex = st.selectbox("เพศ", options=[0, 1], format_func=lambda x: "ชาย" if x == 1 else "หญิง")
    cp = st.number_input("CP (อาการเจ็บหน้าอก)", 0, 3, 0)
    trestbps = st.number_input("ความดันโลหิตขณะพัก", 80, 200, 120)
    chol = st.number_input("คอเลสเตอรอล", 100, 400, 200)
    if st.button("ทำนาย"):
        X_new = scaler.transform([[age, sex, cp, trestbps, chol]])
        result = ml_model.predict(X_new)[0]
        st.success("ผลลัพธ์: " + ("มีความเสี่ยงโรคหัวใจ" if result == 1 else "ไม่มีความเสี่ยง"))

# หน้า CNN
elif choice == "จำแนกรูปภาพ (CNN)":
    st.header("จำแนกรูปภาพ (Deep Learning)")
    uploaded_file = st.file_uploader("อัปโหลดรูปภาพ (32x32)", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file).resize((32, 32))
        st.image(image, caption="รูปภาพที่อัปโหลด", use_column_width=True)
        img_array = np.array(image) / 255.0
        prediction = cnn_model.predict(np.expand_dims(img_array, axis=0))
        classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        st.success(f"ผลลัพธ์: {classes[np.argmax(prediction)]}")

# หน้าเกี่ยวกับ
elif choice == "เกี่ยวกับโครงการ":
    st.header("เกี่ยวกับโครงการ")
    st.write("โครงการนี้เป็นส่วนหนึ่งของการเรียนรู้ AI โดยใช้ Machine Learning และ Deep Learning")

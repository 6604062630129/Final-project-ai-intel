
import streamlit as st
import joblib
import tensorflow as tf
import numpy as np
from PIL import Image

ml_model = joblib.load('models/heart_disease_model.pkl')
ml_scaler = joblib.load('models/heart_disease_scaler.pkl')
cnn_model = tf.keras.models.load_model('models/cnn_cifar10_model.h5')

menu = ["หน้าหลัก", "ทำนายโรคหัวใจ (ML)", "จำแนกรูปภาพ (CNN)", "เกี่ยวกับโครงการ"]
choice = st.sidebar.selectbox("เลือกหน้า", menu)

if choice == "หน้าหลัก":
    st.title("โครงการ AI จำแนกข้อมูลและภาพ")
    st.write("โครงการนี้ใช้ Machine Learning และ Neural Network ในการจำแนกโรคหัวใจและรูปภาพ CIFAR-10")

elif choice == "ทำนายโรคหัวใจ (ML)":
    st.header("ทำนายโรคหัวใจด้วย Machine Learning")
    age = st.number_input('อายุ', 1, 120)
    sex = st.selectbox('เพศ (ชาย=1, หญิง=0)', [0, 1])
    cp = st.number_input('CP (ประเภทอาการเจ็บหน้าอก)', 0, 3)
    trestbps = st.number_input('ความดันโลหิตขณะพัก', 90, 200)
    chol = st.number_input('คอเลสเตอรอล', 100, 400)
    if st.button('ทำนาย'):
        X_new = ml_scaler.transform([[age, sex, cp, trestbps, chol]])
        result = ml_model.predict(X_new)[0]
        st.success("ผลลัพธ์: " + ("มีความเสี่ยงโรคหัวใจ" if result == 1 else "ไม่มีความเสี่ยง"))

elif choice == "จำแนกรูปภาพ (CNN)":
    st.header("จำแนกรูปภาพด้วย CNN")
    uploaded_file = st.file_uploader("อัปโหลดรูป (32x32)", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file).resize((32, 32))
        st.image(image, caption='รูปที่อัปโหลด', use_column_width=True)
        img_array = np.array(image) / 255.0
        prediction = cnn_model.predict(np.expand_dims(img_array, axis=0))
        class_idx = np.argmax(prediction)
        classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        st.success(f"ผลลัพธ์: {classes[class_idx]}")

elif choice == "เกี่ยวกับโครงการ":
    st.header("เกี่ยวกับโครงการ")
    st.write("โครงการนี้ใช้ AI ในการจำแนกข้อมูลและรูปภาพ เพื่องานจริง")

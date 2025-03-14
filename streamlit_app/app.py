
import streamlit as st
import requests
import numpy as np
from PIL import Image

st.set_page_config(page_title="AI Project")

menu = ["หน้าหลัก", "ทำนายโรคหัวใจ", "จำแนกรูปภาพ"]
choice = st.sidebar.selectbox("เลือกเมนู", menu)

API_URL = "https://final-ai-api.onrender.com"

if choice == "หน้าหลัก":
    st.title("AI Project: โรคหัวใจ & จำแนกรูปภาพ")

elif choice == "ทำนายโรคหัวใจ":
    st.header("ทำนายโรคหัวใจ")
    age = st.number_input("อายุ", 1, 120)
    sex = st.selectbox("เพศ", [0, 1], format_func=lambda x: "ชาย" if x else "หญิง")
    cp = st.number_input("CP", 0, 3)
    trestbps = st.number_input("ความดัน", 80, 200)
    chol = st.number_input("คอเลสเตอรอล", 100, 400)
    if st.button("ทำนาย"):
        data = {"age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol}
        res = requests.post(f"{API_URL}/predict_heart", json=data)
        st.success(f"ผลลัพธ์: {'มีความเสี่ยง' if res.json()['result'] else 'ไม่มีความเสี่ยง'}")

elif choice == "จำแนกรูปภาพ":
    st.header("จำแนกรูปภาพ CIFAR-10")
    uploaded_file = st.file_uploader("อัพโหลดรูปภาพ (32x32)", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file).resize((32,32))
        st.image(image, caption="รูปที่อัพโหลด")
        image_array = np.array(image).tolist()
        res = requests.post(f"{API_URL}/predict_image", json={"image": image_array})
        st.success(f"ผลลัพธ์: {res.json()['result']}")

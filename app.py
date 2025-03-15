import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import os

# โหลดโมเดลที่เทรนไว้
model_path = "rf_model.pkl"
rf_model = joblib.load(model_path)
cnn_model_path = "cnn_model.h5"
cnn_model = keras.models.load_model(cnn_model_path)

# ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="Electronics Price & Image Classifier", layout="wide")
st.title("Final Project Intelligence System")
st.write("ทำนายราคาสินค้าอิเล็กทรอนิกส์ และจำแนกประเภทจากภาพ")

# Sidebar Menu
menu = st.sidebar.radio("เลือกเมนู", ["🏠 หน้าหลัก", "📊 ทำนายราคาสินค้า", "🖼️ จำแนกประเภทสินค้าจากภาพ", "📄 อธิบายโมเดล ML", "📄 อธิบายโมเดล CNN"])

if menu == "🏠 หน้าหลัก":
    st.header("📌 ข้อมูลโครงการ")
    st.write("โครงการนี้มีวัตถุประสงค์เพื่อสร้างระบบที่สามารถทำนายราคาสินค้าอิเล็กทรอนิกส์และจำแนกประเภทสินค้าจากภาพ โดยใช้ Machine Learning (Random Forest) และ Deep Learning (CNN)")
    st.write("**พัฒนาโดย:**")
    st.write(f"ชื่อ: เจนิภัทร สุรินทร์สภานนท์")
    st.write(f"รหัสนักศึกษา: 6604062630129")
    st.write(f"Section: 2")
    st.write(f"สาขาวิชา: วิทยาการคอมพิวเตอร์")
    st.write(f"มหาวิทยาลัย: มหาวิทยาลัยเทคโนโลยีพระจอมเกล้าพระนครเหนือ")

    st.markdown("""
    ### 📊 โมเดลที่พัฒนา
                
     #### 📌 Model 1: Price Prediction (Random Forest)
    - **Dataset:** ข้อมูลคุณสมบัติสินค้าอิเล็กทรอนิกส์
    - **อัลกอริทึม:** Random Forest Regressor
    - **เป้าหมาย:** ทำนายราคาสินค้า
    - **ผลลัพธ์:** แสดงราคาที่คาดการณ์
    
    #### 📌 Model 2: Image Classification (CNN)
    - **Dataset:** ภาพสินค้า 3 หมวดหมู่ (Smartphone, Laptop, Tablet)
    - **อัลกอริทึม:** Convolutional Neural Network (CNN)
    - **เป้าหมาย:** จำแนกประเภทสินค้าจากภาพ
    - **ผลลัพธ์:** แสดงผลการจำแนกประเภท
    
    ---
    
    ### 📖 วิธีใช้งาน
    
    1. เลือกเมนูที่ต้องการใช้งานด้านซ้ายมือ
    2. กรอกข้อมูลหรืออัปโหลดภาพสินค้า
    3. ระบบจะแสดงผลการทำนาย
    
    """)

elif menu == "📊 ทำนายราคาสินค้า":
    st.sidebar.header("ป้อนข้อมูลสินค้าของคุณ")
    brand = st.sidebar.selectbox("แบรนด์", ["Apple", "Samsung", "Sony", "Xiaomi", "Asus", "Dell", "HP", "Lenovo"])
    category = st.sidebar.selectbox("หมวดหมู่สินค้า", ["Smartphone", "Laptop", "Tablet"])
    year = st.sidebar.slider("ปีที่ผลิต", 2015, 2024, 2020)
    ram = st.sidebar.selectbox("RAM (GB)", [4, 8, 16, 32])
    storage = st.sidebar.selectbox("Storage (GB)", [64, 128, 256, 512, 1024])
    battery = st.sidebar.slider("ความจุแบตเตอรี่ (mAh)", 2000, 6000, 4000)
    screen_size = st.sidebar.slider("ขนาดหน้าจอ (นิ้ว)", 5.0, 17.0, 6.5)
    
    def preprocess_input(brand, category, year, ram, storage, battery, screen_size):
        data = pd.DataFrame([[year, ram, storage, battery, screen_size]], 
                             columns=["Year", "RAM", "Storage", "Battery", "Screen Size"])
        brand_cols = ["Brand_Apple", "Brand_Asus", "Brand_Dell", "Brand_HP", "Brand_Lenovo", "Brand_Samsung", "Brand_Sony", "Brand_Xiaomi"]
        category_cols = ["Category_Laptop", "Category_Smartphone", "Category_Tablet"]
        for col in brand_cols:
            data[col] = 1 if col == f"Brand_{brand}" else 0
        for col in category_cols:
            data[col] = 1 if col == f"Category_{category}" else 0
        return data
    
    input_data = preprocess_input(brand, category, year, ram, storage, battery, screen_size)
    if st.sidebar.button("ทำนายราคา"):
        predicted_price = rf_model.predict(input_data)[0]
        st.success(f"💰 ราคาที่คาดการณ์: {predicted_price:,.2f} บาท")
    
    st.subheader("📊 ข้อมูลที่ใช้ในการทำนาย")
    st.markdown(
    '<p style="color:red; font-size:16px;">ข้อมูลทํานายที่ได้ไม่ได้มีความน่าเชื่อถือ 100% อาจมีการเปลี่ยนแปลงตามสภาพการตลาดของสินค้า</p>', 
    unsafe_allow_html=True
    )

    st.write(input_data)

elif menu == "🖼️ จำแนกประเภทสินค้าจากภาพ":
    st.subheader("🖼️ อัปโหลดรูปภาพสินค้าอิเล็กทรอนิกส์")
    st.write("สิ่งที่train มีสินค้าสามชนิดประกอบด้วย")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.image("https://raw.githubusercontent.com/6604062630129/Final-project-ai-intel/main/assets/Smartphone.jpg", caption="Smartphone", use_container_width=True)

    with col2:
        st.image("https://raw.githubusercontent.com/6604062630129/Final-project-ai-intel/main/assets/Laptop.jpg", caption="Laptop", use_container_width=True)

    with col3:
        st.image("https://raw.githubusercontent.com/6604062630129/Final-project-ai-intel/main/assets/tablet.jpg", caption="Tablet", use_container_width=True)


    uploaded_file = st.file_uploader("เลือกไฟล์ภาพ", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        img = image.load_img(uploaded_file, target_size=(64, 64))
        img_array = image.img_to_array(img)/255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        prediction = cnn_model.predict(img_array)
        class_index = np.argmax(prediction)
        categories = ["Smartphone", "Laptop", "Tablet"]
        predicted_category = categories[class_index]
        
        st.image(uploaded_file, caption=f"สินค้า: {predicted_category}", use_container_width=True)
        st.success(f"🔍 ผลการจำแนก: {predicted_category}")

        st.subheader("📊 ความมั่นใจของโมเดล (Prediction Confidence)")

        percentages = (prediction[0] * 100).round(2) 

        for i in range(len(categories)):
            st.write(f"- **{categories[i]}**: {percentages[i]} %")

        st.success(f"🔍 ผลการจำแนก: **{predicted_category}** ด้วยความมั่นใจ {percentages[class_index]}%")

elif menu == "📄 อธิบายโมเดล ML":
    st.header("📄 อธิบายโมเดล Machine Learning (Random Forest)")
    st.markdown("""
    ### 📌 รายละเอียดโมเดล Random Forest Regressor
    - **โมเดล Random Forest** ถูกนำมาใช้สำหรับทำนายราคาสินค้าอิเล็กทรอนิกส์ โดยอิงจากข้อมูลต่าง ๆ เช่น
        - แบรนด์สินค้า (Brand)
        - หมวดหมู่สินค้า (Category)
        - ปีที่ผลิต (Year)
        - หน่วยความจำ RAM (GB)
        - ขนาด Storage (GB)
        - ความจุแบตเตอรี่ (mAh)
        - ขนาดหน้าจอ (นิ้ว)
    - **หลักการทำงาน**:
        - Random Forest เป็นโมเดลแบบ **Ensemble Learning** ที่รวมผลลัพธ์จาก **Decision Trees** จำนวนมากเพื่อลดความผิดพลาดและเพิ่มความแม่นยำ
        - การสุ่มเลือกตัวแปรและตัวอย่างข้อมูลเพื่อสร้างต้นไม้แต่ละต้น ทำให้โมเดลมีความยืดหยุ่นและทนต่อ Overfitting ได้ดี
    - **เป้าหมาย**:
        - ทำนายราคาสินค้าอิเล็กทรอนิกส์ตามคุณสมบัติต่าง ๆ ที่ผู้ใช้กรอกเข้ามา
    - **ผลลัพธ์ที่ได้**:
        - ราคาสินค้าอิเล็กทรอนิกส์ที่คาดการณ์ตามคุณสมบัติที่ป้อนเข้ามา
    """, unsafe_allow_html=True)


elif menu == "📄 อธิบายโมเดล CNN":
    st.header("📄 อธิบายโมเดล Convolutional Neural Network (CNN)")
    st.markdown("""
    ### 📌 รายละเอียดโมเดล Convolutional Neural Network (CNN)
    - **โมเดล CNN** ถูกใช้สำหรับจำแนกประเภทสินค้าจากภาพถ่าย โดยแบ่งออกเป็น 3 ประเภท ได้แก่:
        1. Smartphone
        2. Laptop
        3. Tablet
    - **หลักการทำงาน**:
        - CNN เป็นเครือข่ายประสาทเทียมที่ออกแบบมาเฉพาะสำหรับงานด้านการวิเคราะห์ภาพ
        - ประกอบด้วยชั้นสำคัญ เช่น:
            - **Convolution Layer**: สำหรับดึงคุณลักษณะสำคัญ (Feature Extraction)
            - **Pooling Layer**: สำหรับลดขนาดข้อมูล (Downsampling) เพื่อลดความซับซ้อน
            - **Fully Connected Layer**: เชื่อมต่อข้อมูลเพื่อทำการจำแนกประเภท
    - **เป้าหมาย**:
        - จำแนกภาพสินค้าออกเป็นหมวดหมู่ที่ตรงกับภาพที่อัปโหลดเข้ามา
    - **ผลลัพธ์ที่ได้**:
        - หมวดหมู่สินค้าที่คาดการณ์จากภาพ เช่น Smartphone, Laptop หรือ Tablet
    """, unsafe_allow_html=True)


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
    ## 🔍 แนวทางการพัฒนาโมเดล Machine Learning (Random Forest)

    ### 📌 1. การเตรียมข้อมูล (Data Preparation)
    - ข้อมูลที่ใช้ในการพัฒนาโมเดลรวบรวมจากข้อมูลจำเพาะของสินค้าอิเล็กทรอนิกส์ที่หลากหลาย เช่น:
        - **แบรนด์สินค้า (Brand)** เช่น Apple, Samsung, Asus ฯลฯ
        - **หมวดหมู่สินค้า (Category)** เช่น Smartphone, Laptop, Tablet
        - **ปีที่ผลิต (Year)** เพื่อบ่งบอกความใหม่-เก่าของสินค้า
        - **หน่วยความจำ RAM (GB)** ขนาด RAM ที่ติดตั้งในสินค้า
        - **ขนาด Storage (GB)** ความจุหน่วยเก็บข้อมูล
        - **ความจุแบตเตอรี่ (mAh)**
        - **ขนาดหน้าจอ (นิ้ว)**

    - ข้อมูลในแต่ละฟีเจอร์จะถูกทำการ **แปลงให้อยู่ในรูปแบบตัวเลข (Numerical Encoding)**:
        - ใช้ **One-Hot Encoding** สำหรับข้อมูลเชิงหมวดหมู่ (Categorical Data) เช่น Brand, Category

    ### 📌 2. ทฤษฎีของอัลกอริทึม (Algorithm Theory)
    - **Random Forest** เป็นอัลกอริทึมแบบ **Ensemble Learning**:
        - รวมเอาต้นไม้ตัดสินใจ (Decision Trees) จำนวนมากมารวมกัน
        - ใช้หลักการ **การโหวต (Voting)** หรือ **ค่าเฉลี่ย (Averaging)** ของผลลัพธ์จากต้นไม้แต่ละต้น
        - ลดความเสี่ยง Overfitting ที่เกิดจากต้นไม้เดียว
    - จุดเด่นของ Random Forest:
        - ทนต่อ Outliers และ Noise ได้ดี
        - แม่นยำสูง
        - ไม่ต้องปรับแต่งค่าพารามิเตอร์ (Hyperparameter) มากนัก

    ### 📌 3. ขั้นตอนการพัฒนาโมเดล (Model Development Process)
    1. **รวบรวมข้อมูลและทำความสะอาดข้อมูล (Data Cleaning)**
    2. **แปลงข้อมูลหมวดหมู่เป็นตัวเลข (One-Hot Encoding)**
    3. **แบ่งข้อมูลเป็นชุด Training และ Testing (Train-Test Split)**
    4. **สร้างและฝึกโมเดล Random Forest Regressor**
    5. **ประเมินผลลัพธ์ของโมเดลด้วยค่าความแม่นยำ (Accuracy) และค่าความผิดพลาด (Error)**
    6. **บันทึกโมเดลที่ผ่านการฝึกแล้วในรูปแบบ .pkl สำหรับใช้งานจริง**

    ### 🎯 เป้าหมายและผลลัพธ์ที่ได้ (Goal & Output)
    - **ทำนายราคาสินค้าอิเล็กทรอนิกส์ตามคุณสมบัติที่กรอกเข้ามา**
    - ได้ราคาที่คาดการณ์จากข้อมูลที่ผู้ใช้ป้อนอย่างแม่นยำ
    """, unsafe_allow_html=True)


elif menu == "📄 อธิบายโมเดล CNN":
    st.header("📄 อธิบายโมเดล Convolutional Neural Network (CNN)")
    st.markdown("""
    ## 🤖 แนวทางการพัฒนาโมเดล Neural Network (CNN)

    ### 📌 1. การเตรียมข้อมูล (Data Preparation)
    - รวบรวม **รูปภาพสินค้าจริง** จำนวนมากเพื่อใช้ฝึกโมเดล โดยแบ่งออกเป็น 3 หมวดหมู่หลัก:
        1. Smartphone
        2. Laptop
        3. Tablet
    - ข้อมูลภาพทุกภาพจะถูกปรับแต่งขนาดเป็น **128x128 พิกเซล** ให้มีขนาดเท่ากันทั้งหมด
    - ใช้ **เทคนิคเพิ่มข้อมูล (Data Augmentation)** เพื่อจำลองความหลากหลายของภาพ เช่น:
        - การหมุนภาพ (Rotation)
        - การกลับภาพแนวนอน (Horizontal Flip)
        - การขยาย/ย่อ (Zoom)
        - การเลื่อนตำแหน่ง (Shift)

    ### 📌 2. ทฤษฎีของอัลกอริทึม (Algorithm Theory)
    - **CNN (Convolutional Neural Network)** เป็นโครงข่ายประสาทเทียมที่ออกแบบเฉพาะสำหรับวิเคราะห์ภาพ:
        - **Convolution Layer**: ดึงคุณลักษณะสำคัญจากภาพ เช่น ขอบ, สี, รูปทรง
        - **Pooling Layer**: ลดขนาดข้อมูล (Downsampling) เพื่อเพิ่มความเร็วและลด Overfitting
        - **Batch Normalization**: ทำให้ข้อมูลกระจายตัวดีขึ้นและช่วยให้การเรียนรู้มีประสิทธิภาพ
        - **Dropout Layer**: ลด Overfitting โดยการสุ่มปิดบางโหนดระหว่างฝึก
        - **Fully Connected Layer (Dense Layer)**: ทำหน้าที่จำแนกหมวดหมู่ของภาพ

    ### 📌 3. ขั้นตอนการพัฒนาโมเดล (Model Development Process)
    1. **เตรียมรูปภาพและทำ Data Augmentation เพิ่มข้อมูลเทียม**
    2. **ออกแบบโมเดล CNN หลายชั้น**:
        - Conv2D + BatchNorm + MaxPooling2D
        - GlobalAveragePooling2D เพื่อแปลงข้อมูลเข้าสู่ Dense
        - Dense + Dropout
        - Output Layer ใช้ Softmax สำหรับ 3 หมวดหมู่
    3. **กำหนด Loss Function เป็น Categorical Crossentropy**
    4. **กำหนด Optimizer เป็น Adam สำหรับเรียนรู้เร็วและแม่นยำ**
    5. **เพิ่ม EarlyStopping และ Checkpoint เพื่อหยุดเมื่อ Val Loss ไม่พัฒนา และบันทึกโมเดลที่ดีที่สุด**
    6. **ทำการเทรนโมเดลด้วยข้อมูลทั้งหมด**
    7. **บันทึกโมเดลเป็นไฟล์ .h5 สำหรับนำไปใช้งานจริง**

    ### 🎯 เป้าหมายและผลลัพธ์ที่ได้ (Goal & Output)
    - จำแนกประเภทสินค้าในภาพเป็น:
        1. Smartphone
        2. Laptop
        3. Tablet
    - แสดงผลลัพธ์เป็น **หมวดหมู่ที่คาดการณ์พร้อมเปอร์เซ็นต์ความมั่นใจ (Confidence Score)**

    """, unsafe_allow_html=True)



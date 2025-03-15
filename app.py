
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
st.title("📱💻🔊 Electronics Price Prediction & Image Classification")
st.write("ทำนายราคาสินค้าอิเล็กทรอนิกส์ และจำแนกประเภทจากภาพ")

# Sidebar Menu
menu = st.sidebar.radio("เลือกเมนู", ["📊 ทำนายราคาสินค้า", "🖼️ จำแนกประเภทสินค้าจากภาพ"])

if menu == "📊 ทำนายราคาสินค้า":
    st.sidebar.header("📌 ป้อนข้อมูลสินค้าของคุณ")
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
        brand_cols = ["Brand_Asus", "Brand_Dell", "Brand_HP", "Brand_Lenovo", "Brand_Samsung", "Brand_Sony", "Brand_Xiaomi"]
        category_cols = ["Category_Laptop", "Category_Smartphone","Category_Tablet"]
        for col in brand_cols:
            data[col] = 1 if col == f"Brand_{brand}" else 0
        for col in category_cols:
            data[col] = 1 if col == f"Category_{category}" else 0
        return data
    
    input_data = preprocess_input(brand, category, year, ram, storage, battery, screen_size)
    if st.sidebar.button("🔍 ทำนายราคา"):
        predicted_price = rf_model.predict(input_data)[0]
        st.success(f"💰 ราคาที่คาดการณ์: {predicted_price:,.2f} บาท")
    
    st.subheader("📊 ข้อมูลที่ใช้ในการทำนาย")
    st.write(input_data)

elif menu == "🖼️ จำแนกประเภทสินค้าจากภาพ":
    st.subheader("🖼️ อัปโหลดรูปภาพสินค้าอิเล็กทรอนิกส์")
    uploaded_file = st.file_uploader("เลือกไฟล์ภาพ", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        img = image.load_img(uploaded_file, target_size=(64, 64))
        img_array = image.img_to_array(img)/255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        prediction = cnn_model.predict(img_array)
        class_index = np.argmax(prediction)
        categories = ["Smartphone", "Laptop", "Tablet", "Smartwatch", "Headphones"]
        predicted_category = categories[class_index]
        
        st.image(uploaded_file, caption=f"สินค้า: {predicted_category}", use_column_width=True)
        st.success(f"🔍 ผลการจำแนก: {predicted_category}")
    
import streamlit as st

menu = st.sidebar.selectbox("เลือกเมนู", ["Home", "ML", "ML doc", "NN", "NN doc"])

if menu == "Home":
    st.title("โครงงาน AI ทำนายโรคหัวใจและจำแนกรูปภาพ")
    st.write("รายละเอียดโครงงาน...")

elif menu == "ML":
    st.header("ทำนายโรคหัวใจ")
    st.write("ฟอร์มกรอกข้อมูล + ปุ่มทำนาย")

elif menu == "ML doc":
    st.header("รายละเอียดโมเดล ML")
    st.write("ข้อมูล Dataset และโมเดลที่ใช้")

elif menu == "NN":
    st.header("ทำนายรูปภาพ (CNN)")
    st.write("อัพโหลดภาพ + แสดงผลลัพธ์")

elif menu == "NN doc":
    st.header("รายละเอียดโมเดล NN")
    st.write("ข้อมูล Dataset และโครงสร้างโมเดล CNN")

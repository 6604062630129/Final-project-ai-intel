# Final AI Project
## รายละเอียดโปรเจค
- ทำนายโรคหัวใจด้วย Machine Learning (Logistic Regression)
- จำแนกรูปภาพ CIFAR-10 ด้วย Neural Network (CNN)

## โครงสร้างไฟล์
- dataset/: ไฟล์ข้อมูลที่ใช้ในการ Train
- models/: ไฟล์โมเดลที่ Train แล้ว
- notebooks/: Jupyter Notebook สำหรับ Train โมเดล
- streamlit_app/: โค้ดของเว็บ Streamlit
- requirements.txt: รายการไลบรารีที่ใช้

## วิธีใช้งาน
### ติดตั้งไลบรารี
```
pip install -r requirements.txt
```
### รัน Streamlit App
```
streamlit run streamlit_app/app.py
```

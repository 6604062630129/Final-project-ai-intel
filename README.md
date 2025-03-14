# Final AI Project

## รายละเอียดโปรเจค
- ทำนายโรคหัวใจด้วย Machine Learning (Logistic Regression)
- จำแนกรูปภาพ CIFAR-10 ด้วย Deep Learning (CNN)

## โครงสร้างไฟล์
- models/: ไฟล์โมเดล AI
- dataset/: ข้อมูล Dataset (หากมี)
- streamlit_app/: เว็บ Streamlit
- api_app/: API สำหรับโมเดล

## วิธีใช้งาน
### ติดตั้งไลบรารี
```bash
pip install -r requirements.txt
```

### รัน API
```bash
uvicorn api_app.api_app:app --reload --host 0.0.0.0 --port 8000
```

### รัน Streamlit
```bash
streamlit run streamlit_app/app.py
```

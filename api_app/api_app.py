
from fastapi import FastAPI, Request
import joblib
import tensorflow as tf
import numpy as np

app = FastAPI()

# โหลดโมเดล
heart_model = joblib.load('models/heart_disease_model.pkl')
heart_scaler = joblib.load('models/heart_disease_scaler.pkl')
cnn_model = tf.keras.models.load_model('models/cnn_cifar10_model.h5')

@app.post('/predict_heart')
async def predict_heart(data: dict):
    features = [data['age'], data['sex'], data['cp'], data['trestbps'], data['chol']]
    scaled = heart_scaler.transform([features])
    result = heart_model.predict(scaled)[0]
    return {"result": int(result)}

@app.post('/predict_image')
async def predict_image(data: dict):
    image_data = np.array(data['image']).astype('float32') / 255.0
    prediction = cnn_model.predict(np.expand_dims(image_data, axis=0))
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    predicted_class = class_names[np.argmax(prediction)]
    return {"result": predicted_class}

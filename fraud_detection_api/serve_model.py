from flask import Flask, request, jsonify
import joblib
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# Initialize Flask app
app = Flask(__name__)

# Load model from Google Drive
model_path = '/models/lstm_fraud_model.h5'
model = tf.keras.models.load_model(model_path)
print("Model loaded successfully.")

# Initialize scaler
scaler = MinMaxScaler()

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json.get('data', None)

    if data is None:
        return jsonify({"error": "No data provided"}), 400

    # Preprocess input data
    data_array = np.array(data).reshape(-1, len(data))
    data_scaled = scaler.transform(data_array)
    data_reshaped = data_scaled.reshape(data_scaled.shape[0], data_scaled.shape[1], 1)

    # Make prediction
    prediction = model.predict(data_reshaped)
    fraud_prediction = (prediction > 0.5).astype(int).tolist()

    return jsonify({"prediction": fraud_prediction})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

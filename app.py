from flask import Flask, request, jsonify
import joblib
import numpy as np
from datetime import datetime

app = Flask(__name__)

# Load trained model
model = joblib.load("iris_model.pkl")

# Log file (ELK stack will read this)
LOG_FILE = "/var/log/ml_api.log"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Extract features
    features = np.array(data['features']).reshape(1, -1)
    
    # Predict
    prediction = model.predict(features)[0]

    # Log the request & prediction
    log = {
        "timestamp": datetime.utcnow().isoformat(),
        "input": data['features'],
        "prediction": str(prediction)
    }

    with open(LOG_FILE, "a") as f:
        f.write(str(log) + "\n")

    return jsonify({"prediction": str(prediction)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

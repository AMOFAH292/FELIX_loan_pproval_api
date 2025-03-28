from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load the trained model
model = joblib.load("XGBClassifier.pkl")

# Initialize Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return "Loan Approval Prediction API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()

        # Convert input data into a NumPy array
        features = np.array([list(data.values())], dtype=np.float64).reshape(1, -1)

        # Make prediction
        prediction = int(model.predict(features)[0])  # Ensure integer output
        probability = float(model.predict_proba(features)[0][1])  # Convert to Python float

        # Return the response
        return jsonify({
            "loan_approval_status": "Approved" if prediction == 1 else "Denied",
            "approval_probability": round(probability, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

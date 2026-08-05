from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load("telecom_tower_model.pkl")


@app.route("/")
def home():
    print("Home route accessed")
    return """
    <html>
        <head>
            <title>Telecom Tower Prediction</title>
        </head>
        <body>
            <h1>Telecom Tower Prediction API is Running Successfully!</h1>
        </body>
    </html>
    """


@app.route("/predict", methods=["POST"])
def predict():
    print("Predict route accessed")

    data = request.get_json()
    print("Received Data:", data)

    df = pd.DataFrame([data])

    prediction = model.predict(df)

    print("Prediction:", prediction[0])

    return jsonify({
        "prediction": int(prediction[0])
    })


if __name__ == "__main__":
    app.run(debug=True)
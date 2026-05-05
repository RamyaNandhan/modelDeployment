from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load("../deployment_files/best_random_forest_model.pkl")
preprocessor = joblib.load("../deployment_files/preprocessor.pkl")

@app.route("/")
def home():
    return "SuperKart Sales Forecast API Running"

@app.route("/predict", methods=["POST"])
def predict():

    data = request.json

    df = pd.DataFrame([data])

    X = preprocessor.transform(df)

    prediction = model.predict(X)[0]

    return jsonify({"predicted_sales": float(prediction)})

if __name__ == "__main__":
    app.run(debug=True)
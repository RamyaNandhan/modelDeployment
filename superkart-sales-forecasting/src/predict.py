import joblib
import pandas as pd

model = joblib.load("../deployment_files/best_random_forest_model.pkl")
preprocessor = joblib.load("../deployment_files/preprocessor.pkl")

def predict_sales(input_dict):

    df = pd.DataFrame([input_dict])

    X = preprocessor.transform(df)

    pred = model.predict(X)

    return pred[0]
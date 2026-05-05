import joblib
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from preprocess import load_data, preprocess

df = load_data("../data/sales_data.csv")

X_train, X_test, y_train, y_test = preprocess(df, "Product_Store_Sales_Total")

model = joblib.load("../deployment_files/best_random_forest_model.pkl")

preds = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, preds))
mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)

print("RMSE:", rmse)
print("MAE:", mae)
print("R2:", r2)
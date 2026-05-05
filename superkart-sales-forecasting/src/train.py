import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from preprocess import load_data, preprocess

df = load_data("../data/sales_data.csv")

X_train, X_test, y_train, y_test = preprocess(df, "Product_Store_Sales_Total")

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=12,
    random_state=42
)

model.fit(X_train, y_train)

preds = model.predict(X_test)

print("R2 Score:", r2_score(y_test, preds))

joblib.dump(model, "../deployment_files/best_random_forest_model.pkl")

print("Model Saved")
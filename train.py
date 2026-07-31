import os
import joblib
import pandas as pd

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet
)

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

# -----------------------------
# Load Dataset
# -----------------------------

housing = fetch_california_housing(as_frame=True)

df = housing.frame

print(df.head())

X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

# -----------------------------
# Train Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# Models
# -----------------------------

models = {

    "Linear Regression":
        LinearRegression(),

    "Ridge":
        Ridge(alpha=1.0),

    "Lasso":
        Lasso(alpha=0.01),

    "ElasticNet":
        ElasticNet(alpha=0.01, l1_ratio=0.5),

    "Decision Tree":
        DecisionTreeRegressor(random_state=42),

    "Random Forest":
        RandomForestRegressor(
            n_estimators=200,
            random_state=42
        ),

    "Gradient Boosting":
        GradientBoostingRegressor(
            random_state=42
        )

}

results = []

best_model = None
best_score = -999

# -----------------------------
# Training
# -----------------------------

for name, model in models.items():

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)

    mse = mean_squared_error(y_test, predictions)

    rmse = mse ** 0.5

    r2 = r2_score(y_test, predictions)

    results.append({

        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2

    })

    print("=" * 60)
    print(name)
    print("MAE :", round(mae,4))
    print("RMSE:", round(rmse,4))
    print("R2  :", round(r2,4))

    if r2 > best_score:

        best_score = r2
        best_model = pipeline

# -----------------------------
# Results
# -----------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="R2",
    ascending=False
)

print("\n")
print(results_df)

# -----------------------------
# Save Model
# -----------------------------

os.makedirs("models", exist_ok=True)

joblib.dump(
    best_model,
    "models/house_price_model.pkl"
)

print("\nBest Model Saved Successfully!")
import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"

# ========================================
# Function to build preprocessing pipeline
# ========================================


def build_pipeline(num_attribs, cat_attribs):
    # Numerical Pipeline
    num_pipeline = Pipeline(
        [("Imputer", SimpleImputer(strategy="median")), ("Scalar", StandardScaler())]
    )

    # Categorial Pipeline
    cat_pipeline = Pipeline(
        [
            ("Imputer", SimpleImputer(strategy="most_frequent")),
            ("Onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # Construct full pipeline
    full_pipeline = ColumnTransformer(
        [("num", num_pipeline, num_attribs), ("cat", cat_pipeline, cat_attribs)]
    )

    return full_pipeline


# ================================
# Training
# ================================


def train_model():
    student = pd.read_csv("Student_Performance.csv")

    # Stratified split based on descriptive Hours Studied
    student["studied_grouped"] = pd.cut(
        student["Hours Studied"],
        bins=[0, 3, 6, 9, np.inf],
        labels=["Normal", "Average", "Serious", "Topper"],
    )

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    for train_idx, test_idx in split.split(student, student["studied_grouped"]):
        train_set = student.loc[train_idx].drop("studied_grouped", axis=1)
        test_set = student.loc[test_idx].drop("studied_grouped", axis=1)

    # Save test for inference
    test_set.to_csv("input.csv", index=False)

    # Split the features and labels
    X_train_features = train_set.drop("Performance Index", axis=1)
    Y_train_labels = train_set["Performance Index"]

    X_test_features = test_set.drop("Performance Index", axis=1)
    Y_test_labels = test_set["Performance Index"]

    # Numerical and categorial attributes
    num_attribs = X_train_features.select_dtypes(include=[np.number]).columns.tolist()
    cat_attribs = X_train_features.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    # Build Pipeline
    pipeline = build_pipeline(num_attribs, cat_attribs)
    X_train_prepared = pipeline.fit_transform(X_train_features)
    X_test_prepared = pipeline.transform(X_test_features)

    # Full Pipeline with model for easier cross-validation
    full_model = Pipeline(
        [("preprocess", pipeline), ("clf", RandomForestRegressor(random_state=42))]
    )

    # Train Model
    full_model.fit(X_train_features, Y_train_labels)
    joblib.dump(full_model, MODEL_FILE)
    print("Model trained and Saved")

    # Evaluation (Regression Metrics)
    Y_pred = full_model.predict(X_test_features)

    mse = mean_squared_error(Y_test_labels, Y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(Y_test_labels, Y_pred)
    r2 = r2_score(Y_test_labels, Y_pred)

    print("📊 Regression Metrics:")
    print("MSE:", mse)
    print("RMSE:", rmse)
    print("MAE:", mae)
    print("R²:", r2)

    # Cross-Validation
    scores = cross_val_score(
        full_model, X_train_features, Y_train_labels, cv=5, scoring="r2"
    )

    print("Cross-validation R² scores: ", scores)
    print("Average CV R²: ", scores.mean())

    # Feature Importance
    model = full_model.named_steps["clf"]
    features_name = full_model.named_steps["preprocess"].get_feature_names_out()
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(6, 14))
    plt.barh(np.array(features_name)[indices], importances[indices])
    plt.xlabel("Feature Importance")
    plt.ylabel("Features")
    plt.title("Student Performance Features Importance")
    plt.tight_layout()
    plt.show()

    # Save the predictions
    test_set["predictions"] = Y_pred
    test_set.to_csv("output.csv", index=False)
    print("Predictions saved to output.csv")


# ================================
# Inference
# ================================
def inference():
    if not os.path.exists(MODEL_FILE):
        print("Model not found. Train the model first first.")
        return

    full_model = joblib.load(MODEL_FILE)
    input_data = pd.read_csv("input.csv")

    X_input = input_data.drop(columns=["Performance Index"], errors="ignore")
    Y_true = (
        input_data["Performance Index"]
        if "Performance Index" in input_data.columns
        else None
    )

    predictions = full_model.predict(X_input)
    input_data["predictions"] = predictions
    input_data.to_csv("output.csv", index=False)
    print("Inference complete! Results saved to output.csv")

    if Y_true is not None:
        mse = mean_squared_error(Y_true, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(Y_true, predictions)
        r2 = r2_score(Y_true, predictions)

        print("Regression Metrics:")
        print("MSE:", mse)
        print("RMSE:", rmse)
        print("MAE:", mae)
        print("R²:", r2)

# ==============================
# Main
# ==============================
if __name__ == "__main__":
    if not os.path.exists(MODEL_FILE):
        train_model()
    else:
        inference()

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, r2_score
import joblib


# 📂 Read Data
def read_data(dataset_name):
    try:
        path = os.path.join("data", dataset_name)
        if dataset_name.endswith(".csv"):
            return pd.read_csv(path)
        elif dataset_name.endswith(".xlsx"):
            return pd.read_excel(path)
    except:
        return None


# ⚙️ Preprocess Data
def preprocess_data(df, target_column, scaler_type):

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Detect column types
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = [col for col in X.columns if col not in num_cols]

    # Pipelines
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler() if scaler_type == "standard" else MinMaxScaler())
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols)
    ])

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Fit transform
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    return X_train, X_test, y_train, y_test


# 🧠 Train Model
def train_model(X_train, y_train, model, model_name):

    model.fit(X_train, y_train)

    # Save model
    os.makedirs("trained_model", exist_ok=True)
    path = os.path.join("trained_model", f"{model_name}.pkl")
    joblib.dump(model, path)

    return model


# 📊 Evaluate Model
def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test)

    # Detect classification vs regression
    if y_test.dtype == "object" or y_test.nunique() < 20:
        return accuracy_score(y_test, y_pred)
    else:
        return r2_score(y_test, y_pred)
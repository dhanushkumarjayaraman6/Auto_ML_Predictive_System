"""
AutoML Predictive System
-------------------------
Upload a CSV / Excel / JSON file, clean it, pick a target column,
and let the app train several models and tell you which one performs best.

Run with:
    streamlit run app.py
"""

import io
import pickle

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    r2_score, mean_absolute_error, mean_squared_error,
)

try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

st.set_page_config(page_title="AutoML Predictive System", layout="wide")

# ----------------------------------------------------------------------
# 1. FILE LOADING
# ----------------------------------------------------------------------

def load_file(uploaded_file) -> pd.DataFrame:
    """Read csv / xlsx / xls / json / tsv into a DataFrame."""
    name = uploaded_file.name.lower()

    if name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded_file)
    elif name.endswith(".json"):
        df = pd.read_json(uploaded_file)
    elif name.endswith(".tsv"):
        df = pd.read_csv(uploaded_file, sep="\t")
    else:
        # last resort: try csv sniffing
        raw = uploaded_file.read()
        df = pd.read_csv(io.BytesIO(raw), sep=None, engine="python")
    return df


# ----------------------------------------------------------------------
# 2. CLEANING
# ----------------------------------------------------------------------

def clean_data(df: pd.DataFrame, missing_strategy: str, drop_thresh: float) -> pd.DataFrame:
    """Remove duplicates and handle missing values."""
    df = df.drop_duplicates().copy()

    # Drop columns that are mostly empty
    col_null_frac = df.isnull().mean()
    cols_to_drop = col_null_frac[col_null_frac > drop_thresh].index.tolist()
    df = df.drop(columns=cols_to_drop)

    if missing_strategy == "Drop rows with any missing values":
        df = df.dropna()
    else:
        for col in df.columns:
            if df[col].isnull().sum() == 0:
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                fill_val = df[col].median() if missing_strategy == "Fill (median/mode)" else df[col].mean()
                df[col] = df[col].fillna(fill_val)
            else:
                mode = df[col].mode()
                df[col] = df[col].fillna(mode.iloc[0] if not mode.empty else "Unknown")

    return df, cols_to_drop


# ----------------------------------------------------------------------
# 3. PROBLEM TYPE DETECTION
# ----------------------------------------------------------------------

def detect_problem_type(y: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(y):
        unique_ratio = y.nunique() / len(y)
        if y.nunique() <= 15 and unique_ratio < 0.05:
            return "classification"
        return "regression"
    return "classification"


# ----------------------------------------------------------------------
# 4. PREPROCESSING
# ----------------------------------------------------------------------

def preprocess(df: pd.DataFrame, target: str, problem_type: str):
    X = df.drop(columns=[target])
    y = df[target]

    # Encode categorical features
    encoders = {}
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            encoders[col] = le

    target_encoder = None
    if problem_type == "classification" and not pd.api.types.is_numeric_dtype(y):
        target_encoder = LabelEncoder()
        y = target_encoder.fit_transform(y.astype(str))

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

    return X, X_scaled, y, encoders, target_encoder, scaler


# ----------------------------------------------------------------------
# 5. MODEL ZOO
# ----------------------------------------------------------------------

def get_models(problem_type: str):
    if problem_type == "classification":
        models = {
            "Logistic Regression": (LogisticRegression(max_iter=1000), True),
            "Random Forest": (RandomForestClassifier(n_estimators=200, random_state=42), False),
            "Decision Tree": (DecisionTreeClassifier(random_state=42), False),
            "KNN": (KNeighborsClassifier(), True),
            "SVM": (SVC(probability=True), True),
        }
        if XGBOOST_AVAILABLE:
            models["XGBoost"] = (XGBClassifier(eval_metric="logloss", random_state=42), False)
    else:
        models = {
            "Linear Regression": (LinearRegression(), True),
            "Random Forest": (RandomForestRegressor(n_estimators=200, random_state=42), False),
            "Decision Tree": (DecisionTreeRegressor(random_state=42), False),
            "KNN": (KNeighborsRegressor(), True),
            "SVM": (SVR(), True),
        }
        if XGBOOST_AVAILABLE:
            models["XGBoost"] = (XGBRegressor(random_state=42), False)
    return models


def evaluate_classification(y_test, y_pred):
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "Precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
    }


def evaluate_regression(y_test, y_pred):
    return {
        "R2": r2_score(y_test, y_pred),
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
    }


# ----------------------------------------------------------------------
# STREAMLIT UI
# ----------------------------------------------------------------------

st.title("🤖 AutoML Predictive System")
st.caption("Upload a dataset → auto-clean it → pick a target → get the best model, automatically.")

uploaded_file = st.file_uploader(
    "Upload your dataset (CSV, XLSX, XLS, JSON, TSV)",
    type=["csv", "xlsx", "xls", "json", "tsv"],
)

if uploaded_file is not None:
    try:
        raw_df = load_file(uploaded_file)
    except Exception as e:
        st.error(f"Could not read this file: {e}")
        st.stop()

    st.subheader("1. Raw data preview")
    st.write(f"Shape: {raw_df.shape[0]} rows × {raw_df.shape[1]} columns")
    st.dataframe(raw_df.head(10))

    st.subheader("2. Cleaning options")
    col1, col2 = st.columns(2)
    with col1:
        missing_strategy = st.selectbox(
            "How should missing values be handled?",
            ["Fill (median/mode)", "Fill (mean/mode)", "Drop rows with any missing values"],
        )
    with col2:
        drop_thresh = st.slider(
            "Drop a column if more than this fraction of it is missing",
            min_value=0.3, max_value=1.0, value=0.6, step=0.05,
        )

    clean_df, dropped_cols = clean_data(raw_df, missing_strategy, drop_thresh)

    st.write(f"After cleaning: {clean_df.shape[0]} rows × {clean_df.shape[1]} columns "
             f"(removed {raw_df.shape[0] - clean_df.shape[0]} duplicate/incomplete rows)")
    if dropped_cols:
        st.write(f"Dropped mostly-empty columns: {dropped_cols}")
    st.dataframe(clean_df.head(10))

    if clean_df.shape[0] < 20:
        st.warning("Very few rows remain after cleaning — results may be unreliable.")

    st.subheader("3. Choose your target column")
    target = st.selectbox("What do you want to predict?", clean_df.columns)

    if st.button("🚀 Train models and find the best one"):
        problem_type = detect_problem_type(clean_df[target])
        st.info(f"Detected problem type: **{problem_type.upper()}**")

        X, X_scaled, y, encoders, target_encoder, scaler = preprocess(clean_df, target, problem_type)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        Xs_train, Xs_test, _, _ = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        models = get_models(problem_type)
        results = []
        trained_models = {}

        progress = st.progress(0.0, text="Training models...")
        for i, (name, (model, needs_scaling)) in enumerate(models.items()):
            try:
                tr_X, te_X = (Xs_train, Xs_test) if needs_scaling else (X_train, X_test)
                model.fit(tr_X, y_train)
                y_pred = model.predict(te_X)

                if problem_type == "classification":
                    metrics = evaluate_classification(y_test, y_pred)
                    primary = metrics["Accuracy"]
                else:
                    metrics = evaluate_regression(y_test, y_pred)
                    primary = metrics["R2"]

                metrics["Model"] = name
                metrics["_primary"] = primary
                results.append(metrics)
                trained_models[name] = model
            except Exception as e:
                st.warning(f"{name} failed: {e}")
            progress.progress((i + 1) / len(models), text=f"Trained {name}")

        progress.empty()

        results_df = pd.DataFrame(results).sort_values("_primary", ascending=False)
        display_df = results_df.drop(columns=["_primary"]).set_index("Model")

        st.subheader("4. Leaderboard")
        st.dataframe(display_df.style.highlight_max(axis=0, color="lightgreen"
                     if problem_type == "regression" else "lightgreen"))

        best_name = results_df.iloc[0]["Model"]
        best_score = results_df.iloc[0]["_primary"]
        best_model = trained_models[best_name]
        metric_label = "Accuracy" if problem_type == "classification" else "R²"

        st.success(f"🏆 Best model: **{best_name}** ({metric_label} = {best_score:.4f})")

        # Feature importance if available
        if hasattr(best_model, "feature_importances_"):
            st.subheader("5. Feature importance (best model)")
            imp_df = pd.DataFrame({
                "Feature": X.columns,
                "Importance": best_model.feature_importances_,
            }).sort_values("Importance", ascending=False)
            st.bar_chart(imp_df.set_index("Feature"))

        # Download best model
        buf = io.BytesIO()
        pickle.dump({
            "model": best_model,
            "problem_type": problem_type,
            "target_encoder": target_encoder,
            "feature_encoders": encoders,
            "scaler": scaler,
            "needs_scaling": models[best_name][1],
            "feature_columns": list(X.columns),
        }, buf)
        st.download_button(
            "⬇️ Download best model (.pkl)",
            data=buf.getvalue(),
            file_name=f"best_model_{best_name.replace(' ', '_')}.pkl",
            mime="application/octet-stream",
        )
else:
    st.info("Upload a file to get started.")

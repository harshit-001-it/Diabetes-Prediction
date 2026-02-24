"""
src/pipeline.py — Diabetes ML Training Pipeline
================================================
Trains 6 classifiers, prints comparison table, saves
best_model.pkl + scaler.pkl + model_results.json to models/.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from tabulate import tabulate
    TABULATE_AVAILABLE = True
except ImportError:
    TABULATE_AVAILABLE = False

# ── Paths (always relative to project root, not this file) ──────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH    = os.path.join(PROJECT_ROOT, "data", "diabetes.csv")
MODELS_DIR   = os.path.join(PROJECT_ROOT, "models")


# ---------------------------------------------------------------------------
# 1. LOAD
# ---------------------------------------------------------------------------
def load_data(filepath: str) -> pd.DataFrame:
    if not os.path.exists(filepath):
        sys.exit(
            f"[ERROR] Dataset not found: {filepath}\n"
            "Place 'diabetes.csv' inside the data/ folder."
        )
    df = pd.read_csv(filepath)
    print(f"[INFO] Loaded dataset: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


# ---------------------------------------------------------------------------
# 2. PREPROCESS
# ---------------------------------------------------------------------------
def preprocess(df: pd.DataFrame):
    zero_invalid = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    for col in zero_invalid:
        if col in df.columns:
            df[col] = df[col].replace(0, df[col].replace(0, np.nan).median())

    X = df.drop(columns=["Outcome"]).values
    y = df["Outcome"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    print(f"[INFO] Train: {X_train.shape[0]}  |  Test: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test, scaler


# ---------------------------------------------------------------------------
# 3. MODELS
# ---------------------------------------------------------------------------
def get_models() -> dict:
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=42),
        "SVM (RBF)":           SVC(kernel="rbf", probability=True, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7),
        "MLP Classifier":      MLPClassifier(hidden_layer_sizes=(128, 64),
                                              max_iter=500, random_state=42),
    }
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=200, use_label_encoder=False,
            eval_metric="logloss", random_state=42)
    else:
        models["Decision Tree"] = DecisionTreeClassifier(max_depth=6, random_state=42)
    return models


# ---------------------------------------------------------------------------
# 4. TRAIN & EVALUATE
# ---------------------------------------------------------------------------
def train_and_evaluate(models, X_train, X_test, y_train, y_test) -> pd.DataFrame:
    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred    = model.predict(X_test)
        accuracy  = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        recall    = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        results.append({
            "Model":     name,
            "Accuracy":  round(accuracy * 100, 2),
            "Precision": round(precision * 100, 2),
            "Recall":    round(recall * 100, 2),
        })
        print(f"  [✔] {name:25s}  Acc={accuracy*100:.2f}%  "
              f"Prec={precision*100:.2f}%  Rec={recall*100:.2f}%")
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# 5. DISPLAY TABLE
# ---------------------------------------------------------------------------
def display_results(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("Accuracy", ascending=False).reset_index(drop=True)
    df.index += 1
    print("\n" + "=" * 65)
    print("            📊  MODEL COMPARISON RESULTS")
    print("=" * 65)
    if TABULATE_AVAILABLE:
        print(tabulate(df, headers="keys", tablefmt="fancy_grid",
                       floatfmt=".2f", showindex=True))
    else:
        print(df.to_string())
    print("=" * 65)
    return df


# ---------------------------------------------------------------------------
# 6. SAVE BEST MODEL
# ---------------------------------------------------------------------------
def save_best(results_df, models, scaler):
    os.makedirs(MODELS_DIR, exist_ok=True)
    best_row  = results_df.loc[results_df["Accuracy"].idxmax()]
    best_name = best_row["Model"]

    model_path   = os.path.join(MODELS_DIR, "best_model.pkl")
    scaler_path  = os.path.join(MODELS_DIR, "scaler.pkl")
    results_path = os.path.join(MODELS_DIR, "model_results.json")

    joblib.dump(models[best_name], model_path)
    joblib.dump(scaler, scaler_path)
    results_df.to_json(results_path, orient="records", indent=2)

    print(f"\n  🏆  Best Model  : {best_name}")
    print(f"  🎯  Accuracy    : {best_row['Accuracy']:.2f}%")
    print(f"  📁  Saved to    : models/")
    return best_name


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def run_pipeline():
    print("\n" + "=" * 65)
    print("   🩺  DIABETES PREDICTION — ML TRAINING PIPELINE")
    print("=" * 65 + "\n")

    df                                        = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test, scaler = preprocess(df)

    print("\n[INFO] Training models…\n")
    models     = get_models()
    results_df = train_and_evaluate(models, X_train, X_test, y_train, y_test)
    results_df = display_results(results_df)
    save_best(results_df, models, scaler)

    return results_df


if __name__ == "__main__":
    run_pipeline()

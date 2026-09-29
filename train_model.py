"""
train_model.py
--------------
Trains a binary classification model on the Adult Census Income dataset.

Problem   : Binary Classification  (income <=50K  vs  >50K)
Dataset   : data/adult.csv
Output    : models/model.pkl  (full sklearn Pipeline)
            models/feature_info.pkl  (metadata used by app.py)

Run with:
    python train_model.py
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")

# --------------------------- paths --------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "adult.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "model.pkl")
FEATURE_INFO_PATH = os.path.join(MODELS_DIR, "feature_info.pkl")

os.makedirs(MODELS_DIR, exist_ok=True)

# --------------------------- load & inspect -----------------------------------
print("=" * 60)
print("SALARY PREDICTION - ADULT CENSUS INCOME")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
print(f"\nDataset shape : {df.shape}")
print(f"Columns       : {df.columns.tolist()}")

TARGET_COL = "income"

# Replace '?' with NaN so imputers handle them correctly
df.replace("?", np.nan, inplace=True)
df[TARGET_COL] = df[TARGET_COL].str.strip()

print(f"\nTarget distribution:\n{df[TARGET_COL].value_counts()}")

# --------------------------- feature splits -----------------------------------
NUMERIC_FEATURES = [
    "age",
    "fnlwgt",
    "educational-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]

CATEGORICAL_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "gender",
    "native-country",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

X = df[ALL_FEATURES].copy()
y = df[TARGET_COL].copy()

# Encode target: >50K → 1, <=50K → 0
le = LabelEncoder()
y_enc = le.fit_transform(y)          # <=50K=0, >50K=1
print(f"\nLabel mapping : {dict(zip(le.classes_, le.transform(le.classes_)))}")

# --------------------------- train/test split ---------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.20, random_state=42, stratify=y_enc
)
print(f"\nTrain size : {len(X_train):,}  |  Test size : {len(X_test):,}")

# --------------------------- preprocessing pipeline --------------------------
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, NUMERIC_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES),
    ]
)

# --------------------------- model candidates --------------------------------
# HistGradientBoostingClassifier handles NaN natively — no imputer needed for
# the numeric branch, but we keep the pipeline uniform.
candidates = {
    "Logistic Regression": LogisticRegression(
        max_iter=300, random_state=42, C=1.0, solver="lbfgs"
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=42, n_jobs=1
    ),
    "Hist Gradient Boosting": HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.1, max_depth=5, random_state=42
    ),
}

print("\n" + "-" * 60)
print("MODEL COMPARISON  (3-fold stratified CV on training set)")
print("-" * 60)

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
cv_results = {}

for name, clf in candidates.items():
    pipe = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=1)
    cv_results[name] = scores.mean()
    print(f"  {name:<25}  AUC = {scores.mean():.4f}  ± {scores.std():.4f}")

best_name = max(cv_results, key=cv_results.get)
print(f"\nBest model: {best_name}  (AUC = {cv_results[best_name]:.4f})")

# --------------------------- final training ----------------------------------
print("\n" + "-" * 60)
print(f"TRAINING FINAL MODEL: {best_name}")
print("-" * 60)

best_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", candidates[best_name]),
    ]
)
best_pipeline.fit(X_train, y_train)

# --------------------------- evaluation on test set --------------------------
y_pred = best_pipeline.predict(X_test)
y_prob = best_pipeline.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print(f"\nTest Accuracy : {acc:.4f}")
print(f"Test ROC-AUC  : {auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# --------------------------- save model & metadata ---------------------------
with open(MODEL_PATH, "wb") as f:
    pickle.dump(best_pipeline, f)

# Collect unique (non-null) values for each categorical feature for the UI
cat_unique = {}
for col in CATEGORICAL_FEATURES:
    vals = sorted(df[col].dropna().unique().tolist())
    cat_unique[col] = vals

# Collect numeric ranges for the UI
num_ranges = {}
for col in NUMERIC_FEATURES:
    num_ranges[col] = {
        "min": float(df[col].min()),
        "max": float(df[col].max()),
        "median": float(df[col].median()),
    }

feature_info = {
    "numeric_features": NUMERIC_FEATURES,
    "categorical_features": CATEGORICAL_FEATURES,
    "all_features": ALL_FEATURES,
    "target_col": TARGET_COL,
    "label_encoder_classes": le.classes_.tolist(),
    "cat_unique": cat_unique,
    "num_ranges": num_ranges,
    "best_model_name": best_name,
    "test_accuracy": acc,
    "test_roc_auc": auc,
    "problem_type": "classification",
}

with open(FEATURE_INFO_PATH, "wb") as f:
    pickle.dump(feature_info, f)

print(f"\nModel saved    : {MODEL_PATH}")
print(f"Metadata saved : {FEATURE_INFO_PATH}")
print("\nTraining complete!")

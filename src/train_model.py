"""
Delay prediction model.

Trains a logistic regression to estimate P(arrival delay > 15 min) from features
a traveler controls or knows at booking time: departure hour, route distance,
route congestion, carrier, and month.

Exports the fitted weights + the StandardScaler parameters so the exact same
model can run client-side in JavaScript (dot product + sigmoid). This is how you
deploy a model to a static site with no backend.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

from pipeline import CARRIER_NAMES

NUMERIC = ["DEP_HOUR", "DISTANCE", "ROUTE_CONGESTION", "MONTH"]
CARRIERS = list(CARRIER_NAMES.keys())


def _add_congestion(df: pd.DataFrame) -> pd.DataFrame:
    """Per-route congestion = that route's historical delayed share (0-1)."""
    congestion = df.groupby("ROUTE")["DELAYED"].mean()
    df = df.copy()
    df["ROUTE_CONGESTION"] = df["ROUTE"].map(congestion)
    return df


def build_features(df: pd.DataFrame):
    """One-hot carriers (drop first to avoid collinearity) + numeric features."""
    carrier_dummies = pd.get_dummies(
        pd.Categorical(df["OP_UNIQUE_CARRIER"], categories=CARRIERS),
        prefix="CARR", drop_first=True,
    ).astype(float)
    X = pd.concat([df[NUMERIC].reset_index(drop=True),
                   carrier_dummies.reset_index(drop=True)], axis=1)
    return X


def train(df: pd.DataFrame) -> dict:
    df = _add_congestion(df)
    X = build_features(df)
    y = df["DELAYED"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train_s, y_train)

    proba = model.predict_proba(X_test_s)[:, 1]
    preds = (proba >= 0.5).astype(int)
    metrics = {
        "auc": round(float(roc_auc_score(y_test, proba)), 3),
        "accuracy": round(float(accuracy_score(y_test, preds)), 3),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "base_rate": round(float(y.mean()), 3),
    }

    # Export everything JS needs to reproduce inference exactly.
    return {
        "features": list(X.columns),
        "means": scaler.mean_.round(6).tolist(),
        "scales": scaler.scale_.round(6).tolist(),
        "coef": model.coef_[0].round(6).tolist(),
        "intercept": round(float(model.intercept_[0]), 6),
        "carriers": CARRIERS,
        "metrics": metrics,
    }

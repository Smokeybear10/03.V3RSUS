"""Train final production model on v2 features (no odds — predictions for arbitrary matchups).

Outputs:
  data/model_v2.pkl  — artifact loaded by backend/model_engine.py
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
FEAT = ROOT / "features"
REPO = ROOT.parent
DATA_OUT = REPO / "data" / "model_v2.pkl"


def main():
    # Prefer v3 (with pedigree) if available
    v3 = FEAT / "final_features_v3.parquet"
    v2 = FEAT / "final_features_v2.parquet"
    path = v3 if v3.exists() else v2
    df = pd.read_parquet(path).sort_values("date").reset_index(drop=True)
    print(f"Using {path.name}")

    # Skip odds features — they aren't available at prediction time for arbitrary matchups
    feat_cols = [c for c in df.columns if c.startswith("diff_") and c != "diff_odds"] \
        + [c for c in df.columns if c.startswith("wc_")]

    X = df[feat_cols].copy()
    y = df["target"].values
    print(f"Training on {len(df)} fights × {len(feat_cols)} features")

    # Impute on the full set so the scaler+model see complete data
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)

    models = {
        "logistic_regression": LogisticRegression(max_iter=2000, random_state=42),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=10, min_samples_leaf=5,
            random_state=42, n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42,
        ),
    }

    # CV for honest metrics
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    tscv = TimeSeriesSplit(n_splits=5)
    metrics = {}
    for name, model in models.items():
        random_scores = cross_val_score(model, X_scaled, y, cv=kf, scoring="accuracy", n_jobs=-1)
        ts_scores = cross_val_score(model, X_scaled, y, cv=tscv, scoring="accuracy", n_jobs=-1)
        model.fit(X_scaled, y)
        metrics[name] = {
            "accuracy": round(float(random_scores.mean()), 4),
            "ts_accuracy": round(float(ts_scores.mean()), 4),
            "std": round(float(random_scores.std()), 4),
        }
        print(f"  {name:22s} random:{random_scores.mean():.4f} ts:{ts_scores.mean():.4f}")

    lr_coef = models["logistic_regression"].coef_[0]

    artifact = {
        "models": models,
        "imputer": imputer,
        "scaler": scaler,
        "features": feat_cols,
        "model_metrics": metrics,
        "lr_coefficients": lr_coef,
    }
    DATA_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, DATA_OUT, compress=3)
    size_mb = DATA_OUT.stat().st_size / 1e6
    print(f"\nSaved {DATA_OUT} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()

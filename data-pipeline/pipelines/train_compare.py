"""Train the same 3-model ensemble on the new feature set and compare to V3RSUS baseline."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
FEAT = ROOT / "features"


def main():
    for v in ("v4", "v3", "v2"):
        candidate = FEAT / f"final_features_{v}.parquet"
        if candidate.exists():
            path = candidate
            break
    else:
        path = FEAT / "final_features.parquet"
    df = pd.read_parquet(path).sort_values("date").reset_index(drop=True)
    print(f"Total: {len(df)} fights (loaded {path.name})\n")

    feat_cols = (
        [c for c in df.columns if c.startswith("diff_")]
        + [c for c in df.columns if c in {
            "red_odds", "blue_odds", "red_rank", "blue_rank",
            "bfo_red_open_p", "bfo_red_close_p", "bfo_red_movement_pct",
            "bfo_blue_open_p", "bfo_blue_close_p",
        }]
        + [c for c in df.columns if c.startswith("wc_")]
    )
    X = df[feat_cols].values
    y = df["target"].values

    print(f"Features: {len(feat_cols)}")

    models = {
        "logistic_regression": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, random_state=42)),
        ]),
        "random_forest": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_leaf=5,
                                           random_state=42, n_jobs=-1)),
        ]),
        "gradient_boosting": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("clf", GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                               random_state=42)),
        ]),
    }

    # Standard 5-fold (matches existing V3RSUS evaluation)
    print("\n=== 5-fold CV (random shuffle) ===")
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=kf, scoring="accuracy", n_jobs=-1)
        print(f"  {name:22s}: {scores.mean():.4f} ± {scores.std():.4f}")

    # Time-series CV (more honest for predictions of future fights)
    print("\n=== TimeSeriesSplit CV (5 folds, chronological) ===")
    tscv = TimeSeriesSplit(n_splits=5)
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=tscv, scoring="accuracy", n_jobs=-1)
        print(f"  {name:22s}: {scores.mean():.4f} ± {scores.std():.4f}")

    # Ensemble (mean of probabilities) — time-series eval only
    print("\n=== Ensemble (probability mean), TimeSeriesSplit ===")
    ens_scores = []
    for train_idx, test_idx in tscv.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        preds = []
        for model in models.values():
            model.fit(X_tr, y_tr)
            preds.append(model.predict_proba(X_te)[:, 1])
        ens_prob = np.mean(preds, axis=0)
        ens_pred = (ens_prob > 0.5).astype(int)
        ens_scores.append((ens_pred == y_te).mean())
    print(f"  ensemble             : {np.mean(ens_scores):.4f} ± {np.std(ens_scores):.4f}")

    # Baseline: always-predict-red (the trivial 63.7% baseline)
    print(f"\n  baseline (always red): {y.mean():.4f}")


if __name__ == "__main__":
    main()

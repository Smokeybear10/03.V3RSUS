"""Enrich final_features.parquet with odds, rankings, weight class.

Adds:
  - diff_odds                (R_odds - B_odds from ufc-master.csv)
  - red_odds, blue_odds      (raw odds, useful for non-LR models)
  - red_rank, blue_rank, diff_rank   (as-of pre-fight ranking from rankings_history.csv)
  - weight_class_*           (one-hot encoded weight classes)
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw"
FEAT = ROOT / "features"


def add_odds(df: pd.DataFrame) -> pd.DataFrame:
    master = pd.read_csv(RAW / "kaggle/ultimate-ufc-dataset/ufc-master.csv", low_memory=False)
    master["date"] = pd.to_datetime(master["date"])
    m = master[["R_fighter", "B_fighter", "date", "R_odds", "B_odds"]].copy()
    m["key"] = (m["R_fighter"].str.lower() + "|" + m["B_fighter"].str.lower()
                + "|" + m["date"].dt.strftime("%Y-%m-%d"))
    m = m[["key", "R_odds", "B_odds"]].rename(columns={"R_odds": "red_odds", "B_odds": "blue_odds"})

    # We need red/blue names — re-pull from fights
    fights = pd.read_parquet(ROOT / "processed/fights.parquet")
    df = df.merge(fights[["fight_id", "red_name", "blue_name"]], on="fight_id", how="left")
    df["key"] = (df["red_name"].str.lower() + "|" + df["blue_name"].str.lower()
                 + "|" + df["date"].dt.strftime("%Y-%m-%d"))
    df = df.merge(m, on="key", how="left").drop(columns=["key", "red_name", "blue_name"])
    df["diff_odds"] = df["red_odds"] - df["blue_odds"]
    print(f"  odds joined: {df['red_odds'].notna().sum()} / {len(df)}")
    return df


def add_rankings(df: pd.DataFrame) -> pd.DataFrame:
    rk = pd.read_csv(RAW / "kaggle/ufc-rankings/rankings_history.csv")
    rk["date"] = pd.to_datetime(rk["date"])
    rk["fighter_lower"] = rk["fighter"].str.lower()
    rk = rk.sort_values(["fighter_lower", "date"])

    fighters = pd.read_parquet(ROOT / "processed/fighters.parquet")
    name_lookup = dict(zip(fighters["fighter_id"], fighters["name"].str.lower()))

    def rank_for(fighter_id, fight_date):
        name = name_lookup.get(fighter_id)
        if not name or pd.isna(fight_date):
            return np.nan
        sub = rk[(rk["fighter_lower"] == name) & (rk["date"] < fight_date)]
        if sub.empty:
            return np.nan
        return sub.iloc[-1]["rank"]

    # Build a fast cache: per fighter, list of (date, rank); binary-search at lookup
    fighter_history = {}
    for fn, sub in rk.groupby("fighter_lower"):
        fighter_history[fn] = (sub["date"].values, sub["rank"].values)

    def fast_rank(fid, date):
        name = name_lookup.get(fid)
        if not name or pd.isna(date) or name not in fighter_history:
            return np.nan
        dates, ranks = fighter_history[name]
        idx = np.searchsorted(dates, np.datetime64(date), side="left")
        return float(ranks[idx - 1]) if idx > 0 else np.nan

    print("  Looking up red ranks...")
    df["red_rank"] = [fast_rank(r, d) for r, d in zip(df["red_fighter_id"], df["date"])]
    print("  Looking up blue ranks...")
    df["blue_rank"] = [fast_rank(b, d) for b, d in zip(df["blue_fighter_id"], df["date"])]
    df["diff_rank"] = df["red_rank"] - df["blue_rank"]
    print(f"  ranks joined: red={df['red_rank'].notna().sum()}, blue={df['blue_rank'].notna().sum()} / {len(df)}")
    return df


def add_weight_class(df: pd.DataFrame) -> pd.DataFrame:
    # weightclass already present; normalize and one-hot
    wc = df["weightclass"].fillna("Unknown").str.strip()
    # Group rare classes into Other
    counts = wc.value_counts()
    common = counts[counts >= 50].index
    wc_clean = wc.where(wc.isin(common), "Other")
    dummies = pd.get_dummies(wc_clean, prefix="wc").astype(int)
    df = pd.concat([df, dummies], axis=1)
    print(f"  weight class dummies: {len(dummies.columns)}")
    return df


def main():
    df = pd.read_parquet(FEAT / "final_features.parquet")
    print(f"Loaded {len(df)} fights")
    df["date"] = pd.to_datetime(df["date"])

    print("→ Adding odds...")
    df = add_odds(df)
    print("→ Adding rankings...")
    df = add_rankings(df)
    print("→ Adding weight class dummies...")
    df = add_weight_class(df)

    df.to_parquet(FEAT / "final_features_v2.parquet")
    print(f"\nWrote {len(df)} rows × {len(df.columns)} cols → final_features_v2.parquet")


if __name__ == "__main__":
    main()

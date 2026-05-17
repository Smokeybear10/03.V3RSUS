"""Enrich features with BestFightOdds open + close + movement.

Joins BFO odds (per-fighter-per-fight) to our fights table by (red_name, blue_name, date).
Outputs final_features_v4.parquet.

New features:
  diff_bfo_open_prob       red opening-implied prob - blue
  diff_bfo_close_prob      red closing-implied prob - blue  (avg of close_low and close_high)
  diff_bfo_movement_pct    red line movement % - blue
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw" / "bestfightodds"
FEAT = ROOT / "features"
PROC = ROOT / "processed"


def parse_odds(s):
    if not isinstance(s, str):
        return None
    m = re.search(r"([+-]?\d+)", s)
    return int(m.group(1)) if m else None


def implied_prob(odds):
    if odds is None or pd.isna(odds):
        return None
    odds = float(odds)
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return -odds / (-odds + 100.0)


def parse_movement(s):
    if not isinstance(s, str):
        return None
    m = re.search(r"([+-]?\d+(?:\.\d+)?)%", s)
    return float(m.group(1)) if m else None


def parse_bfo_date(s):
    """e.g., 'Oct 24th 2020' → datetime."""
    if not isinstance(s, str):
        return None
    cleaned = re.sub(r"(st|nd|rd|th)", "", s)
    try:
        return pd.to_datetime(cleaned, format="%b %d %Y")
    except (ValueError, TypeError):
        try:
            return pd.to_datetime(cleaned)
        except Exception:
            return None


def main():
    bfo = pd.read_parquet(RAW / "odds.parquet")
    print(f"BFO rows: {len(bfo)}")

    bfo["bfo_date"] = bfo["date"].map(parse_bfo_date)
    bfo["self_open_n"] = bfo["self_open"].map(parse_odds)
    bfo["self_close_low_n"] = bfo["self_close_low"].map(parse_odds)
    bfo["self_close_high_n"] = bfo["self_close_high"].map(parse_odds)
    bfo["self_movement_n"] = bfo["self_movement"].map(parse_movement)
    bfo["opp_open_n"] = bfo["opp_open"].map(parse_odds)
    bfo["opp_close_low_n"] = bfo["opp_close_low"].map(parse_odds)
    bfo["opp_close_high_n"] = bfo["opp_close_high"].map(parse_odds)

    # Implied probabilities (averaged close low+high)
    bfo["self_open_p"] = bfo["self_open_n"].map(implied_prob)
    bfo["self_close_p"] = bfo[["self_close_low_n", "self_close_high_n"]].mean(axis=1).map(implied_prob)
    bfo["opp_open_p"] = bfo["opp_open_n"].map(implied_prob)
    bfo["opp_close_p"] = bfo[["opp_close_low_n", "opp_close_high_n"]].mean(axis=1).map(implied_prob)

    bfo = bfo.dropna(subset=["bfo_date"])
    print(f"After date parse: {len(bfo)} rows")

    bfo["match_key"] = (
        bfo["fighter"].str.lower().str.strip()
        + "|"
        + bfo["opponent"].str.lower().str.strip()
        + "|"
        + bfo["bfo_date"].dt.strftime("%Y-%m-%d")
    )
    bfo = bfo.drop_duplicates(subset=["match_key"])

    # Load v3 features
    v3 = FEAT / "final_features_v3.parquet"
    df = pd.read_parquet(v3).copy()
    df["date_parsed"] = pd.to_datetime(df["date"])

    # Join red names
    fights = pd.read_parquet(PROC / "fights.parquet")[["fight_id", "red_name", "blue_name"]]
    df = df.merge(fights, on="fight_id", how="left")
    df["match_key"] = (
        df["red_name"].str.lower().str.strip()
        + "|"
        + df["blue_name"].str.lower().str.strip()
        + "|"
        + df["date_parsed"].dt.strftime("%Y-%m-%d")
    )

    merged = df.merge(
        bfo[["match_key", "self_open_p", "self_close_p", "self_movement_n",
             "opp_open_p", "opp_close_p"]],
        on="match_key", how="left",
    )
    merged.rename(columns={
        "self_open_p": "bfo_red_open_p",
        "self_close_p": "bfo_red_close_p",
        "self_movement_n": "bfo_red_movement_pct",
        "opp_open_p": "bfo_blue_open_p",
        "opp_close_p": "bfo_blue_close_p",
    }, inplace=True)

    merged["diff_bfo_open_prob"] = merged["bfo_red_open_p"] - merged["bfo_blue_open_p"]
    merged["diff_bfo_close_prob"] = merged["bfo_red_close_p"] - merged["bfo_blue_close_p"]
    merged["diff_bfo_movement"] = merged["bfo_red_movement_pct"]

    matched = merged["bfo_red_open_p"].notna().sum()
    print(f"\nFights matched with BFO: {matched} / {len(merged)} ({100*matched/len(merged):.0f}%)")

    merged.drop(columns=["match_key", "date_parsed", "red_name", "blue_name"], inplace=True)
    merged.to_parquet(FEAT / "final_features_v4.parquet")
    print(f"Saved v4: {len(merged)} rows × {len(merged.columns)} cols")


if __name__ == "__main__":
    main()

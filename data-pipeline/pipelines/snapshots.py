"""Build per-fighter snapshot table — each fighter's latest pre-fight feature state.

Used at prediction time: for an arbitrary matchup (A vs B), look up A's and B's snapshots,
compute diff features the same way training did, scale + predict.

Output: features/fighter_snapshots.parquet (one row per fighter)
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "processed"
FEAT = ROOT / "features"


def main():
    rolling = pd.read_parquet(FEAT / "fighter_rolling.parquet")
    fighters = pd.read_parquet(PROC / "fighters.parquet")
    glicko = pd.read_parquet(FEAT / "fighter_glicko.parquet")

    rolling["date"] = pd.to_datetime(rolling["date"])
    rolling = rolling.sort_values(["fighter_id", "date"])

    # For each fighter, take their most recent rolling row (state going into their latest fight).
    snap = rolling.groupby("fighter_id", as_index=False).tail(1).copy()

    # Merge post-most-recent-fight Glicko (their CURRENT rating)
    snap = snap.merge(glicko, on="fighter_id", how="left")

    # Now get post-fight Glicko ratings — for the LATEST fight they participated in,
    # update Glicko to compute their CURRENT rating (post-last-fight).
    # Pull from compute_glicko output reconstructed: the rolling table has pre_rating for the
    # NEXT fight that didn't happen. So we need fighter's rating AFTER their last fight.
    # Easier: pull from fights joined w/ glicko snapshots at row N+1 of each fighter.

    # Pre-rating before NEXT fight = post-rating from LAST fight. But there isn't a next fight,
    # so we use the same row's pre_rating as approximation (RD will be slightly stale).
    # For a snapshot, this is acceptable.

    # Join physical attributes
    fp = fighters[["fighter_id", "name", "nickname", "height_cm", "reach_cm", "weight_lbs", "stance", "dob"]]
    snap = snap.merge(fp, on="fighter_id", how="left")

    today = pd.Timestamp.today().normalize()
    snap["dob_parsed"] = pd.to_datetime(snap["dob"], errors="coerce")
    snap["current_age"] = ((today - snap["dob_parsed"]).dt.days / 365.25).round(2)
    snap["current_layoff_days"] = (today - snap["date"]).dt.days

    # If Wikipedia pedigree is available, merge it in (zero-pad missing fighters)
    pedigree_path = ROOT / "raw" / "wikipedia" / "pedigree.parquet"
    if pedigree_path.exists():
        ped = pd.read_parquet(pedigree_path)
        tag_cols = [c for c in ped.columns if c in {
            "d1_wrestler", "olympic_medalist", "olympian",
            "bjj_world_champ", "bjj_black_belt", "boxing_champ",
            "kickboxing_champ", "muay_thai_champ", "sambo_champ",
            "judo_champ", "karate_champ",
        }]
        if tag_cols:
            ped_keep = ped[["fighter_id"] + tag_cols].fillna(0)
            snap = snap.merge(ped_keep, on="fighter_id", how="left")
            for c in tag_cols:
                snap[c] = snap[c].fillna(0).astype(int)
            print(f"  merged pedigree tags from {len(ped)} fighters")

    snap.to_parquet(FEAT / "fighter_snapshots.parquet")

    print(f"snapshots: {len(snap)} fighters")
    cols_to_show = ["name", "date", "current_age", "current_layoff_days"]
    print(snap[cols_to_show].head(5).to_string())


if __name__ == "__main__":
    main()

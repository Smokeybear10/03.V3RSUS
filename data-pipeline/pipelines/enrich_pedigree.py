"""Merge Wikipedia pedigree tags into final_features_v2 → final_features_v3."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FEAT = ROOT / "features"
PEDIGREE_RAW = ROOT / "raw" / "wikipedia" / "pedigree.parquet"

TAG_COLS = [
    "d1_wrestler", "olympic_medalist", "olympian",
    "bjj_world_champ", "bjj_black_belt", "boxing_champ",
    "kickboxing_champ", "muay_thai_champ", "sambo_champ",
    "judo_champ", "karate_champ",
]


def main():
    if not PEDIGREE_RAW.exists():
        print(f"Pedigree data not yet present at {PEDIGREE_RAW}. Run Wikipedia scraper first.")
        return

    ped = pd.read_parquet(PEDIGREE_RAW)
    # Keep just the columns we need + fill NaN tags with 0
    keep = ["fighter_id"] + TAG_COLS
    ped = ped[keep].fillna(0)
    for c in TAG_COLS:
        ped[c] = ped[c].astype(int)

    df = pd.read_parquet(FEAT / "final_features_v2.parquet")
    print(f"Loaded {len(df)} fights × {len(df.columns)} cols")

    # Merge red fighter pedigree
    red = ped.rename(columns={"fighter_id": "red_fighter_id",
                              **{c: f"red_{c}" for c in TAG_COLS}})
    df = df.merge(red, on="red_fighter_id", how="left")

    # Merge blue fighter pedigree
    blue = ped.rename(columns={"fighter_id": "blue_fighter_id",
                               **{c: f"blue_{c}" for c in TAG_COLS}})
    df = df.merge(blue, on="blue_fighter_id", how="left")

    # Fill missing with 0 (fighter not found on Wikipedia → no pedigree tags)
    for c in TAG_COLS:
        df[f"red_{c}"] = df[f"red_{c}"].fillna(0).astype(int)
        df[f"blue_{c}"] = df[f"blue_{c}"].fillna(0).astype(int)
        df[f"diff_{c}"] = df[f"red_{c}"] - df[f"blue_{c}"]

    df.to_parquet(FEAT / "final_features_v3.parquet")
    print(f"Saved v3: {len(df)} rows × {len(df.columns)} cols")

    # Coverage report
    print("\nPedigree coverage in fights:")
    for c in TAG_COLS:
        nonzero = ((df[f"red_{c}"] == 1) | (df[f"blue_{c}"] == 1)).sum()
        print(f"  {c:20s} present in {nonzero} fights ({100 * nonzero / len(df):.1f}%)")


if __name__ == "__main__":
    main()

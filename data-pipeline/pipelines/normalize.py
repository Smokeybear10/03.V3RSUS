"""Normalize raw ufcstats CSVs into canonical tables keyed by URL hashes.

Outputs to data-pipeline/processed/:
  fighters.parquet  — one row per fighter, fighter_id is the ufcstats URL hash
  events.parquet    — one row per event
  fights.parquet    — one row per fight, with red/blue fighter ids
  rounds.parquet    — one row per (fight, fighter, round) — granular striking
"""
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw" / "ufcstats"
OUT = ROOT / "processed"
OUT.mkdir(parents=True, exist_ok=True)


def url_hash(url: str) -> str | None:
    if not isinstance(url, str):
        return None
    return url.rstrip("/").rsplit("/", 1)[-1] or None


def parse_height_cm(h):
    if not isinstance(h, str) or h.strip() in {"--", ""}:
        return None
    m = re.match(r"(\d+)'\s*(\d+)", h)
    if not m:
        return None
    feet, inches = int(m.group(1)), int(m.group(2))
    return round((feet * 12 + inches) * 2.54, 1)


def parse_reach_cm(r):
    if not isinstance(r, str) or r.strip() in {"--", ""}:
        return None
    m = re.match(r"([\d.]+)", r)
    return round(float(m.group(1)) * 2.54, 1) if m else None


def parse_weight_lbs(w):
    if not isinstance(w, str) or w.strip() in {"--", ""}:
        return None
    m = re.match(r"([\d.]+)", w)
    return float(m.group(1)) if m else None


def parse_landed_attempted(s):
    if not isinstance(s, str):
        return (None, None)
    m = re.match(r"(\d+)\s*of\s*(\d+)", s.strip())
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)


def parse_ctrl_seconds(s):
    if not isinstance(s, str) or s.strip() in {"--", ""}:
        return None
    m = re.match(r"(\d+):(\d+)", s.strip())
    return int(m.group(1)) * 60 + int(m.group(2)) if m else None


def parse_pct(s):
    if not isinstance(s, str):
        return None
    m = re.match(r"(\d+)%", s.strip())
    return int(m.group(1)) / 100.0 if m else None


def build_fighters():
    tott = pd.read_csv(RAW / "ufc_fighter_tott.csv")
    det = pd.read_csv(RAW / "ufc_fighter_details.csv")

    tott["fighter_id"] = tott["URL"].map(url_hash)
    det["fighter_id"] = det["URL"].map(url_hash)

    out = tott.merge(det[["fighter_id", "FIRST", "LAST", "NICKNAME"]], on="fighter_id", how="left")
    out["name"] = (out["FIRST"].fillna("") + " " + out["LAST"].fillna("")).str.strip()
    out["name"] = out["name"].where(out["name"].str.len() > 0, out["FIGHTER"])

    out["height_cm"] = out["HEIGHT"].map(parse_height_cm)
    out["reach_cm"] = out["REACH"].map(parse_reach_cm)
    out["weight_lbs"] = out["WEIGHT"].map(parse_weight_lbs)
    out["stance"] = out["STANCE"]
    out["dob"] = pd.to_datetime(out["DOB"], errors="coerce")
    out["nickname"] = out["NICKNAME"]

    cols = ["fighter_id", "name", "nickname", "height_cm", "reach_cm", "weight_lbs", "stance", "dob"]
    out = out[cols].drop_duplicates(subset="fighter_id").reset_index(drop=True)
    out.to_parquet(OUT / "fighters.parquet")
    print(f"fighters: {len(out)} rows")
    return out


def build_events():
    ev = pd.read_csv(RAW / "ufc_event_details.csv")
    ev["event_id"] = ev["URL"].map(url_hash)
    ev["date"] = pd.to_datetime(ev["DATE"], errors="coerce")
    ev["location"] = ev["LOCATION"]
    ev["country"] = ev["LOCATION"].str.split(",").str[-1].str.strip()
    ev["name"] = ev["EVENT"]
    ev = ev[["event_id", "name", "date", "location", "country"]].drop_duplicates(subset="event_id")
    ev.to_parquet(OUT / "events.parquet")
    print(f"events: {len(ev)} rows")
    return ev


def _split_bout(bout: str):
    if not isinstance(bout, str):
        return (None, None)
    parts = re.split(r"\s+vs\.?\s+", bout, maxsplit=1, flags=re.IGNORECASE)
    return (parts[0].strip(), parts[1].strip()) if len(parts) == 2 else (None, None)


def build_fights(fighters: pd.DataFrame, events: pd.DataFrame):
    fr = pd.read_csv(RAW / "ufc_fight_results.csv")
    fr["fight_id"] = fr["URL"].map(url_hash)

    fr[["red_name", "blue_name"]] = fr["BOUT"].apply(lambda b: pd.Series(_split_bout(b)))

    # Map name → fighter_id (case-insensitive)
    name_to_id = dict(zip(fighters["name"].str.lower(), fighters["fighter_id"]))
    fr["red_fighter_id"] = fr["red_name"].str.lower().map(name_to_id)
    fr["blue_fighter_id"] = fr["blue_name"].str.lower().map(name_to_id)

    # Outcome decoding: W/L → red wins, L/W → blue, D/D → draw, NC/NC → no contest
    def decode(o):
        if not isinstance(o, str):
            return None
        o = o.upper().strip()
        if o.startswith("W/L"):
            return "red"
        if o.startswith("L/W"):
            return "blue"
        if o.startswith("D"):
            return "draw"
        if "NC" in o:
            return "nc"
        return None

    fr["winner"] = fr["OUTCOME"].map(decode)
    fr["is_title_bout"] = fr["WEIGHTCLASS"].fillna("").str.contains("Title", case=False)
    fr["weightclass"] = fr["WEIGHTCLASS"].fillna("").str.replace("UFC ", "", regex=False).str.replace(" Title Bout", "", regex=False).str.replace(" Bout", "", regex=False).str.strip()
    fr["method"] = fr["METHOD"]
    fr["finish_round"] = pd.to_numeric(fr["ROUND"], errors="coerce")
    fr["finish_time"] = fr["TIME"]
    fr["referee"] = fr["REFEREE"]
    fr["num_rounds"] = fr["TIME FORMAT"].str.extract(r"(\d+)\s*Rnd").iloc[:, 0].astype("Int64")
    fr["details"] = fr["DETAILS"]

    # Join event_id via event name (strip whitespace — ufcstats CSVs have padding)
    ev_lookup = dict(zip(events["name"].str.strip(), events["event_id"]))
    fr["event_id"] = fr["EVENT"].str.strip().map(ev_lookup)

    cols = [
        "fight_id", "event_id", "red_fighter_id", "blue_fighter_id", "red_name", "blue_name",
        "winner", "weightclass", "is_title_bout", "method", "finish_round", "finish_time",
        "num_rounds", "referee", "details",
    ]
    out = fr[cols].drop_duplicates(subset="fight_id").reset_index(drop=True)
    out.to_parquet(OUT / "fights.parquet")
    print(f"fights: {len(out)} rows ({out['winner'].value_counts().to_dict()})")
    print(f"  fights w/ both fighter_ids resolved: {((out.red_fighter_id.notna()) & (out.blue_fighter_id.notna())).sum()} / {len(out)}")
    return out


def build_rounds(fights: pd.DataFrame, fighters: pd.DataFrame):
    fs = pd.read_csv(RAW / "ufc_fight_stats.csv", low_memory=False)

    # Need fight_id — join via EVENT + BOUT match against the fights table
    fd = pd.read_csv(RAW / "ufc_fight_details.csv")
    fd["fight_id"] = fd["URL"].map(url_hash)
    key_to_id = dict(zip(zip(fd["EVENT"].str.strip(), fd["BOUT"].str.strip()), fd["fight_id"]))
    fs["fight_id"] = list(zip(fs["EVENT"].str.strip(), fs["BOUT"].str.strip()))
    fs["fight_id"] = fs["fight_id"].map(key_to_id)

    name_to_id = dict(zip(fighters["name"].str.lower(), fighters["fighter_id"]))
    fs["fighter_id"] = fs["FIGHTER"].str.lower().map(name_to_id)

    fs["round"] = fs["ROUND"].str.extract(r"(\d+)").astype("Int64")

    fs["kd"] = pd.to_numeric(fs["KD"], errors="coerce").astype("Int64")
    fs[["sig_str_landed", "sig_str_attempted"]] = fs["SIG.STR."].apply(lambda s: pd.Series(parse_landed_attempted(s)))
    fs[["total_str_landed", "total_str_attempted"]] = fs["TOTAL STR."].apply(lambda s: pd.Series(parse_landed_attempted(s)))
    fs[["td_landed", "td_attempted"]] = fs["TD"].apply(lambda s: pd.Series(parse_landed_attempted(s)))
    fs[["head_landed", "head_attempted"]] = fs["HEAD"].apply(lambda s: pd.Series(parse_landed_attempted(s)))
    fs[["body_landed", "body_attempted"]] = fs["BODY"].apply(lambda s: pd.Series(parse_landed_attempted(s)))
    fs[["leg_landed", "leg_attempted"]] = fs["LEG"].apply(lambda s: pd.Series(parse_landed_attempted(s)))
    fs[["distance_landed", "distance_attempted"]] = fs["DISTANCE"].apply(lambda s: pd.Series(parse_landed_attempted(s)))
    fs[["clinch_landed", "clinch_attempted"]] = fs["CLINCH"].apply(lambda s: pd.Series(parse_landed_attempted(s)))
    fs[["ground_landed", "ground_attempted"]] = fs["GROUND"].apply(lambda s: pd.Series(parse_landed_attempted(s)))
    fs["sub_att"] = pd.to_numeric(fs["SUB.ATT"], errors="coerce").astype("Int64")
    fs["rev"] = pd.to_numeric(fs["REV."], errors="coerce").astype("Int64")
    fs["ctrl_seconds"] = fs["CTRL"].map(parse_ctrl_seconds)

    cols = [
        "fight_id", "fighter_id", "round", "kd",
        "sig_str_landed", "sig_str_attempted", "total_str_landed", "total_str_attempted",
        "td_landed", "td_attempted", "sub_att", "rev", "ctrl_seconds",
        "head_landed", "head_attempted", "body_landed", "body_attempted", "leg_landed", "leg_attempted",
        "distance_landed", "distance_attempted", "clinch_landed", "clinch_attempted",
        "ground_landed", "ground_attempted",
    ]
    out = fs[cols].dropna(subset=["fight_id", "fighter_id"]).reset_index(drop=True)
    out.to_parquet(OUT / "rounds.parquet")
    print(f"rounds: {len(out)} rows")
    return out


def main():
    print("→ fighters")
    fighters = build_fighters()
    print("→ events")
    events = build_events()
    print("→ fights")
    fights = build_fights(fighters, events)
    print("→ rounds")
    build_rounds(fights, fighters)
    print("\nDone. Tables in", OUT)


if __name__ == "__main__":
    main()

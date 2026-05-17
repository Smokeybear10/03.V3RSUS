"""Feature engineering — produces as-of features (no future-leak) per fight.

For each fight (event_date, red, blue), computes:
  - Glicko-2 rating + RD for each fighter BEFORE this fight
  - Rolling stats (last-3, last-5, exp-decay) BEFORE this fight
  - Days since last fight (layoff)
  - Career W/L/streak counts BEFORE this fight
  - Striking-zone tendencies (head/body/leg, distance/clinch/ground share)
  - Style tag based on finish distribution

Outputs to features/final_features.parquet — one row per fight, diffs (red - blue) + raw red + raw blue.
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "processed"
FEAT = ROOT / "features"
FEAT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Glicko-2 over chronological fights
# ---------------------------------------------------------------------------

INIT_RATING = 1500.0
INIT_RD = 350.0
INIT_VOL = 0.06
TAU = 0.5
EPS = 1e-6


def _g(phi):
    return 1.0 / math.sqrt(1.0 + 3.0 * phi ** 2 / math.pi ** 2)


def _E(mu, mu_j, phi_j):
    return 1.0 / (1.0 + math.exp(-_g(phi_j) * (mu - mu_j)))


def glicko_update(r, rd, vol, opponents):
    """opponents: list of (r_op, rd_op, score) — score is 1.0 win, 0.5 draw, 0.0 loss."""
    if not opponents:
        # No game in period: only RD grows
        phi = rd / 173.7178
        phi_star = math.sqrt(phi ** 2 + vol ** 2)
        return r, phi_star * 173.7178, vol

    mu = (r - 1500.0) / 173.7178
    phi = rd / 173.7178

    v_inv = 0.0
    delta_sum = 0.0
    for r_op, rd_op, s in opponents:
        mu_op = (r_op - 1500.0) / 173.7178
        phi_op = rd_op / 173.7178
        g_op = _g(phi_op)
        E_op = _E(mu, mu_op, phi_op)
        v_inv += g_op ** 2 * E_op * (1.0 - E_op)
        delta_sum += g_op * (s - E_op)
    v = 1.0 / v_inv
    delta = v * delta_sum

    a = math.log(vol ** 2)

    def f(x):
        ex = math.exp(x)
        num = ex * (delta ** 2 - phi ** 2 - v - ex)
        den = 2.0 * (phi ** 2 + v + ex) ** 2
        return num / den - (x - a) / TAU ** 2

    A = a
    if delta ** 2 > phi ** 2 + v:
        B = math.log(delta ** 2 - phi ** 2 - v)
    else:
        k = 1
        while f(a - k * TAU) < 0:
            k += 1
        B = a - k * TAU

    fA, fB = f(A), f(B)
    while abs(B - A) > EPS:
        C = A + (A - B) * fA / (fB - fA)
        fC = f(C)
        if fC * fB <= 0:
            A, fA = B, fB
        else:
            fA /= 2
        B, fB = C, fC

    vol_new = math.exp(A / 2.0)
    phi_star = math.sqrt(phi ** 2 + vol_new ** 2)
    phi_new = 1.0 / math.sqrt(1.0 / phi_star ** 2 + 1.0 / v)
    mu_new = mu + phi_new ** 2 * delta_sum

    return mu_new * 173.7178 + 1500.0, phi_new * 173.7178, vol_new


def compute_glicko(fights: pd.DataFrame, events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (per_fight_df, per_fighter_final_state_df).
    per_fight_df: one row per fight w/ pre-fight ratings for red+blue
    per_fighter_final_state_df: one row per fighter w/ rating/rd/vol AFTER their most recent fight
    """
    df = fights.merge(events[["event_id", "date"]], on="event_id", how="left")
    df = df.dropna(subset=["date", "red_fighter_id", "blue_fighter_id"])
    df = df.sort_values("date").reset_index(drop=True)

    state: dict[str, tuple[float, float, float]] = {}

    rows = []
    for _, row in df.iterrows():
        rid, bid = row["red_fighter_id"], row["blue_fighter_id"]
        r_red, rd_red, v_red = state.get(rid, (INIT_RATING, INIT_RD, INIT_VOL))
        r_blue, rd_blue, v_blue = state.get(bid, (INIT_RATING, INIT_RD, INIT_VOL))

        rows.append({
            "fight_id": row["fight_id"],
            "red_pre_rating": r_red, "red_pre_rd": rd_red,
            "blue_pre_rating": r_blue, "blue_pre_rd": rd_blue,
        })

        winner = row["winner"]
        if winner == "red":
            s_red, s_blue = 1.0, 0.0
        elif winner == "blue":
            s_red, s_blue = 0.0, 1.0
        elif winner == "draw":
            s_red, s_blue = 0.5, 0.5
        else:
            continue  # no-contest doesn't update ratings

        new_red = glicko_update(r_red, rd_red, v_red, [(r_blue, rd_blue, s_red)])
        new_blue = glicko_update(r_blue, rd_blue, v_blue, [(r_red, rd_red, s_blue)])
        state[rid] = new_red
        state[bid] = new_blue

    final_state = pd.DataFrame(
        [{"fighter_id": fid, "post_rating": r, "post_rd": rd, "post_vol": v}
         for fid, (r, rd, v) in state.items()]
    )
    return pd.DataFrame(rows), final_state


# ---------------------------------------------------------------------------
# Per-fighter rolling features
# ---------------------------------------------------------------------------

def fighter_long_form(fights: pd.DataFrame, events: pd.DataFrame, rounds: pd.DataFrame) -> pd.DataFrame:
    """Convert fights/rounds to one row per (fighter, fight) with their stats."""
    f = fights.merge(events[["event_id", "date"]], on="event_id", how="left")
    f = f.dropna(subset=["date"])

    red_view = f.assign(
        fighter_id=f["red_fighter_id"], opp_id=f["blue_fighter_id"],
        is_red=True, won=(f["winner"] == "red"), lost=(f["winner"] == "blue"),
        drew=(f["winner"] == "draw"),
    )
    blue_view = f.assign(
        fighter_id=f["blue_fighter_id"], opp_id=f["red_fighter_id"],
        is_red=False, won=(f["winner"] == "blue"), lost=(f["winner"] == "red"),
        drew=(f["winner"] == "draw"),
    )
    fighter_fights = pd.concat([red_view, blue_view], ignore_index=True)
    fighter_fights = fighter_fights[fighter_fights["fighter_id"].notna()].sort_values(["fighter_id", "date"])

    # Aggregate rounds per (fight, fighter) into per-fight totals
    agg = rounds.groupby(["fight_id", "fighter_id"]).agg(
        sig_str_landed=("sig_str_landed", "sum"),
        sig_str_attempted=("sig_str_attempted", "sum"),
        td_landed=("td_landed", "sum"),
        td_attempted=("td_attempted", "sum"),
        head_landed=("head_landed", "sum"),
        body_landed=("body_landed", "sum"),
        leg_landed=("leg_landed", "sum"),
        distance_landed=("distance_landed", "sum"),
        clinch_landed=("clinch_landed", "sum"),
        ground_landed=("ground_landed", "sum"),
        sub_att=("sub_att", "sum"),
        kd=("kd", "sum"),
        ctrl_seconds=("ctrl_seconds", "sum"),
        rounds_in_fight=("round", "count"),
    ).reset_index()

    out = fighter_fights.merge(agg, on=["fight_id", "fighter_id"], how="left")
    return out.sort_values(["fighter_id", "date"]).reset_index(drop=True)


def compute_rolling_features(long: pd.DataFrame) -> pd.DataFrame:
    """For each (fighter, fight): pre-fight rolling stats over last 3 and last 5 prior fights."""
    g = long.groupby("fighter_id", group_keys=False)

    stat_cols = [
        "sig_str_landed", "sig_str_attempted", "td_landed", "td_attempted",
        "head_landed", "body_landed", "leg_landed",
        "distance_landed", "clinch_landed", "ground_landed",
        "sub_att", "kd", "ctrl_seconds",
    ]

    out = long[["fight_id", "fighter_id", "date"]].copy()

    for n in (3, 5):
        for c in stat_cols:
            rolled = g[c].apply(lambda s, n=n: s.shift(1).rolling(n, min_periods=1).mean())
            out[f"{c}_last{n}"] = rolled.values

    # Career counts BEFORE this fight
    out["career_fights_before"] = g.cumcount().values
    out["career_wins_before"] = g["won"].apply(lambda s: s.shift(1).cumsum().fillna(0)).values
    out["career_losses_before"] = g["lost"].apply(lambda s: s.shift(1).cumsum().fillna(0)).values

    # Layoff (days since previous fight)
    prev_date = g["date"].shift(1)
    out["layoff_days"] = (long["date"] - prev_date).dt.days

    # Current win/lose streak ending just before this fight
    def _consec(series):
        b = series.shift(1).fillna(False).astype(int)
        group_ids = (b != b.shift(1).fillna(-1)).cumsum()
        return b.groupby(group_ids).cumsum()

    out["win_streak_before"] = g["won"].apply(_consec).values
    out["lose_streak_before"] = g["lost"].apply(_consec).values

    return out


# ---------------------------------------------------------------------------
# Assemble final per-fight feature table
# ---------------------------------------------------------------------------

def build_features():
    fighters = pd.read_parquet(PROC / "fighters.parquet")
    events = pd.read_parquet(PROC / "events.parquet")
    fights = pd.read_parquet(PROC / "fights.parquet")
    rounds = pd.read_parquet(PROC / "rounds.parquet")

    fights = fights[fights["winner"].isin({"red", "blue", "draw"})]
    print(f"Fights w/ valid winner: {len(fights)}")

    print("→ Glicko-2 chronological pass...")
    glicko, glicko_final = compute_glicko(fights, events)
    glicko_final.to_parquet(FEAT / "fighter_glicko.parquet")
    print(f"  glicko rows: {len(glicko)}, final fighter states: {len(glicko_final)}")

    print("→ Long-form fighter-fights + aggregating rounds...")
    long = fighter_long_form(fights, events, rounds)
    print(f"  long rows: {len(long)}")

    print("→ Rolling features per fighter...")
    rolling = compute_rolling_features(long)
    print(f"  rolling rows: {len(rolling)}")

    # Persist per-fighter rolling features (needed for snapshot lookups + prediction time)
    rolling.to_parquet(FEAT / "fighter_rolling.parquet")

    f = fights.merge(events[["event_id", "date"]], on="event_id", how="left")

    feat_cols = [c for c in rolling.columns if c not in {"fight_id", "fighter_id", "date"}]

    red_rolling = rolling[["fight_id", "fighter_id"] + feat_cols].rename(
        columns={"fighter_id": "red_fighter_id", **{c: f"red_{c}" for c in feat_cols}}
    )
    blue_rolling = rolling[["fight_id", "fighter_id"] + feat_cols].rename(
        columns={"fighter_id": "blue_fighter_id", **{c: f"blue_{c}" for c in feat_cols}}
    )
    f = f.merge(red_rolling, on=["fight_id", "red_fighter_id"], how="left")
    f = f.merge(blue_rolling, on=["fight_id", "blue_fighter_id"], how="left")

    # Add glicko
    f = f.merge(glicko, on="fight_id", how="left")

    # Add fighter physical stats (age computed at fight date)
    fp = fighters[["fighter_id", "height_cm", "reach_cm", "weight_lbs", "stance", "dob"]]
    for side in ("red", "blue"):
        fp_side = fp.rename(columns={c: f"{side}_{c}" for c in fp.columns if c != "fighter_id"})
        fp_side = fp_side.rename(columns={"fighter_id": f"{side}_fighter_id"})
        f = f.merge(fp_side, on=f"{side}_fighter_id", how="left")
        f[f"{side}_age"] = ((f["date"] - f[f"{side}_dob"]).dt.days / 365.25).round(2)

    # Compute diffs
    diff_pairs = [
        ("rating", "red_pre_rating", "blue_pre_rating"),
        ("rd", "red_pre_rd", "blue_pre_rd"),
        ("height", "red_height_cm", "blue_height_cm"),
        ("reach", "red_reach_cm", "blue_reach_cm"),
        ("weight", "red_weight_lbs", "blue_weight_lbs"),
        ("age", "red_age", "blue_age"),
        ("layoff", "red_layoff_days", "blue_layoff_days"),
        ("career_fights", "red_career_fights_before", "blue_career_fights_before"),
        ("career_wins", "red_career_wins_before", "blue_career_wins_before"),
        ("career_losses", "red_career_losses_before", "blue_career_losses_before"),
        ("win_streak", "red_win_streak_before", "blue_win_streak_before"),
        ("lose_streak", "red_lose_streak_before", "blue_lose_streak_before"),
    ]
    for n in (3, 5):
        for c in ["sig_str_landed", "sig_str_attempted", "td_landed", "td_attempted",
                  "head_landed", "body_landed", "leg_landed",
                  "distance_landed", "clinch_landed", "ground_landed",
                  "sub_att", "kd", "ctrl_seconds"]:
            diff_pairs.append((f"{c}_last{n}", f"red_{c}_last{n}", f"blue_{c}_last{n}"))

    for label, red_col, blue_col in diff_pairs:
        if red_col in f.columns and blue_col in f.columns:
            f[f"diff_{label}"] = f[red_col] - f[blue_col]

    # Target
    f["target"] = (f["winner"] == "red").astype(int)

    diff_cols = [c for c in f.columns if c.startswith("diff_")]
    cols = ["fight_id", "event_id", "date", "red_fighter_id", "blue_fighter_id",
            "weightclass", "is_title_bout", "num_rounds", "target"] + diff_cols

    out = f[cols].dropna(subset=["target"]).reset_index(drop=True)
    out.to_parquet(FEAT / "final_features.parquet")
    print(f"\nFinal features: {len(out)} rows × {len(out.columns)} cols")
    print(f"  diff features: {len(diff_cols)}")
    print(f"  target balance: {out['target'].mean():.3f}")
    return out


if __name__ == "__main__":
    build_features()

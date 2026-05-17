"""Quick inventory of every CSV under raw/. Prints rows, cols, schema preview."""
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "raw"

rows = []
for csv in sorted(ROOT.rglob("*.csv")):
    try:
        df = pd.read_csv(csv, low_memory=False, nrows=5)
        full = pd.read_csv(csv, low_memory=False, usecols=[df.columns[0]])
        n = len(full)
    except Exception as e:
        rows.append({"file": str(csv.relative_to(ROOT)), "error": str(e)[:80]})
        continue
    rows.append({
        "file": str(csv.relative_to(ROOT)),
        "rows": n,
        "cols": len(df.columns),
        "first_cols": ", ".join(df.columns[:6].tolist()),
    })

summary = pd.DataFrame(rows)
out = Path(__file__).resolve().parents[1] / "processed" / "inventory.csv"
out.parent.mkdir(exist_ok=True)
summary.to_csv(out, index=False)
print(summary.to_string(index=False))
print(f"\nWrote {out}")

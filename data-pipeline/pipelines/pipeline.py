"""Run the full pipeline end-to-end:
    normalize → features → enrich (odds+rank+wc) → enrich_pedigree → snapshots → train production.

Use this for nightly refreshes after new raw data lands.
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

STEPS = [
    ("normalize", HERE / "normalize.py"),
    ("features", HERE / "features.py"),
    ("enrich_odds_rank_wc", HERE / "enrich.py"),
    ("enrich_pedigree", HERE / "enrich_pedigree.py"),
    ("snapshots", HERE / "snapshots.py"),
    ("train_production", HERE / "train_production.py"),
]


def main():
    for name, script in STEPS:
        if not script.exists():
            print(f"  skipping {name} ({script.name} missing)")
            continue
        print(f"\n{'='*60}\n→ {name}\n{'='*60}")
        result = subprocess.run([sys.executable, str(script)])
        if result.returncode != 0:
            print(f"  {name} failed (exit {result.returncode})")
            sys.exit(result.returncode)
    print("\nPipeline complete.")


if __name__ == "__main__":
    main()

import os
import sys

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_root, 'backend'))

from model_engine import FightPredictor

predictor = FightPredictor()
predictor.train(os.path.join(_root, 'data-pipeline', 'processed', 'ufc-master-display.csv'))

out = os.path.join(_root, 'data', 'model.pkl')
predictor.save_artifact(out)
size_mb = os.path.getsize(out) / 1024 / 1024
print(f"Saved {out} ({size_mb:.1f} MB)")

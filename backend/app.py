from flask import Flask, send_from_directory, jsonify, request
from flask_cors import CORS
import sys
import os
import numpy as np

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from model_engine import FightPredictor

app = Flask(__name__, static_folder=os.path.join(_root, 'public'))
CORS(app)

print("Initializing ML Engine...")
predictor = FightPredictor()
_data_path = os.path.join(_root, 'data', 'ufc-master.csv')
_v2_path = os.path.join(_root, 'data', 'model_v2.pkl')
_legacy_path = os.path.join(_root, 'data', 'model.pkl')
_artifact_path = _v2_path if os.path.exists(_v2_path) else _legacy_path

if os.path.exists(_artifact_path):
    print(f"Loading pre-trained model from {_artifact_path}...")
    success = predictor.load_artifact(_artifact_path, _data_path)
else:
    success = predictor.train(_data_path)

if not success:
    print("WARNING: Model failed to initialize. Make sure data is present.")


def safe(val):
    if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
        return None
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return round(float(val), 1)
    return val


def build_profile(stats):
    wins = safe(stats.get('Wins', 0)) or 0
    losses = safe(stats.get('Losses', 0)) or 0
    draws = safe(stats.get('Draws', 0)) or 0
    ko_wins = safe(stats.get('WinsByKO', 0)) or 0
    sub_wins = safe(stats.get('WinsBySubmission', 0)) or 0
    tko_wins = safe(stats.get('WinsByTKODoctorStoppage', 0)) or 0
    dec_wins = (safe(stats.get('WinsByDecisionUnanimous', 0)) or 0) + \
               (safe(stats.get('WinsByDecisionSplit', 0)) or 0) + \
               (safe(stats.get('WinsByDecisionMajority', 0)) or 0)

    height_cm = safe(stats.get('HeightCms'))
    reach_cm = safe(stats.get('ReachCms'))

    return {
        "record": f"{wins}-{losses}-{draws}",
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "koWins": ko_wins + (tko_wins or 0),
        "subWins": sub_wins,
        "decWins": dec_wins,
        "avgSigStrLanded": safe(stats.get('AvgSigStrLanded')),
        "avgSigStrPct": safe(stats.get('AvgSigStrPct')),
        "avgTDLanded": safe(stats.get('AvgTDLanded')),
        "avgTDPct": safe(stats.get('AvgTDPct')),
        "avgSubAtt": safe(stats.get('AvgSubAtt')),
        "height": f"{round(height_cm / 2.54)}\u2033" if height_cm else None,
        "heightCm": height_cm,
        "reach": f"{round(reach_cm / 2.54)}\u2033" if reach_cm else None,
        "reachCm": reach_cm,
        "weight": safe(stats.get('WeightLbs')),
        "age": safe(stats.get('Age')),
        "stance": stats.get('Stance') if isinstance(stats.get('Stance'), str) else None,
        "currentWinStreak": safe(stats.get('CurrentWinStreak', 0)) or 0,
        "currentLoseStreak": safe(stats.get('CurrentLoseStreak', 0)) or 0,
        "longestWinStreak": safe(stats.get('LongestWinStreak', 0)) or 0,
        "totalRounds": safe(stats.get('TotalRoundsFought', 0)) or 0,
        "titleBouts": safe(stats.get('TotalTitleBouts', 0)) or 0,
        "totalFights": safe(stats.get('TotalFights', 0)) or 0,
    }


@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/<path:path>')
def serve_static(path):
    if os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/api/fighters', methods=['GET'])
def get_fighters():
    if predictor.fights_df is not None:
        red = predictor.fights_df['RedFighter'].dropna().unique().tolist()
        blue = predictor.fights_df['BlueFighter'].dropna().unique().tolist()
        fighters = sorted(list(set(red + blue)))
        return jsonify({"fighters": fighters})
    return jsonify({"fighters": []})


@app.route('/api/model-info', methods=['GET'])
def model_info():
    return jsonify(predictor.get_model_info())


@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    f1 = data.get('fighter1')
    f2 = data.get('fighter2')

    if not f1 or not f2:
        return jsonify({"error": "Both fighter1 and fighter2 are required"}), 400

    try:
        result = predictor.predict_matchup(f1, f2)

        f1_profile = build_profile(result['f1Stats'])
        f2_profile = build_profile(result['f2Stats'])

        return jsonify({
            "fighter1": result['f1Name'],
            "fighter2": result['f2Name'],
            "winner": result['winner'],
            "confidence": result['confidence'],
            "f1Prob": result['f1Prob'],
            "f2Prob": result['f2Prob'],
            "f1": f1_profile,
            "f2": f2_profile,
            "analysis": {
                "keyFactors": result['keyFactors'],
                "categories": result['categoryAnalysis'],
                "modelBreakdown": result['modelBreakdown'],
                "historicalMatchups": result['historicalMatchups'],
            },
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Server error during prediction"}), 500


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=2102)

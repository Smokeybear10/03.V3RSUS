# V3RSUS | MMA Predictive Analytics

**Live demo: [v3rsus.vercel.app](https://v3rsus.vercel.app/)**

Ensemble machine learning engine that predicts UFC fight outcomes using career stat differentials across 6,500+ historical bouts.

![Landing page walkthrough](docs/media/home.gif)

![Predictor walkthrough](docs/media/predict.gif)

## Quick Start

```bash
pip install -r requirements.txt

# Pull the trained model + lookup tables from the data repo
# (needs a GitHub token with read access to Smokeybear10/802-DATA.MMA)
gh auth login   # or: export GITHUB_TOKEN=...
bash scripts/fetch-data.sh

python backend/app.py
# open http://localhost:2102
```

Data is sourced from a separate repo: [802-DATA.MMA](https://github.com/Smokeybear10/802-DATA.MMA) — pipelines, scrapers, and ML training all live there. The fetch script downloads the latest release artifacts (model + lookup tables) into `data/`.

## What It Does

Pick two fighters. V3RSUS pulls their career-wide stats, computes the gap between them across dozens of dimensions, and runs an ensemble of classifiers to predict the winner. Each prediction includes:

- Win probability with per-model breakdown
- Top factors driving the call, ranked by impact
- Edge analysis across striking, grappling, physical, and experience categories
- Full fighter stat profiles and head-to-head comparison
- Historical matchup results if the fighters have fought before

## How It Works

1. Look up each fighter's most recent career-wide stats from the dataset
2. Compute differential features (Fighter 1 stat − Fighter 2 stat)
3. Standardize against training data distribution
4. Multiple models independently produce win probabilities; their mean is the final prediction
5. Explanation engine decomposes the logistic regression output to surface the top contributing factors

Models are trained on 6,528 UFC fights with 5-fold cross-validation. Ensemble accuracy is ~60%, competitive with academic ML studies on MMA (58–68% range). Vegas sits at ~65–70% with access to camp intel, injuries, and insider info we don't have.

## Tech Stack

| Layer | Tools |
|-------|-------|
| Frontend | Vanilla HTML / CSS / JS. Instrument Serif + Inter Tight + IBM Plex Mono |
| Backend | Python, Flask, Flask-CORS |
| ML | scikit-learn (Logistic Regression, Random Forest, Gradient Boosting) |
| Data | pandas, numpy, 6,500+ UFC fights from ufcstats.com |
| Deploy | Vercel (serverless Python) |

## Project Structure

```
V3RSUS/
├── api/                 # Vercel serverless entrypoint
│   └── index.py
├── backend/             # Flask app + ML engine
│   ├── app.py           # API routes, fighter profile builder
│   └── model_engine.py  # Training, prediction, explanation logic
├── public/              # Static frontend
│   ├── index.html       # Landing page
│   ├── predict.html     # Interactive predictor
│   ├── about.html       # Methodology + references
│   ├── home.css         # Global styles
│   ├── predict.css      # Predictor-specific styles
│   ├── script.js        # Client logic (autocomplete, render, fetch)
│   ├── scroll-anim.js   # Scroll-reveal animations
│   ├── icon.svg         # Σ favicon
│   ├── robots.txt
│   └── sitemap.xml
├── data/                # Training datasets
│   └── ufc-master.csv
├── scripts/             # Training + exploration scripts
├── notebooks/           # Research notebooks
├── docs/                # Paper, presentation, site demo gifs
├── requirements.txt
└── vercel.json
```

## API

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/fighters` | List of all fighters in the dataset |
| GET | `/api/model-info` | Fighter count, fight count, feature count |
| POST | `/api/predict` | Predict a matchup (body: `{ fighter1, fighter2 }`) |

## Credits

Built for CIS 5450 (Big Data Analytics) at the University of Pennsylvania by Thomas Ou, Aakash Jha, and Kevin Jiang.

---

Built by Thomas Ou

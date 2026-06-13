# ✈️ Flight Route Efficiency Analytics

A data analytics project answering a question every traveler has: **which routes, times, and carriers actually get you there on time?** — and a model that predicts your own flight's delay risk, running live in the browser.

🔗 **Live site:** https://gunpo.github.io/flight-route-analytics
📊 **Data schema:** [US DOT Bureau of Transportation Statistics — On-Time Performance](https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FGJ)

## What it does

1. **Route Efficiency Score** — a composite 0–100 metric combining *excess air time* (actual vs. distance-predicted flight duration) and delay frequency, ranking routes by real-world reliability.
2. **The delay cascade** — quantifies how departure delays compound hour-by-hour through the operating day, pinpointing the optimal booking window.
3. **Delay predictor** — a scikit-learn logistic regression that estimates P(arrival delay > 15 min) for any route / carrier / hour / month combination, running **client-side** in JavaScript.
4. **Carrier recovery & seasonality** — which airlines make up time in the air, and the worst months to fly.

## Architecture

The interesting part: a model trained in Python is deployed to a **fully static site** with no backend.

```
  data (BTS CSV)
        │
        ▼
  src/pipeline.py        → route scores, delay cascade, recovery, seasonality
  src/train_model.py     → logistic regression; exports weights + scaler params
        │
        ▼
  src/build_site.py      → writes everything to docs/data.json
        │
        ▼
  docs/ (static site)    → loads data.json, renders charts, and reproduces the
                           exact model inference in JS (standardize → dot
                           product → sigmoid). Hosted free on GitHub Pages.
```

No external runtime dependencies — Chart.js and the fonts are vendored into `docs/`, so the site loads instantly and works offline.

## Quick start

```bash
pip install -r requirements.txt

# 1) generate the bundled sample data (60k flights, BTS schema)
python src/generate_sample_data.py

# 2) run analysis + train model + export the site's data
python src/build_site.py

# 3) preview the site locally
cd docs && python -m http.server 8000   # then open http://localhost:8000
```

### Use real data

Download monthly On-Time Performance CSVs from [transtats.bts.gov](https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FGJ), drop them in `data/`, and point the builder at one:

```bash
python src/build_site.py data/your_bts_download.csv
```

Same code, real records — the website re-renders automatically.

## Deploy to GitHub Pages

1. Push this repo to GitHub.
2. Settings → **Pages** → Source: **Deploy from a branch** → Branch: `main`, folder: **/docs**.
3. Live at `https://gunpo.github.io/flight-route-analytics` in ~1 minute.

## Tech

`Python` · `pandas` · `NumPy` · `scikit-learn` · `Chart.js` · vanilla JS · GitHub Pages

## Roadmap

- [ ] Ingest full-year real BTS data (~7M flights/yr)
- [ ] Upgrade the model to gradient boosting and compare AUC
- [ ] Join NOAA weather data to separate weather vs. operational delay
- [ ] Add airport-level (not just route-level) drill-down

---

*Demo runs on simulated data matching the BTS schema. Built by SeungJin Paik.*

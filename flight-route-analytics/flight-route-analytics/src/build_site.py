"""
Build the website's data file.

Runs the full analysis pipeline and trains the prediction model, then writes
everything the static site needs into docs/data.json. Run this once whenever
the underlying flight data changes; the website re-renders automatically.

Usage:
    python src/build_site.py [path/to/flights.csv]
"""

import json
import sys
from pathlib import Path

import pandas as pd

import pipeline as P
import train_model as M

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "data" / "flights_2025_sample.csv"
OUT = ROOT / "docs" / "data.json"


def main(path: str):
    df = P.load(path)

    routes = P.route_efficiency(df)
    hourly = P.delay_by_hour(df)
    carriers = P.carrier_recovery(df)
    months = P.seasonality(df)
    model = M.train(df)

    # Per-route congestion so the client predictor can look it up by route.
    congestion = df.groupby("ROUTE")["DELAYED"].mean().round(4)

    route_records = []
    for name, row in routes.iterrows():
        route_records.append({
            "route": name,
            "score": row["score"],
            "avg_excess_air_min": row["avg_excess_air_min"],
            "avg_arr_delay": row["avg_arr_delay"],
            "pct_delayed": row["pct_delayed"],
            "distance": int(row["distance"]),
            "flights": int(row["flights"]),
            "congestion": float(congestion.get(name, df["DELAYED"].mean())),
        })

    payload = {
        "meta": {
            "total_flights": int(len(df)),
            "n_routes": int(df["ROUTE"].nunique()),
            "n_carriers": int(df["OP_UNIQUE_CARRIER"].nunique()),
            "date_start": df["FL_DATE"].min().strftime("%Y-%m-%d"),
            "date_end": df["FL_DATE"].max().strftime("%Y-%m-%d"),
            "avg_arr_delay": round(float(df["ARR_DELAY"].mean()), 1),
            "pct_delayed": round(float(df["DELAYED"].mean() * 100), 1),
            "best_hour": int(hourly["avg_dep_delay"].idxmin()),
            "worst_hour": int(hourly["avg_dep_delay"].idxmax()),
        },
        "hourly": [
            {"hour": int(h), "avg_dep_delay": r["avg_dep_delay"],
             "pct_delayed": r["pct_delayed"]}
            for h, r in hourly.iterrows()
        ],
        "routes": route_records,
        "carriers": [
            {"code": c, "name": r["carrier_name"],
             "avg_min_recovered": r["avg_min_recovered"],
             "avg_arr_delay_when_late": r["avg_arr_delay_when_late"],
             "delayed_flights": int(r["delayed_flights"])}
            for c, r in carriers.iterrows()
        ],
        "monthly": [
            {"month": int(m), "avg_arr_delay": r["avg_arr_delay"],
             "pct_delayed": r["pct_delayed"]}
            for m, r in months.iterrows()
        ],
        "model": model,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {OUT.relative_to(ROOT)}")
    print(f"  {payload['meta']['total_flights']:,} flights | "
          f"{payload['meta']['n_routes']} routes | "
          f"model AUC {model['metrics']['auc']}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else str(DEFAULT_DATA)
    main(path)

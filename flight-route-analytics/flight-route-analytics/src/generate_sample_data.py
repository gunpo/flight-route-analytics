"""
Generate realistic sample flight data matching the schema of the
US DOT Bureau of Transportation Statistics (BTS) On-Time Performance dataset.

Swap this out for real data: download monthly CSVs from
https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FGJ
and place them in /data — the analysis pipeline reads the same columns.
"""

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

# Major US routes with realistic distance (miles) and baseline congestion factor
ROUTES = [
    ("LAX", "SFO", 337, 1.3), ("LAX", "JFK", 2475, 1.1), ("LAX", "SEA", 954, 1.0),
    ("LAX", "LAS", 236, 1.2), ("LAX", "ORD", 1744, 1.4), ("SFO", "SEA", 679, 1.1),
    ("SFO", "ORD", 1846, 1.5), ("JFK", "MIA", 1089, 1.2), ("JFK", "ORD", 740, 1.4),
    ("ORD", "DFW", 802, 1.3), ("ATL", "MCO", 404, 1.1), ("ATL", "LGA", 762, 1.5),
    ("DEN", "PHX", 602, 1.0), ("DEN", "LAX", 862, 1.1), ("SEA", "ANC", 1448, 0.8),
    ("BOS", "DCA", 399, 1.3), ("EWR", "SFO", 2565, 1.4), ("IAH", "MIA", 964, 1.2),
]

CARRIERS = {
    "DL": 0.85, "UA": 1.05, "AA": 1.10, "WN": 0.95, "AS": 0.80, "B6": 1.25, "F9": 1.40,
}

def generate(n_flights: int = 60_000) -> pd.DataFrame:
    routes = [ROUTES[i] for i in rng.integers(0, len(ROUTES), n_flights)]
    origin = np.array([r[0] for r in routes])
    dest = np.array([r[1] for r in routes])
    distance = np.array([r[2] for r in routes], dtype=float)
    congestion = np.array([r[3] for r in routes])

    carriers = rng.choice(list(CARRIERS.keys()), n_flights)
    carrier_factor = np.array([CARRIERS[c] for c in carriers])

    dates = pd.to_datetime("2025-01-01") + pd.to_timedelta(
        rng.integers(0, 365, n_flights), unit="D"
    )
    dep_hour = rng.choice(range(5, 23), n_flights, p=_hour_distribution())

    # Delay model: later departures cascade, congestion & carrier matter,
    # plus seasonal (summer storms / winter weather) noise.
    hour_effect = np.clip((dep_hour - 6) * 1.8, 0, None)          # cascade through the day
    month = dates.month
    seasonal = np.where(np.isin(month, [6, 7, 8]), 8, 0) + np.where(
        np.isin(month, [12, 1, 2]), 6, 0
    )
    base = rng.gamma(shape=1.5, scale=6, size=n_flights)
    dep_delay = (base + hour_effect * congestion * 0.8 + seasonal * rng.random(n_flights)) \
        * carrier_factor - 5
    dep_delay = np.round(np.clip(dep_delay, -15, 360), 0)

    # Air time: distance-based cruise + congestion overhead at arrival
    air_time = distance / 7.8 + 25 + congestion * rng.normal(8, 3, n_flights)
    arr_delay = dep_delay + rng.normal(0, 9, n_flights) - np.where(
        dep_delay > 0, distance / 600, 0  # pilots make up time on long hauls
    )
    cancelled = (rng.random(n_flights) < 0.012 + (dep_delay > 120) * 0.05).astype(int)

    return pd.DataFrame({
        "FL_DATE": dates,
        "OP_UNIQUE_CARRIER": carriers,
        "ORIGIN": origin,
        "DEST": dest,
        "CRS_DEP_TIME": dep_hour * 100,
        "DEP_DELAY": dep_delay,
        "ARR_DELAY": np.round(arr_delay, 0),
        "AIR_TIME": np.round(air_time, 0),
        "DISTANCE": distance,
        "CANCELLED": cancelled,
    })

def _hour_distribution():
    # Flights cluster in morning and late-afternoon banks
    weights = np.array([4, 7, 8, 8, 7, 6, 6, 6, 5, 5, 6, 7, 7, 6, 5, 4, 3, 2], float)
    return weights / weights.sum()

if __name__ == "__main__":
    df = generate()
    df.to_csv("data/flights_2025_sample.csv", index=False)
    print(f"Generated {len(df):,} flights → data/flights_2025_sample.csv")

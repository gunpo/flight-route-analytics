"""
Analysis pipeline for US flight on-time performance data (BTS schema).

Pure functions that take a cleaned DataFrame and return analysis tables.
Imported by build_site.py (exports JSON) and usable standalone in notebooks.
"""

import numpy as np
import pandas as pd

DELAY_THRESHOLD = 15  # minutes; FAA/BTS standard for a "delayed" flight

CARRIER_NAMES = {
    "DL": "Delta", "UA": "United", "AA": "American", "WN": "Southwest",
    "AS": "Alaska", "B6": "JetBlue", "F9": "Frontier",
}


def load(path: str) -> pd.DataFrame:
    """Load and clean a BTS-schema CSV. Drops cancelled flights, derives fields."""
    df = pd.read_csv(path, parse_dates=["FL_DATE"])
    df = df[df["CANCELLED"] == 0].copy()
    df = df.dropna(subset=["ARR_DELAY", "DEP_DELAY", "AIR_TIME", "DISTANCE"])
    df["ROUTE"] = df["ORIGIN"] + "\u2192" + df["DEST"]
    df["DEP_HOUR"] = (df["CRS_DEP_TIME"] // 100).astype(int)
    df["MONTH"] = df["FL_DATE"].dt.month
    df["DELAYED"] = (df["ARR_DELAY"] > DELAY_THRESHOLD).astype(int)
    return df


def route_efficiency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Route Efficiency Score (0-100, higher = more reliable).

    Combines two standardized penalties:
      - excess air time: actual minus distance-predicted flight time
      - delay frequency: share of arrivals more than 15 min late
    """
    coeffs = np.polyfit(df["DISTANCE"], df["AIR_TIME"], 1)
    df = df.assign(
        EXPECTED_AIR_TIME=np.polyval(coeffs, df["DISTANCE"]),
    )
    df["EXCESS_AIR_TIME"] = df["AIR_TIME"] - df["EXPECTED_AIR_TIME"]

    g = df.groupby("ROUTE").agg(
        flights=("ROUTE", "size"),
        avg_excess_air_min=("EXCESS_AIR_TIME", "mean"),
        avg_arr_delay=("ARR_DELAY", "mean"),
        pct_delayed=("DELAYED", lambda s: s.mean() * 100),
        distance=("DISTANCE", "first"),
    ).round(1)

    def z(s):
        return (s - s.mean()) / s.std(ddof=0)

    raw = 50 - (z(g["avg_excess_air_min"]) * 15 + z(g["pct_delayed"]) * 15)
    # Rescale to a clean 0-100 band for readability
    g["score"] = (
        (raw - raw.min()) / (raw.max() - raw.min()) * 60 + 35
    ).round(1)
    return g.sort_values("score", ascending=False)


def delay_by_hour(df: pd.DataFrame) -> pd.DataFrame:
    """The delay cascade: how departure delay grows through the operating day."""
    return df.groupby("DEP_HOUR").agg(
        avg_dep_delay=("DEP_DELAY", "mean"),
        pct_delayed=("DELAYED", lambda s: s.mean() * 100),
        flights=("DEP_HOUR", "size"),
    ).round(1)


def carrier_recovery(df: pd.DataFrame) -> pd.DataFrame:
    """Minutes a carrier makes up in the air after departing more than 15 min late."""
    late = df[df["DEP_DELAY"] > DELAY_THRESHOLD].copy()
    late["recovered"] = late["DEP_DELAY"] - late["ARR_DELAY"]
    g = late.groupby("OP_UNIQUE_CARRIER").agg(
        delayed_flights=("recovered", "size"),
        avg_min_recovered=("recovered", "mean"),
        avg_arr_delay_when_late=("ARR_DELAY", "mean"),
    ).round(1)
    g["carrier_name"] = [CARRIER_NAMES.get(code, code) for code in g.index]
    return g.sort_values("avg_min_recovered", ascending=False)


def seasonality(df: pd.DataFrame) -> pd.DataFrame:
    """Average arrival delay by month."""
    return df.groupby("MONTH").agg(
        avg_arr_delay=("ARR_DELAY", "mean"),
        pct_delayed=("DELAYED", lambda s: s.mean() * 100),
    ).round(1)

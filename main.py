"""Forecast GTA VI's launch-day Twitch viewers hour by hour (main.py).

Method: for each hour h = 2..24, take every comparable past launch's
release-day curve, express it as log(viewers[h] / viewers[hour 1]), and
read the forecast quantiles off the weighted empirical distribution of
those log-ratios, where a comparable's weight is a Gaussian kernel on
how close its hour-1 audience is to this case's hour-1 audience (in
log space, bandwidth 1.8). Big launches behave like big launches, so
similarity in launch size is what the weights capture. The forecast is
hour-1 viewers times exp(quantile log-ratio), rounded to 2 decimals.

Backtest (leave-one-out over the 37 comparable launches in the kit):
mean pinball loss on the log scale 0.202, against 0.465 for the house
baseline that holds every hour flat at hour 1. Coverage of the five
quantiles is close to nominal (6% / 23% / 47% / 76% / 93% of actuals
fall below q05 / q25 / q50 / q75 / q95).

Standard library only. The grader runs `python3 -P -s /bot/main.py`
(PYTHONHASHSEED=0) in python:3.13-slim with no network.
"""

import json
import math
import os
import sys
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so `import
# model` finds a model.py shipped beside this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]

# Gaussian kernel bandwidth on log hour-1 viewers (tuned by backtest:
# 1.5 -> 0.2021, 1.8 -> 0.2019, 2.0 -> 0.2061, 3.0 -> 0.2289).
BANDWIDTH = 1.8

# Fallback spread (log units per sqrt hour) if there are too few
# comparables to read quantiles from; the house baseline uses 0.15.
FALLBACK_SIGMA = 0.15
FALLBACK_Z = [-1.645, -0.674, 0, 0.674, 1.645]

MIN_COMPARABLES = 5


def round2(x: float) -> float | int:
    """x rounded to 2 decimals, half up, as JavaScript's
    Math.round(x * 100) / 100 rounds it. A whole number comes back as
    an int, so json.dumps writes 1240 where JavaScript writes 1240
    (not 1240.0)."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    out = whole / 100
    return int(out) if out == int(out) else out


def weighted_quantile(pairs, tau: float) -> float:
    """Weighted empirical quantile of [(value, weight), ...]."""
    pairs = sorted(pairs)
    total = sum(w for _, w in pairs)
    if total <= 0:
        return pairs[len(pairs) // 2][0]
    target = tau * total
    acc = 0.0
    for value, weight in pairs:
        acc += weight
        if acc >= target:
            return value
    return pairs[-1][0]


def day_hours(hourly, day_start: str) -> dict:
    """Release-day {hour 1..24: viewers} from a comparable's hourly
    series, where hour 1 is [day_start, day_start + 1h)."""
    from datetime import datetime, timedelta

    start = datetime.fromisoformat(day_start.replace("Z", "+00:00"))
    out = {}
    for point in hourly or []:
        try:
            at = datetime.fromisoformat(point["from"].replace("Z", "+00:00"))
        except (KeyError, ValueError):
            continue
        if start <= at < start + timedelta(days=1):
            hour = (at - start).seconds // 3600 + 1
            out[hour] = max(float(point.get("viewers") or 0), 1.0)
    return out


def forecast(case: dict) -> list:
    """Return [{hour, q: [q05, q25, q50, q75, q95]}, ...] for hours 2..24."""
    known = case.get("known") or []
    hour1 = 1.0
    for row in known:
        if row.get("hour") == 1:
            hour1 = max(float(row.get("viewers") or 0), 1.0)
            break
    log_h1 = math.log(hour1)

    # Comparable launch shapes: log(viewers[h] / viewers[hour 1]).
    shapes = []  # [(log hour-1 viewers, {hour: log-ratio}), ...]
    for comp in case.get("comparables") or []:
        hours = day_hours(comp.get("hourly"), comp.get("day_start", ""))
        if len(hours) < 24 or hours.get(1, 0) < 1:
            continue
        v1 = hours[1]
        shapes.append(
            (math.log(v1), {h: math.log(hours[h] / v1) for h in range(2, 25)})
        )

    rows = []
    for h in range(2, 25):
        if len(shapes) >= MIN_COMPARABLES:
            pairs = [
                (shape[h], math.exp(-((log_v1 - log_h1) / BANDWIDTH) ** 2))
                for log_v1, shape in shapes
                if h in shape
            ]
            if len(pairs) >= MIN_COMPARABLES:
                log_qs = [weighted_quantile(pairs, tau) for tau in LEVELS]
            else:
                log_qs = None
        else:
            log_qs = None
        if log_qs is None:
            sigma = FALLBACK_SIGMA * math.sqrt(h - 1)
            log_qs = [z * sigma for z in FALLBACK_Z]
        qs = [hour1 * math.exp(q) for q in log_qs]
        # Enforce finite, positive, capped and non-decreasing after rounding.
        qs = [min(max(round2(q), 0.01), 1e9) for q in qs]
        for i in range(1, 5):
            if qs[i] < qs[i - 1]:
                qs[i] = qs[i - 1]
        rows.append({"hour": h, "q": qs})
    return rows


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for in_path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(in_path.read_text())
        out = {"hours": forecast(case)}
        out_path = OUTPUT_DIR / (in_path.stem + ".out")
        text = json.dumps(out)
        assert len(text.encode()) <= 65536, "output over 65,536 bytes"
        out_path.write_text(text)


if __name__ == "__main__":
    main()

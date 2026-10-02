"""Bundled course data: measurements generated from stated models with documented faults
(see tools/make_course_data.py).

>>> from engstoch import datasets
>>> datasets.available()
['call_centre.csv', 'machine_log.csv', 'prices.csv', 'weather.csv']
"""

from __future__ import annotations

from importlib import resources

INFO = {
    "call_centre.csv": "20 weekdays of calls to a service line: arrival times and handle times, with an outage and dropped calls (notebooks 09, 12 and the project).",
    "weather.csv": "Ten years of daily wet/dry indicators at one station, with seasonally varying persistence (notebooks 05 and 06).",
    "machine_log.csv": "Two years of a machine's up/degraded/down state changes, the last sojourn censored (notebook 11).",
    "prices.csv": "1501 daily closing prices from geometric Brownian motion with a volatility change (notebooks 15 and 16).",
}


def path(name: str):
    """Path-like object of a bundled data file (pass it to pandas.read_csv with comment='#')."""
    if name not in INFO:
        raise KeyError(f"Unknown dataset {name!r}. Available: {', '.join(sorted(INFO))}")
    return resources.files("engstoch.data") / name


def available() -> list[str]:
    """Names of the bundled data files."""
    return sorted(INFO)


def describe(name: str) -> str:
    """One-line description of a bundled data file."""
    return INFO[name]

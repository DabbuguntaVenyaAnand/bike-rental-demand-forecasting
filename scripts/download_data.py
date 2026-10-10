"""Download the UCI Bike Sharing hourly dataset into data/hour.csv.

Usage:
    .venv/Scripts/python scripts/download_data.py
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "hour.csv"
URL = (
    "https://raw.githubusercontent.com/danwild/bike-share-prediction/"
    "master/Bike-Sharing-Dataset/hour.csv"
)

EXPECTED_HEADER = "instant,dteday,season,yr,mnth,hr,holiday,weekday,workingday,weathersit,temp,atemp,hum,windspeed,casual,registered,cnt"


def main() -> None:
    if DEST.exists():
        header = DEST.read_text(encoding="utf-8").splitlines()[0].strip()
        if header == EXPECTED_HEADER:
            print(f"Already present: {DEST}")
            return
        print("Existing file has unexpected header; re-downloading...")
    DEST.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(URL, DEST)
    header = DEST.read_text(encoding="utf-8").splitlines()[0].strip()
    assert header == EXPECTED_HEADER, f"Unexpected header: {header}"
    rows = len(DEST.read_text(encoding="utf-8").splitlines()) - 1
    print(f"Downloaded {DEST} ({rows:,} rows)")


if __name__ == "__main__":
    main()

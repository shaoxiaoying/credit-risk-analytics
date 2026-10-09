"""Download the official UCI dataset into data/raw/ without committing it."""
from pathlib import Path
import argparse
from ucimlrepo import fetch_ucirepo

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Fetch again if the source file already exists")
    args = parser.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    output = RAW / "credit_default_source.csv"
    if output.exists() and not args.force:
        print(f"Already downloaded: {output}")
        return
    # ucimlrepo fetches from the UCI repository; use the original tabular data.
    dataset = fetch_ucirepo(id=350)
    frame = dataset.data.original
    if frame is None or frame.empty:
        raise RuntimeError("UCI returned no original dataset rows.")
    frame.to_csv(output, index=False)
    print(f"Saved {len(frame):,} rows to {output}")


if __name__ == "__main__":
    main()

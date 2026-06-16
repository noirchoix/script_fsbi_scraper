import argparse
import sys
import pandas as pd
from functions import (
    flavordb_df_cols,
    molecules_df_cols,
    load_db,
)
# Access data
def make_query():
    df = pd.read_csv("data/fragrance_food_sources.csv")
    return df["common_name"].dropna().tolist()

def chunk_ranges(start: int, end: int, chunk: int):
    """Yield (a, b) index slices for batching."""
    a = start
    while a < end:
        b = min(a + chunk, end)
        yield a, b
        a = b

def _set_runtime_flags():
    # Optional runtime toggles for verbose errors without editing functions.py
    try:
        import functions as _fx
        _fx.VERBOSE_ERRORS = True
        _fx.FSBI_DEBUG = True
    except Exception:
        pass

def run_fsbi(args) -> int:
    from functions import build_fsbi_dataframes, missing_entity_ids
    # queries = [q.strip() for q in (args.discover_terms or "").split(",") if q.strip()] or None
    queries = make_query()
    for a, b in chunk_ranges(0, len(queries), 50):  # batch size = 500
        batch = queries[a:b]
        print(f"Processing batch {a}-{b} of {len(queries)}: {batch}")

    try:
        flavor_df, molecules_df = build_fsbi_dataframes(
            compound_urls=None,
            food_urls=None,
            discovery_queries=batch,
            max_compounds=args.max_compounds,
            delay=args.delay,
        )
    except Exception as e:
        print("FSBI scrape halted with an error:", e, file=sys.stderr)
        flavor_df = pd.DataFrame(columns=flavordb_df_cols())
        molecules_df = pd.DataFrame(columns=molecules_df_cols())

    try:
        flavor_df.to_csv(args.out_flavordb_csv, index=False)
        molecules_df.to_csv(args.out_molecules_csv, index=False)
    except Exception as e:
        print("Failed to write CSVs:", e, file=sys.stderr)
        return 2

    print(f"\nSaved '{args.out_flavordb_csv}' and '{args.out_molecules_csv}'.")
    print("Rows:", len(flavor_df), "foods,", len(molecules_df), "molecules.")
    if len(flavor_df):
        print("Missing IDs (subset):", missing_entity_ids(flavor_df)[:50])
    return 0

def main() -> int:
    parser = argparse.ArgumentParser(description="Scrape FSBI-DB into CSVs.")
    # FSBI options
    parser.add_argument("--delay", type=float, default=0.25, help="Delay between requests.")
    parser.add_argument("--max_compounds", type=int, default=500, help="Max compounds to process.")
    parser.add_argument("--discover_terms", type=str, default="", help="Comma-separated discovery search terms.")
    # Outputs
    parser.add_argument("--out_flavordb_csv", type=str, default="flavordb.csv")
    parser.add_argument("--out_molecules_csv", type=str, default="molecules.csv")
    args = parser.parse_args()

    _set_runtime_flags()

    return run_fsbi(args)

if __name__ == "__main__":
    raise SystemExit(main())
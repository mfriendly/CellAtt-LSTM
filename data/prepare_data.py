"""Prepare unified data files for TriAtt26 from source datasets.

Reads from original locations and writes clean per-region CSVs into
data/raw/{dataset}/{nation}/{ABBR}.csv with unified column names.

Usage:
  python data/prepare_data.py --all
  python data/prepare_data.py --dengue
  python data/prepare_data.py --covid
"""
import argparse
import json
import os
import shutil
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"

# ─────────────────── Source paths (set via env or CLI) ───────────────────
# Override with: export CELLATT_COVID_SRC=... CELLATT_DENGUE_SRC=...
COVID_SRC = Path(os.environ.get("CELLATT_COVID_SRC", ROOT / "data" / "source" / "covid"))
DENGUE_SRC = Path(os.environ.get("CELLATT_DENGUE_SRC", ROOT / "data" / "source" / "dengue"))

DENGUE_EXOG_COLS = ["tempmin", "tempmed", "tempmax", "umidmin", "umidmed", "umidmax"]

# COVID exog columns to extract from each source (new/flow only, skip cumulative)
COVID_EPI_EXOG = ["new_deceased", "new_recovered", "new_tested"]
COVID_HOSP_COLS = ["new_hospitalized_patients", "new_intensive_care_patients", "new_ventilator_patients"]
COVID_VAX_COLS = ["new_persons_vaccinated", "new_persons_fully_vaccinated", "new_vaccine_doses_administered"]
COVID_MOBILITY_COLS = [
    "mobility_retail_and_recreation", "mobility_grocery_and_pharmacy",
    "mobility_parks", "mobility_transit_stations",
    "mobility_workplaces", "mobility_residential",
]


# ═══════════════════════════ COVID ═══════════════════════════
def _load_covid_source(path, date_col="date", usecols=None):
    """Load a single COVID source CSV, parse date, drop location_key."""
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df[date_col] = pd.to_datetime(df[date_col])
    drop = [c for c in ["location_key"] if c in df.columns]
    df = df.drop(columns=drop)
    if usecols is not None:
        keep = [date_col] + [c for c in usecols if c in df.columns]
        df = df[keep]
    return df.sort_values(date_col).reset_index(drop=True)


def prepare_covid(nation):
    """Merge 5 COVID data sources into unified per-region CSVs.

    Sources: epidemiology, google-search-trends, hospitalizations,
             mobility (US only), vaccinations.
    """
    epi_dir = COVID_SRC / "x_data_epidemiology" / nation
    search_dir = COVID_SRC / "x_data_google-search-trends" / nation
    hosp_dir = COVID_SRC / "x_data_hospitalizations" / nation
    mob_dir = COVID_SRC / "x_data_mobility" / nation
    vax_dir = COVID_SRC / "x_data_vaccinations" / nation
    aux_dir = COVID_SRC / "x_data_aux" / nation
    out_dir = DATA_RAW / "covid" / nation
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(aux_dir / "abbr2id.json") as f:
        abbr2id = json.load(f)
    with open(aux_dir / "config.json") as f:
        date_config = json.load(f)

    converted = 0
    all_exog_cols = set()

    for abbr, node_id in abbr2id.items():
        # 1) Epidemiology (base)
        epi_path = epi_dir / f"{str(node_id).zfill(2)}_{abbr}.csv"
        if not epi_path.exists():
            print(f"  SKIP {nation}/{abbr}: epi not found")
            continue
        base = pd.read_csv(epi_path)
        base["date"] = pd.to_datetime(base["date"])
        base = base.drop(columns=["location_key"], errors="ignore")
        keep_epi = ["date", "new_confirmed"] + [c for c in COVID_EPI_EXOG if c in base.columns]
        base = base[keep_epi].rename(columns={"new_confirmed": "target"})
        base = base.sort_values("date").reset_index(drop=True)

        # 2) Google Search Trends (422 cols)
        search_df = _load_covid_source(search_dir / f"{nation}_{abbr}.csv")

        # 3) Hospitalizations
        hosp_df = _load_covid_source(hosp_dir / f"{nation}_{abbr}.csv", usecols=COVID_HOSP_COLS)

        # 4) Mobility (US only)
        mob_df = _load_covid_source(mob_dir / f"{nation}_{abbr}.csv", usecols=COVID_MOBILITY_COLS)

        # 5) Vaccinations
        vax_df = _load_covid_source(vax_dir / f"{nation}_{abbr}.csv", usecols=COVID_VAX_COLS)

        # Merge all on date (left join on base)
        df = base.copy()
        for src_df in [search_df, hosp_df, mob_df, vax_df]:
            if src_df is not None and len(src_df) > 0:
                df = df.merge(src_df, on="date", how="left")

        # NaN handling per source type
        exog_cols = [c for c in df.columns if c not in ("date", "target")]

        # Vaccinations: fill with 0 before vaccination program started
        for col in COVID_VAX_COLS:
            if col in df.columns:
                df[col] = df[col].fillna(0.0)

        # Others: interpolate + ffill + bfill
        for col in exog_cols:
            if col not in COVID_VAX_COLS and col in df.columns:
                df[col] = df[col].interpolate(method="linear").ffill().bfill()

        # Drop columns that are still all-NaN after interpolation
        still_nan = [c for c in exog_cols if c in df.columns and df[c].isna().all()]
        if still_nan:
            df = df.drop(columns=still_nan)

        df = df.sort_values("date").reset_index(drop=True)
        df.to_csv(out_dir / f"{abbr}.csv", index=False)
        converted += 1
        all_exog_cols.update(c for c in df.columns if c not in ("date", "target"))

        n_exog = len([c for c in df.columns if c not in ("date", "target")])
        print(f"    {abbr}: {len(df)} rows, {n_exog} exog cols")

    # metadata
    meta = {
        "dataset": "covid",
        "nation": nation,
        "granularity": "daily",
        "target_col": "new_confirmed",
        "regions": sorted(abbr2id.keys()),
        "abbr2id": abbr2id,
        "dates": date_config,
        "n_regions": len(abbr2id),
        "exog_sources": ["epidemiology", "google-search-trends",
                         "hospitalizations", "mobility", "vaccinations"],
        "n_exog_cols": len(all_exog_cols),
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  COVID {nation}: {converted} regions, ~{len(all_exog_cols)} exog cols → {out_dir}")

    # Add cross-region case columns
    add_cross_region_cases(out_dir, sorted(abbr2id.keys()))


# ═══════════════════════════ DENGUE ═══════════════════════════
CAPITALS = {
    "AC": 1200401, "AL": 2704302, "AM": 1302603, "AP": 1600303,
    "BA": 2927408, "CE": 2304400, "DF": 5300108, "ES": 3205309,
    "GO": 5208707, "MA": 2111300, "MG": 3106200, "MS": 5002704,
    "MT": 5103403, "PA": 1501402, "PB": 2507507, "PE": 2611606,
    "PI": 2211001, "PR": 4106902, "RJ": 3304557, "RN": 2408102,
    "RO": 1100205, "RR": 1400100, "RS": 4314902, "SC": 4205407,
    "SE": 2800308, "SP": 3550308, "TO": 1721000,
}


def prepare_dengue():
    """Convert dengue raw CSVs to unified format + copy CFS correlation files."""
    raw_src = DENGUE_SRC / "raw"
    proc_src = DENGUE_SRC / "processed"
    out_dir = DATA_RAW / "dengue" / "BR"
    corr_dir = out_dir / "correlation"
    out_dir.mkdir(parents=True, exist_ok=True)
    corr_dir.mkdir(parents=True, exist_ok=True)

    converted = 0
    date_min, date_max = None, None

    for uf, geocode in sorted(CAPITALS.items()):
        src = raw_src / f"{uf}_{geocode}.csv"
        if not src.exists():
            print(f"  SKIP dengue/BR/{uf}: {src} not found")
            continue
        df = pd.read_csv(src)
        keep_cols = ["data_iniSE", "casos"] + [c for c in DENGUE_EXOG_COLS if c in df.columns]
        df = df[keep_cols].copy()
        df = df.rename(columns={"data_iniSE": "date", "casos": "target"})
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        # fill missing exog with interpolation
        for col in DENGUE_EXOG_COLS:
            if col in df.columns:
                df[col] = df[col].interpolate(limit=4).ffill().bfill()

        df.to_csv(out_dir / f"{uf}.csv", index=False)
        converted += 1

        if date_min is None or df["date"].min() < date_min:
            date_min = df["date"].min()
        if date_max is None or df["date"].max() > date_max:
            date_max = df["date"].max()

    # copy correlation / CFS files
    corr_files = [
        "pairwise_lag_corr.csv", "per_state_best_corr.csv",
        "exog_lag_corr.csv", "cfs_selected_features.csv",
        "case_weekly_raw.csv", "case_weekly_matrix.csv",
    ]
    copied = 0
    for fname in corr_files:
        src = proc_src / fname
        if src.exists():
            shutil.copy2(src, corr_dir / fname)
            copied += 1

    # metadata
    meta = {
        "dataset": "dengue",
        "nation": "BR",
        "granularity": "weekly",
        "target_col": "casos",
        "exog_cols": DENGUE_EXOG_COLS,
        "regions": sorted(CAPITALS.keys()),
        "n_regions": len(CAPITALS),
        "date_range": [str(date_min.date()) if date_min else "",
                       str(date_max.date()) if date_max else ""],
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  Dengue BR: {converted} regions → {out_dir}")
    print(f"  Correlation files: {copied} → {corr_dir}")

    # Add cross-region case columns
    add_cross_region_cases(out_dir, sorted(CAPITALS.keys()))


# ═══════════════════════════ CROSS-REGION CASES ═══════════════════════════
def add_cross_region_cases(out_dir, regions):
    """Add other regions' target as cases_{ABBR} columns to each region CSV.

    For region X, adds columns cases_Y, cases_Z, ... from every other region,
    aligned by date. mRMR will later select the most relevant ones.
    """
    # Load all region targets into a single date-indexed DataFrame
    all_targets = {}
    for abbr in regions:
        csv_path = out_dir / f"{abbr}.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path, usecols=["date", "target"])
        df["date"] = pd.to_datetime(df["date"])
        all_targets[abbr] = df.set_index("date")["target"]

    if len(all_targets) < 2:
        print("  Skip cross-region: fewer than 2 regions")
        return

    cases_matrix = pd.DataFrame(all_targets)
    cases_matrix = cases_matrix.sort_index()

    added = 0
    for abbr in regions:
        csv_path = out_dir / f"{abbr}.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        df["date"] = pd.to_datetime(df["date"])
        # Build cross-region columns (all regions except self)
        other_cols = {f"cases_{other}": cases_matrix[other]
                      for other in regions if other != abbr and other in cases_matrix.columns}
        cross_df = pd.DataFrame(other_cols)
        cross_df.index.name = "date"
        cross_df = cross_df.reset_index()
        # Merge on date
        df = df.merge(cross_df, on="date", how="left")
        # Interpolate cross-region NaN
        for col in other_cols:
            if col in df.columns:
                df[col] = df[col].interpolate(method="linear").ffill().bfill()
        df.to_csv(csv_path, index=False)
        added += 1

    print(f"  Cross-region cases: added {len(regions)-1} cols to {added} region CSVs")


# ═══════════════════════════ MAIN ═══════════════════════════
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--all", action="store_true")
    p.add_argument("--covid", action="store_true")
    p.add_argument("--dengue", action="store_true")
    args = p.parse_args()

    if args.all or args.covid:
        print("=== Preparing COVID US ===")
        prepare_covid("US")
        print("=== Preparing COVID AU ===")
        prepare_covid("AU")

    if args.all or args.dengue:
        print("=== Preparing Dengue BR ===")
        prepare_dengue()

    if not (args.all or args.covid or args.dengue):
        print("Usage: python data/prepare_data.py --all | --covid | --dengue")


if __name__ == "__main__":
    main()

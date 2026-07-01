"""Download weekly dengue data for 27 Brazilian state capitals from InfoDengue API."""
import time
import argparse
from pathlib import Path
from io import StringIO
import pandas as pd
import requests

API_URL = "https://info.dengue.mat.br/api/alertcity"

CAPITALS = {
    "AC": ("Rio Branco", 1200401), "AL": ("Maceió", 2704302),
    "AM": ("Manaus", 1302603), "AP": ("Macapá", 1600303),
    "BA": ("Salvador", 2927408), "CE": ("Fortaleza", 2304400),
    "DF": ("Brasília", 5300108), "ES": ("Vitória", 3205309),
    "GO": ("Goiânia", 5208707), "MA": ("São Luís", 2111300),
    "MG": ("Belo Horizonte", 3106200), "MS": ("Campo Grande", 5002704),
    "MT": ("Cuiabá", 5103403), "PA": ("Belém", 1501402),
    "PB": ("João Pessoa", 2507507), "PE": ("Recife", 2611606),
    "PI": ("Teresina", 2211001), "PR": ("Curitiba", 4106902),
    "RJ": ("Rio de Janeiro", 3304557), "RN": ("Natal", 2408102),
    "RO": ("Porto Velho", 1100205), "RR": ("Boa Vista", 1400100),
    "RS": ("Porto Alegre", 4314902), "SC": ("Florianópolis", 4205407),
    "SE": ("Aracaju", 2800308), "SP": ("São Paulo", 3550308),
    "TO": ("Palmas", 1721000),
}

EXOGENOUS_COLS = ["tempmin", "tempmed", "tempmax", "umidmin", "umidmed", "umidmax"]


def fetch_city(geocode, ey_start, ey_end):
    params = {"geocode": geocode, "disease": "dengue", "format": "csv",
              "ew_start": 1, "ew_end": 53, "ey_start": ey_start, "ey_end": ey_end}
    resp = requests.get(API_URL, params=params, timeout=60)
    resp.raise_for_status()
    return pd.read_csv(StringIO(resp.text))


def download_all(raw_dir, ey_start=2015, ey_end=2024):
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    all_frames = []
    for uf, (city, geocode) in CAPITALS.items():
        out_path = raw_dir / f"{uf}_{geocode}.csv"
        if out_path.exists():
            print(f"  {uf} ({city}): cached")
            df = pd.read_csv(out_path)
        else:
            print(f"  {uf} ({city}): fetching {ey_start}-{ey_end}...")
            df = fetch_city(geocode, ey_start, ey_end)
            df.to_csv(out_path, index=False)
            time.sleep(1)
        df["UF"] = uf
        df["city"] = city
        all_frames.append(df)

    combined = pd.concat(all_frames, ignore_index=True)
    combined.to_csv(raw_dir / "all_capitals.csv", index=False)
    print(f"Total: {len(combined)} rows, {combined['UF'].nunique()} states")
    return combined


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=str, default="data/raw/dengue/BR")
    parser.add_argument("--ey-start", type=int, default=2015)
    parser.add_argument("--ey-end", type=int, default=2024)
    a = parser.parse_args()
    download_all(a.raw_dir, a.ey_start, a.ey_end)

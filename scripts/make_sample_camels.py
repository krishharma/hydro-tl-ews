#!/usr/bin/env python3
"""Create a tiny CAMELS-compatible sample under ``data/sample_camels/``.

Generates ~8 synthetic basins with ~15 years of daily Daymet-style forcings
and USGS-style streamflow so the real ``CamelsDataset`` loader can be tested
in a few minutes without downloading the full ~14 GB CAMELS-US archive.

Usage::

    python scripts/make_sample_camels.py
    python scripts/run_sample_pipeline.py
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "sample_camels"

# 8 basins: 1 target (snowmelt-like) + 7 donors. Gauge IDs look like CAMELS.
BASINS = [
    # id, huc2, elev, slope, area_km2, frac_snow, aridity, lat, lon
    ("11264500", "18", 2200.0, 25.0, 460.0, 0.85, 0.9, 37.73, -119.56),  # target
    ("09107000", "14", 2800.0, 18.0, 320.0, 0.70, 1.1, 38.90, -106.95),
    ("14222500", "17", 400.0, 12.0, 210.0, 0.15, 0.6, 45.85, -122.50),
    ("02128000", "03", 250.0, 8.0, 180.0, 0.02, 0.8, 35.20, -79.80),
    ("01544500", "02", 350.0, 10.0, 150.0, 0.25, 0.7, 41.40, -77.90),
    ("11224500", "18", 500.0, 15.0, 90.0, 0.05, 2.2, 36.95, -121.40),
    ("05507600", "07", 200.0, 4.0, 280.0, 0.10, 1.0, 38.60, -91.20),
    ("01013500", "01", 300.0, 9.0, 220.0, 0.35, 0.75, 45.20, -69.30),
]

START = "1995-01-01"
END = "2009-12-31"  # 15 years — enough for thresholds (>=20 yr not required for sample)


def _attr_rows() -> dict[str, pd.DataFrame]:
    rows = []
    for bid, huc, elev, slope, area, fsnow, arid, lat, lon in BASINS:
        rows.append(
            {
                "gauge_id": bid,
                "gauge_lat": lat,
                "gauge_lon": lon,
                "elev_mean": elev,
                "slope_mean": slope,
                "area_gages2": area,
                "huc_02": huc,
                "p_mean": 3.0 + 0.5 * fsnow,
                "pet_mean": 2.5 * arid,
                "p_seasonality": 0.3,
                "frac_snow": fsnow,
                "aridity": arid,
                "high_prec_freq": 20.0,
                "high_prec_dur": 1.5,
                "low_prec_freq": 40.0,
                "low_prec_dur": 5.0,
                "frac_forest": 0.4 + 0.2 * fsnow,
                "lai_max": 3.0,
                "lai_diff": 1.5,
                "gvf_max": 0.7,
                "gvf_diff": 0.3,
                "soil_depth_pelletier": 5.0,
                "soil_depth_statsgo": 1.5,
                "soil_porosity": 0.45,
                "soil_conductivity": 1.2,
                "max_water_content": 0.3,
                "sand_frac": 40.0,
                "silt_frac": 35.0,
                "clay_frac": 25.0,
                "carbonate_rocks_frac": 0.1,
                "geol_permeability": -12.0,
                "q_mean": 1.5,
                "runoff_ratio": 0.4,
            }
        )
    df = pd.DataFrame(rows).set_index("gauge_id")
    groups = {
        "camels_topo.txt": ["gauge_lat", "gauge_lon", "elev_mean", "slope_mean", "area_gages2"],
        "camels_clim.txt": [
            "p_mean", "pet_mean", "p_seasonality", "frac_snow", "aridity",
            "high_prec_freq", "high_prec_dur", "low_prec_freq", "low_prec_dur",
        ],
        "camels_hydro.txt": ["q_mean", "runoff_ratio"],
        "camels_vege.txt": ["frac_forest", "lai_max", "lai_diff", "gvf_max", "gvf_diff"],
        "camels_soil.txt": [
            "soil_depth_pelletier", "soil_depth_statsgo", "soil_porosity",
            "soil_conductivity", "max_water_content", "sand_frac", "silt_frac", "clay_frac",
        ],
        "camels_geol.txt": ["carbonate_rocks_frac", "geol_permeability"],
    }
    out = {}
    for fname, cols in groups.items():
        part = df[cols].reset_index()
        out[fname] = part
    return out


def _simulate_basin(meta, dates: pd.DatetimeIndex, rng: np.random.Generator):
    bid, huc, elev, slope, area, fsnow, arid, lat, lon = meta
    doy = dates.dayofyear.to_numpy()
    # Seasonal precip / temp driven by snow fraction.
    prcp = np.clip(
        rng.gamma(1.2, 2.0, size=len(dates))
        * (1.0 + 0.6 * np.sin(2 * math.pi * (doy - 50) / 365.0) * (1 - fsnow)
           + 0.8 * np.sin(2 * math.pi * (doy - 320) / 365.0) * fsnow),
        0, None,
    )
    tmax = 12.0 - 0.005 * elev + 12 * np.sin(2 * math.pi * (doy - 110) / 365.0) + rng.normal(0, 1.5, len(dates))
    tmin = tmax - (8.0 + 4.0 * fsnow) + rng.normal(0, 0.8, len(dates))
    srad = np.clip(180 + 80 * np.sin(2 * math.pi * (doy - 80) / 365.0) + rng.normal(0, 15, len(dates)), 20, None)
    vp = np.clip(600 + 20 * tmin + rng.normal(0, 30, len(dates)), 50, None)
    dayl = 43200 + 8000 * np.sin(2 * math.pi * (doy - 80) / 365.0)

    # Simple rainfall-runoff + delayed snowmelt.
    q = np.zeros(len(dates))
    store = 20.0
    snowpack = 50.0 * fsnow
    for t in range(len(dates)):
        rain = prcp[t] * (1 - fsnow * max(0.0, min(1.0, (2.0 - tmin[t]) / 6.0)))
        snow = prcp[t] - rain
        snowpack += snow
        melt = 0.0
        if tmax[t] > 0:
            melt = min(snowpack, max(0.0, 2.5 * tmax[t]) * fsnow)
            snowpack -= melt
        store = max(0.0, store + rain + melt - 1.2 * arid)
        runoff = 0.35 * (rain + melt) + 0.08 * store
        q[t] = max(0.01, runoff + rng.normal(0, 0.15))

    forc = pd.DataFrame(
        {
            "Year": dates.year,
            "Mnth": dates.month,
            "Day": dates.day,
            "Hr": 12,
            "Dayl(s)": dayl,  # unused by loader but often present
            "prcp(mm/day)": prcp,
            "srad(W/m2)": srad,
            "swe(mm)": snowpack,  # unused
            "tmax(C)": tmax,
            "tmin(C)": tmin,
            "vp(Pa)": vp,
            "dayl(s)": dayl,
        },
        index=dates,
    )
    # Convert q mm/day -> cfs for the CAMELS streamflow file format.
    mm_to_cfs = (area * 1e6) * 1000.0 / 86400.0 / 28316.846592
    flow = pd.DataFrame(
        {
            "basin": bid,
            "Year": dates.year,
            "Mnth": dates.month,
            "Day": dates.day,
            "QObs(cfs)": q * mm_to_cfs,
            "flag": "A",
        },
        index=dates,
    )
    return huc, forc, flow


def write_forcing(path: Path, forc: pd.DataFrame, lat: float, lon: float, elev: float, area: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    # 3 metadata rows then whitespace table (loader uses skiprows=3).
    with path.open("w", encoding="utf-8") as f:
        f.write(f"{lat} {lon} basin centroid\n")
        f.write(f"Elevation: {elev} m  Area: {area} km2\n")
        f.write("\n")
        cols = [
            "Year", "Mnth", "Day", "Hr",
            "dayl(s)", "prcp(mm/day)", "srad(W/m2)", "swe(mm)",
            "tmax(C)", "tmin(C)", "vp(Pa)",
        ]
        f.write(" ".join(cols) + "\n")
        for _, row in forc.iterrows():
            f.write(
                f"{int(row['Year'])} {int(row['Mnth'])} {int(row['Day'])} {int(row['Hr'])} "
                f"{row['dayl(s)']:.1f} {row['prcp(mm/day)']:.3f} {row['srad(W/m2)']:.2f} "
                f"{row.get('swe(mm)', 0.0):.2f} {row['tmax(C)']:.2f} {row['tmin(C)']:.2f} "
                f"{row['vp(Pa)']:.1f}\n"
            )


def write_streamflow(path: Path, flow: pd.DataFrame):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for _, row in flow.iterrows():
            f.write(
                f"{row['basin']} {int(row['Year'])} {int(row['Mnth'])} {int(row['Day'])} "
                f"{row['QObs(cfs)']:.4f} {row['flag']}\n"
            )


def main() -> None:
    rng = np.random.default_rng(42)
    dates = pd.date_range(START, END, freq="D")
    OUT.mkdir(parents=True, exist_ok=True)

    attr_dir = OUT / "camels_attributes_v2.0"
    attr_dir.mkdir(parents=True, exist_ok=True)
    for fname, df in _attr_rows().items():
        df.to_csv(attr_dir / fname, sep=";", index=False)

    for meta in BASINS:
        huc, forc, flow = _simulate_basin(meta, dates, rng)
        bid = meta[0]
        elev, area, lat, lon = meta[2], meta[4], meta[7], meta[8]
        write_forcing(
            OUT / "basin_dataset_public_v1p2" / "basin_mean_forcing" / "daymet" / huc
            / f"{bid}_lump_cida_forcing_leap.txt",
            forc, lat, lon, elev, area,
        )
        write_streamflow(
            OUT / "basin_dataset_public_v1p2" / "usgs_streamflow" / huc
            / f"{bid}_streamflow_qc.txt",
            flow,
        )

    readme = OUT / "README.md"
    readme.write_text(
        "# Sample CAMELS-compatible dataset\n\n"
        "Synthetic mini-archive for local testing (not real CAMELS observations).\n\n"
        f"- Basins: {len(BASINS)}\n"
        f"- Period: {START} → {END}\n"
        "- Target basin for quick configs: `11264500`\n\n"
        "Regenerate with `python scripts/make_sample_camels.py`.\n"
        "Run with `python scripts/run_sample_pipeline.py`.\n",
        encoding="utf-8",
    )
    print(f"Wrote sample CAMELS layout under {OUT}")
    print(f"  basins={len(BASINS)}  days={len(dates)}  target=11264500")


if __name__ == "__main__":
    main()

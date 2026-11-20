"""Sensitivity to the burned-area threshold (2019 test year, benchmark patches).

The models were trained to forecast any burned area (> 0 ha). Here the target is redefined as
larger burns: burned fraction of the 0.25-degree cell >= 1 %, and burned area >= 1,000 ha (10 km2).
Forecasts are scored without retraining against two climatologies: the any-burn climatology used by
Pyree, and a climatology rebuilt for the larger-fire target (same w = 3, tau = 4 yr, 2002-2017).
Also reports how much of a 0.25-degree cell a recorded burn typically covers.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.metrics import average_precision_score

from common import LEADS, OUT, ROOT, RUNS
from mapdata import TILE, load

sys.path.insert(0, str(ROOT))
from pyree.bootstrap import ap_from_counts, block_hist  # noqa: E402
from pyree.fire_prior import FirePrior  # noqa: E402

ZARR = os.environ.get("SEASFIRE_ZARR", str(ROOT / "data/seasfire_v0.3/SeasFireCube_v3.zarr"))
N_BOOT = 1000


def cell_area_ha(lat):
    r = 6371.0
    d = np.deg2rad(0.25)
    phi = np.deg2rad(lat)
    km2 = r ** 2 * d * np.abs(np.sin(phi + d / 2) - np.sin(phi - d / 2))
    return (km2 * 100).astype(np.float32)


def main():
    md = load()
    n_t = len(md["times"])
    times = pd.to_datetime(md["times"])
    ba = xr.open_zarr(ZARR)["gwis_ba"].isel(time=slice(0, n_t)).fillna(0).values.astype(np.float32)
    frac = ba / cell_area_ha(md["lat"])[None, :, None]
    yrs = times.year
    burned = (ba > 0) & ((yrs >= 2002) & (yrs <= 2019))[:, None, None]
    fb = frac[burned]
    stats = {"n_burned_cells": int(burned.sum()),
             "median_burned_fraction": float(np.median(fb)),
             "share_below_1pct": float((fb < 0.01).mean()),
             "share_below_10pct": float((fb < 0.10).mean()),
             "share_above_50pct": float((fb >= 0.50).mean())}
    print("burned-cell coverage", stats, flush=True)
    out = {"coverage": stats}
    tiles = md["tiles_test"]
    rng = np.random.default_rng(0)
    for name, fire_thr in [("frac1pct", (frac >= 0.01)), ("ha1000", (ba >= 1000))]:
        fire_thr = fire_thr.astype(np.uint8)
        prior = FirePrior(fire_thr, times, sigma=0, w=3, recency_tau=4.0)
        res = {}
        for h in LEADS:
            zo = np.load(RUNS / "v4_noanchor" / f"test_preds.npz")
            zt = np.load(RUNS / f"televit_orig_h{h}" / f"test_preds.npz")
            assert np.array_equal(zo[f"T_h{h}"], tiles[:, 0]) and np.array_equal(zt[f"T_h{h}"], tiles[:, 0])
            sl = [(slice(i * TILE, (i + 1) * TILE), slice(j * TILE, (j + 1) * TILE)) for _, i, j in tiles]
            y = np.concatenate([fire_thr[T][a, b].ravel() for (T, _, _), (a, b) in zip(tiles, sl)])
            cthr = np.concatenate([prior.clim(T)[a, b].ravel() for (T, _, _), (a, b) in zip(tiles, sl)])
            preds = {"pyree": zo[f"p_h{h}"].reshape(-1), "televit": zt[f"p_h{h}"].reshape(-1),
                     "clim_anyburn": zo[f"c_h{h}"].reshape(-1), "clim_threshold": cthr}
            d = {"base_rate": float(y.mean()), "n_pos": int(y.sum())}
            for k, v in preds.items():
                d[f"{k}_auprc"] = float(average_precision_score(y, v))
            Tpx = np.repeat(tiles[:, 0], TILE * TILE)
            blocks = np.unique(Tpx)
            edges = np.unique(np.quantile(np.concatenate(list(preds.values())), np.linspace(0, 1, 20001)[1:-1]))
            hist = {k: block_hist(v, y, Tpx, blocks, edges) for k, v in preds.items()}
            W = np.vstack([np.ones(len(blocks)),
                           rng.multinomial(len(blocks), np.full(len(blocks), 1 / len(blocks)), size=N_BOOT)])
            ap = {k: ap_from_counts(W @ a, W @ b) for k, (a, b) in hist.items()}
            for u, v in [("pyree", "televit"), ("pyree", "clim_threshold"), ("pyree", "clim_anyburn")]:
                diff = ap[u] - ap[v]
                d[f"{u}-{v}"] = float(diff[0])
                d[f"{u}-{v}_ci"] = [float(q) for q in np.percentile(diff[1:], [2.5, 97.5])]
            res[str(h)] = d
            print(f"[{name} h{h}] base {d['base_rate']:.4f} | " +
                  " ".join(f"{k}={d[k + '_auprc']:.3f}" for k in preds) +
                  f" | P-T {d['pyree-televit']:+.3f} {np.round(d['pyree-televit_ci'], 3)}"
                  f" | P-climthr {d['pyree-clim_threshold']:+.3f} {np.round(d['pyree-clim_threshold_ci'], 3)}",
                  flush=True)
        out[name] = res
    (OUT / "work" / "large_fire_thresholds.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

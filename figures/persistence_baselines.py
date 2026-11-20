"""Persistence baselines (no deep learning) on the benchmark protocol, 2019 test year.

For each target date T and lead h, features are taken at the issue date t0 = T - h:
burned at issue time, burn frequency over the last 4, 12 and 46 periods, burned in the same
period one year before the target, and the recency-weighted climatology. Baselines:
  - single-feature persistence forecasts (each feature used directly as a probability),
  - a logistic regression of all persistence features plus logit(climatology), fitted per lead
    on the 2018 validation year and applied unchanged to 2019.
Pyree is compared with the strongest baseline by a paired block bootstrap over target dates.
"""
import json
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from common import LEADS, OUT, ROOT, RUNS
from mapdata import TILE, load

sys.path.insert(0, str(ROOT))
from pyree.bootstrap import ap_from_counts, block_hist  # noqa: E402

N_BOOT = 1000
FEATS = ["fire_issue", "freq_4", "freq_12", "freq_46", "same_period_last_year", "logit_clim"]


def features(md, split, h):
    z = np.load(RUNS / "v4_noanchor" / f"{split}_preds.npz")
    tiles = md[f"tiles_{split}"]
    assert np.array_equal(z[f"T_h{h}"], tiles[:, 0])
    fire = md["fire"]
    rows = []
    for T, i, j in tiles:
        sl = (slice(i * TILE, (i + 1) * TILE), slice(j * TILE, (j + 1) * TILE))
        t0 = T - h
        f = fire[:, sl[0], sl[1]]
        rows.append(np.stack([f[t0], f[t0 - 3:t0 + 1].mean(0), f[t0 - 11:t0 + 1].mean(0),
                              f[t0 - 45:t0 + 1].mean(0), f[T - 46]]).reshape(5, -1))
    X = np.concatenate(rows, axis=1).T.astype(np.float32)
    y = z[f"y_h{h}"].reshape(-1).astype(np.uint8)
    yy = np.concatenate([fire[T, i * TILE:(i + 1) * TILE, j * TILE:(j + 1) * TILE].ravel() for T, i, j in tiles])
    assert np.array_equal(y, yy), "target mismatch"
    c = np.clip(z[f"c_h{h}"].reshape(-1), 1e-4, 1 - 1e-4)
    X = np.concatenate([X, np.log(c / (1 - c))[:, None]], axis=1)
    p = z[f"p_h{h}"].reshape(-1)
    Tpx = np.repeat(tiles[:, 0], TILE * TILE)
    return X, y, c, p, Tpx


def main():
    md = load()
    rng = np.random.default_rng(0)
    out = {}
    for h in LEADS:
        Xv, yv, _, _, _ = features(md, "val", h)
        sub = rng.choice(len(yv), size=min(3_000_000, len(yv)), replace=False)
        lr = LogisticRegression(max_iter=500, C=1.0).fit(Xv[sub], yv[sub])
        X, y, c, p, Tpx = features(md, "test", h)
        preds = {"persist_issue": X[:, 0], "persist_freq4": X[:, 1], "persist_freq12": X[:, 2],
                 "persist_freq46": X[:, 3], "persist_lastyear": X[:, 4], "clim": c,
                 "persist_clim_logistic": lr.predict_proba(X)[:, 1], "pyree": p}
        d = {"coef": dict(zip(FEATS, lr.coef_[0].round(4).tolist()))}
        bs_ref = np.mean((c - y) ** 2)
        for k, v in preds.items():
            d[f"{k}_auprc"] = float(average_precision_score(y, v))
            d[f"{k}_bss"] = float(1 - np.mean((v - y) ** 2) / bs_ref)
        # paired block bootstrap: Pyree vs persistence+climatology logistic, and logistic vs climatology
        blocks = np.unique(Tpx)
        edges = np.unique(np.quantile(np.concatenate([p, preds["persist_clim_logistic"], c]),
                                      np.linspace(0, 1, 20001)[1:-1]))
        hist = {k: block_hist(preds[k], y, Tpx, blocks, edges) for k in ("pyree", "persist_clim_logistic", "clim")}
        W = np.vstack([np.ones(len(blocks)), rng.multinomial(len(blocks), np.full(len(blocks), 1 / len(blocks)),
                                                              size=N_BOOT)])
        ap = {k: ap_from_counts(W @ a, W @ b) for k, (a, b) in hist.items()}
        for u, v in [("pyree", "persist_clim_logistic"), ("persist_clim_logistic", "clim")]:
            diff = ap[u] - ap[v]
            d[f"{u}-{v}"] = float(diff[0])
            d[f"{u}-{v}_ci"] = [float(q) for q in np.percentile(diff[1:], [2.5, 97.5])]
        out[str(h)] = d
        print(f"h{h}: " + " ".join(f"{k}={d[k + '_auprc']:.3f}" for k in preds) +
              f" | pyree-logit {d['pyree-persist_clim_logistic']:+.3f} {np.round(d['pyree-persist_clim_logistic_ci'], 3)}"
              f" | logit-clim {d['persist_clim_logistic-clim']:+.3f} {np.round(d['persist_clim_logistic-clim_ci'], 3)}",
              flush=True)
    (OUT / "work" / "persistence_baselines.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Train/test Lux on the homogenized shared dataset (stardata).

Consumes a dataset directory built by ``stardata.build_dataset`` so Lux sees
exactly the same stars and split as the Cannon and the bingo BNN. Writes a
results parquet in the established Lux schema (reference labels + ``pred_*`` /
``pred_err_*`` + ``spec_chi2/spec_n_pix/spec_rchi2`` + ``split``) — directly
consumable by ``stardiag.load_lux``.

Usage:
    python src/run_lux_shared.py --dataset-dir <dir> [--P 128] [--l2 1000] \
        [--n-iterations 1000] [--out lux_shared_results.parquet]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent            # Lux/src
REPO = HERE.parent
sys.path.insert(0, str(HERE))

for _c in (REPO.parent / "stardiag", Path.home() / "code" / "stardiag",
           Path.home() / "scr_mk27" / "stardiag"):
    if (_c / "stardata.py").exists():
        sys.path.insert(0, str(_c))
        break
else:
    sys.exit("stardiag checkout (stardata.py) not found next to this repo")
import stardata  # noqa: E402


def predict_with_diagnostics(model, fluxes, fluxes_err, chunk=128):
    """predict_labels + formal errors + spectrum chi2 (chunked).

    Same estimator as src/predict_bulge_rgb_ages.py: W = 1/(err^2 + V) with
    V = exp(2 ln_noise); Cov(labels) = A (B^T W B)^-1 A^T.
    """
    import jax.numpy as jnp

    betas, alphas = model.betas, model.alphas
    ln_noise = model.ln_noise_fluxes
    if ln_noise is None:
        ln_noise = jnp.full(betas.shape[0], -20.)
    V = jnp.exp(2. * ln_noise)

    n_stars, M, P = fluxes.shape[0], alphas.shape[0], model.P
    labels_pred = np.empty((n_stars, M))
    labels_pred_err = np.empty((n_stars, M))
    spec_chi2 = np.empty(n_stars)
    for s in range(0, n_stars, chunk):
        f = jnp.array(fluxes[s:s + chunk])
        e = jnp.array(fluxes_err[s:s + chunk])
        pred, zetas = model.predict_labels(f, e, return_zetas=True)
        labels_pred[s:s + chunk] = np.asarray(pred)

        w = 1. / (e**2 + V[None, :])
        resid = f - zetas @ betas.T
        spec_chi2[s:s + chunk] = np.asarray(jnp.sum(w * resid**2, axis=1))
        G = (betas.T[None, :, :] * w[:, None, :]) @ betas
        eps = 1e-12 * jnp.trace(G, axis1=-2, axis2=-1)[:, None, None] / P
        G = G + eps * jnp.eye(P)
        X = jnp.linalg.solve(G, jnp.broadcast_to(alphas.T,
                                                 (f.shape[0], P, M)))
        var = jnp.einsum('mp,npm->nm', alphas, X)
        labels_pred_err[s:s + chunk] = np.asarray(jnp.sqrt(var))

    spec_n_good = (np.asarray(fluxes_err) < 100).sum(axis=1)
    return labels_pred, labels_pred_err, spec_chi2, spec_n_good


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--merged-path", default=None)
    ap.add_argument("--continuum-list", default=None)
    ap.add_argument("--P", type=int, default=128)
    ap.add_argument("--l2", type=float, default=1000.0)
    ap.add_argument("--n-iterations", type=int, default=1000)
    ap.add_argument("--include-val", action="store_true",
                    help="fold the val stars into the training set")
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42,
                    help="numpy global seed: LuxModel initializes its "
                         "latents from np.random.rand, which is otherwise "
                         "unseeded and irreproducible")
    ap.add_argument("--good-pixel-frac", type=float, default=0.99,
                    help="train-split reliable-pixel column threshold")
    ap.add_argument("--out", default=None,
                    help="output parquet (default: <dataset-dir>/"
                         "lux_shared_results.parquet)")
    ap.add_argument("--model-out", default=None,
                    help="save trained model dill here (optional)")
    args = ap.parse_args()

    from lux_model import LuxModel

    stars, manifest = stardata.load_stars(args.dataset_dir)
    print(f"{len(stars)} rows / {stars['APOGEE_ID'].nunique()} stars "
          f"from {args.dataset_dir}")
    dispersion, nflux, nivar = stardata.load_spectra(
        stars, data_root=args.data_root, merged_path=args.merged_path,
        continuum_list=args.continuum_list)

    labels, labels_err = stardata.lux_label_arrays(stars)
    fluxes, fluxes_err = stardata.lux_flux_arrays(nflux, nivar)

    split = stars["split"].to_numpy()
    train_mask = (split == "train") | (args.include_val & (split == "val"))

    # the SAME wavelength columns the Cannon uses — shared, persisted once
    out_dir = Path(args.dataset_dir)
    keep = stardata.shared_pixel_mask(out_dir, nivar, train_mask,
                                      args.good_pixel_frac)
    print(f"pixel mask: keeping {keep.sum()}/{keep.size} columns")
    fluxes, fluxes_err = fluxes[:, keep], fluxes_err[:, keep]

    np.random.seed(args.seed)
    model = LuxModel(P=args.P)
    model.fit(labels[train_mask], labels_err[train_mask],
              fluxes[train_mask], fluxes_err[train_mask],
              n_iterations=args.n_iterations, l2_reg_strength=args.l2,
              label_names=stardata.LUX_LABELS, verbose=True)
    if args.model_out:
        model.save(args.model_out)

    pred, pred_err, chi2, n_good = predict_with_diagnostics(
        model, fluxes, fluxes_err, chunk=args.chunk)

    out = pd.DataFrame({"APOGEE_ID": stars["APOGEE_ID"]})
    for c in ("source", "is_primary", "evo_state_source", "rgb_proba",
              "EvoState", "snr", "numax"):
        if c in stars.columns:
            out[c] = stars[c].to_numpy()
    for j, name in enumerate(stardata.LUX_LABELS):
        out[name] = labels[:, j]
    for name in stardata.LUX_ERRS:
        if name not in stars.columns:
            sys.exit(f"dataset is missing '{name}' — it predates the shared "
                     f"label vector; rebuild it with --rebuild")
        out[name] = stars[name].to_numpy()
    out["split"] = split
    for j, name in enumerate(stardata.LUX_LABELS):
        out[f"pred_{name}"] = pred[:, j]
        out[f"pred_err_{name}"] = pred_err[:, j]
    out["spec_chi2"] = chi2
    out["spec_n_pix"] = n_good
    out["spec_rchi2"] = chi2 / np.maximum(n_good - args.P, 1)

    out_path = Path(args.out) if args.out else \
        out_dir / "lux_shared_results.parquet"
    out.to_parquet(out_path)
    print(f"wrote {len(out)} rows to {out_path}")


if __name__ == "__main__":
    main()

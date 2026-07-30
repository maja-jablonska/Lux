"""
Apply a trained LuxModel to the bulge RGB sample and write a per-star age
catalogue -- the Lux counterpart of AnniesLasso's scripts/predict_ages.py.

The .dill persists neither the wavelength grid nor the reliable-pixel mask,
so both are re-derived by replaying the training data pipeline
(run_sweep_rgb_all_missions.load_dataset, identical to
notebooks/train-rgb-wilett-all-missions.ipynb) and cached in a sidecar .npz
next to the model; later runs load the sidecar and never touch the training
parquet again.

The input spectra parquet (e.g. bulge_rgb_spectra.parquet) must carry per-row
``wavelength``/``flux``/``ivar`` arrays on the full 8575-pixel APOGEE grid.
They are pseudo-continuum-normalized with thecannon.continuum exactly as in
training, masked to the training pixels, and pushed through
``predict_labels`` with the exact formal covariance
A (B^T W B)^{-1} A^T of the linear estimator (the training notebook's
diagnostics helper), chunked over stars so the device memory stays bounded.

Output columns: star identifiers, ``pred_<label>`` / ``pred_err_<label>`` for
every model label, ``spec_chi2`` / ``spec_n_pix`` / ``spec_rchi2``, and the
linearised ``age_gyr`` / ``age_gyr_err`` + ``flag_unphysical_age`` from the
``log_age_L`` label.

NOTE: formal errors only. The systematics notebook measured a residual
slope of -0.39 (shrinkage toward the sample mean) and ~0.13 dex systematic
scatter beyond the formal budget -- budget for that downstream.

Usage (Gadi):
    python predict_bulge_rgb_ages.py \
        --model /scratch/mk27/mj8805/Lux/lux-model-rgb-wilett-all-missions.dill \
        --spectra ~/scr_mk27/bulge-ages-and-orbits/data/bulge_rgb_spectra.parquet \
        --output ~/scr_mk27/bulge-ages-and-orbits/data/lux_ages_rgb.parquet
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# thecannon is used only for its continuum module (same candidates as the
# sweep, but resolved relative to this file so the script works from any cwd)
_REPO_PARENT = Path(__file__).resolve().parent.parent.parent
for cand in [str(_REPO_PARENT / 'AnniesLasso'), '../../AnniesLasso',
             '/scratch/mk27/mj8805/AnniesLasso']:
    if os.path.exists(os.path.join(cand, 'thecannon', 'continuum.py')):
        sys.path.append(os.path.abspath(cand))
        break
else:
    raise FileNotFoundError('could not locate AnniesLasso (needed for '
                            'thecannon.continuum)')

import jax.numpy as jnp

from lux_model import LuxModel
from run_sweep_rgb_all_missions import (APOGEE_REGIONS, DATA_ROOT_CANDIDATES,
                                        load_dataset, to_array)
from thecannon import continuum

ID_CANDIDATES = ('sdss_id', 'obj', 'sdss4_apogee_id', 'spectrum_pk_y', 'snr')


def resolve_data_root(arg):
    if arg is not None:
        return Path(arg)
    for cand in DATA_ROOT_CANDIDATES:
        if os.path.exists(os.path.join(cand, 'bulge-ages-and-orbits', 'data')):
            return Path(cand)
    raise FileNotFoundError('could not locate the bulge-ages-and-orbits data '
                            'root; pass --data-root')


def pixel_mask(model, mask_path, data_root, seed, test_size):
    """The training reliable-pixel mask over the full grid: from the sidecar
    if present, else replayed from the training set and cached there."""
    if mask_path.exists():
        with np.load(mask_path) as f:
            keep = f['keep'].astype(bool)
        print(f'pixel mask from {mask_path}: {int(keep.sum())}/{keep.size} '
              'pixels')
    else:
        print('no sidecar pixel mask; replaying the training pipeline '
              f'(seed={seed}, test_size={test_size}) to derive it...')
        *_, keep = load_dataset(data_root, seed, test_size,
                                return_pixel_mask=True)
        np.savez_compressed(mask_path, keep=keep, seed=seed,
                            test_size=test_size)
        print(f'cached pixel mask -> {mask_path}')
    if int(keep.sum()) != model.n_wavelengths:
        raise ValueError(
            f'pixel mask keeps {int(keep.sum())} pixels but the model was '
            f'trained with Lambda={model.n_wavelengths}; the mask does not '
            'belong to this model (delete the sidecar and check '
            '--seed/--test-size)')
    return keep


def load_bulge_spectra(path):
    """Ids + stacked flux/ivar from a per-row-array spectra parquet."""
    available = pq.ParquetFile(path).schema_arrow.names
    id_cols = [c for c in ID_CANDIDATES if c in available]
    need = ['wavelength', 'flux', 'ivar']
    missing = [c for c in need if c not in available]
    if missing:
        raise ValueError(f'{path} lacks spectra columns {missing}')
    table = pd.read_parquet(path, columns=id_cols + need)

    dispersion = to_array(table['wavelength'].iloc[0])
    flux_arrays, ivar_arrays, bad_rows = [], [], []
    for i, (fx, iv) in enumerate(zip(table['flux'], table['ivar'])):
        f_arr, iv_arr = to_array(fx), to_array(iv)
        if f_arr.size != dispersion.size or iv_arr.size != dispersion.size:
            bad_rows.append(i)
            f_arr = np.full(dispersion.size, np.nan)
            iv_arr = np.zeros(dispersion.size)
        flux_arrays.append(f_arr)
        ivar_arrays.append(iv_arr)
    if bad_rows:
        keep_rows = np.ones(len(table), dtype=bool)
        keep_rows[bad_rows] = False
        table = table.loc[keep_rows].reset_index(drop=True)
        flux_arrays = [a for i, a in enumerate(flux_arrays)
                       if keep_rows[i]]
        ivar_arrays = [a for i, a in enumerate(ivar_arrays)
                       if keep_rows[i]]
        print(f'dropped {len(bad_rows)} rows on a different pixel grid')
    return (table[id_cols], dispersion, np.vstack(flux_arrays),
            np.vstack(ivar_arrays))


def normalize_chunked(dispersion, flux, ivar, continuum_pixels, chunk):
    """thecannon pseudo-continuum normalization, training parameters, chunked
    over stars (the continuum fit is per-star, so chunking is exact)."""
    disp_j = jnp.array(dispersion)
    pix_j = jnp.array(continuum_pixels)
    regions = [jnp.array(r) for r in APOGEE_REGIONS]
    out_flux = np.empty_like(flux)
    out_ivar = np.empty_like(ivar)
    for s in range(0, flux.shape[0], chunk):
        nf, ni, _, _ = continuum.normalize(
            disp_j, jnp.array(flux[s:s + chunk]), jnp.array(ivar[s:s + chunk]),
            pix_j, L=1400, order=3, regions=regions)
        out_flux[s:s + chunk] = np.asarray(nf)
        out_ivar[s:s + chunk] = np.asarray(ni)
    return out_flux, out_ivar


def predict_with_diagnostics(model, fluxes, fluxes_err, chunk):
    """predict_labels + exact formal errors and spectrum-fit chi2, chunked
    over stars (the training notebook's diagnostics helper, with the zeta
    solve chunked too so the whole sample never sits on the device).

    The test-time zetas solve (B^T W_n B) zeta_n = B^T W_n f_n with
    W_n = diag(1 / (err_n^2 + V)), V = exp(2 ln_noise). The estimator is
    linear in the fluxes, so Cov(labels) = A (B^T W_n B)^{-1} A^T exactly.
    """
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

        w = 1. / (e**2 + V[None, :])                             # n x Lambda
        resid = f - zetas @ betas.T
        spec_chi2[s:s + chunk] = np.asarray(jnp.sum(w * resid**2, axis=1))
        G = (betas.T[None, :, :] * w[:, None, :]) @ betas        # n x P x P
        # same negligible ridge as scatters._anchored_solve, for stars whose
        # masked pixels leave a latent direction unconstrained
        eps = 1e-12 * jnp.trace(G, axis1=-2, axis2=-1)[:, None, None] / P
        G = G + eps * jnp.eye(P)
        X = jnp.linalg.solve(G, jnp.broadcast_to(alphas.T,
                                                 (f.shape[0], P, M)))
        var = jnp.einsum('mp,npm->nm', alphas, X)         # diag(A G^-1 A^T)
        labels_pred_err[s:s + chunk] = np.asarray(jnp.sqrt(var))
        if (s // chunk) % 10 == 0:
            print(f'  {min(s + chunk, n_stars)}/{n_stars} stars')

    spec_n_good = (np.asarray(fluxes_err) < 100).sum(axis=1)
    return labels_pred, labels_pred_err, spec_chi2, spec_n_good


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    parser.add_argument('--model', required=True,
                        help='trained LuxModel .dill (e.g. '
                             'lux-model-rgb-wilett-all-missions.dill)')
    parser.add_argument('--spectra', required=True,
                        help='parquet of per-row wavelength/flux/ivar arrays '
                             '(already pure RGB, e.g. '
                             'bulge_rgb_spectra.parquet)')
    parser.add_argument('--output', required=True,
                        help='output age catalogue parquet')
    parser.add_argument('--data-root', default=None,
                        help='directory containing bulge-ages-and-orbits/data '
                             '(continuum.list + the training parquet if the '
                             'pixel-mask sidecar is missing)')
    parser.add_argument('--pixel-mask', default=None,
                        help='sidecar .npz with the training reliable-pixel '
                             'mask (default: <model>.pixel-mask.npz; derived '
                             'from the training set and cached if absent)')
    parser.add_argument('--seed', type=int, default=42,
                        help='training split seed, only for deriving the '
                             'pixel mask (42 = the training notebook)')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='training split test fraction, only for '
                             'deriving the pixel mask')
    parser.add_argument('--chunk', type=int, default=2048,
                        help='stars per device chunk (default 2048)')
    args = parser.parse_args()

    model = LuxModel.load(args.model)
    if not model.is_trained:
        raise ValueError(f'model at {args.model} is not trained')
    label_names = list(model.label_names)
    print(f'loaded {args.model}: P={model.P}, '
          f'Lambda={model.n_wavelengths}, labels={label_names}')
    if 'log_age_L' not in label_names:
        raise ValueError(f'model has no log_age_L label (labels: '
                         f'{label_names}); cannot estimate ages')
    age_index = label_names.index('log_age_L')

    data_root = resolve_data_root(args.data_root)
    mask_path = (Path(args.pixel_mask) if args.pixel_mask else
                 Path(args.model).with_suffix('.pixel-mask.npz'))
    keep = pixel_mask(model, mask_path, data_root, args.seed, args.test_size)

    ids, dispersion, flux, ivar = load_bulge_spectra(args.spectra)
    print(f'{len(ids)} stars x {dispersion.size} pixels from {args.spectra}')
    if dispersion.size != keep.size:
        raise ValueError(f'sample has {dispersion.size} pixels but the '
                         f'training grid had {keep.size}')

    continuum_pixels = np.loadtxt(
        data_root / 'bulge-ages-and-orbits/data/continuum.list',
        dtype=int, comments='#')
    continuum_pixels = continuum_pixels[continuum_pixels < dispersion.size]
    normalized_flux, normalized_ivar = normalize_chunked(
        dispersion, flux, ivar, continuum_pixels, args.chunk)

    # Lux inputs: bad pixels get flux=1, err=9999 (down-weighted to ~0)
    bad_pix = ~((normalized_ivar > 0) & np.isfinite(normalized_flux))
    safe_ivar = np.where(bad_pix, 1., normalized_ivar)
    fluxes = np.where(bad_pix, 1., normalized_flux)[:, keep]
    fluxes_err = np.where(bad_pix, 9999., 1. / np.sqrt(safe_ivar))[:, keep]

    labels_pred, labels_pred_err, spec_chi2, spec_n_good = \
        predict_with_diagnostics(model, fluxes, fluxes_err, args.chunk)

    catalogue = ids.copy()
    for m, name in enumerate(label_names):
        catalogue[f'pred_{name}'] = labels_pred[:, m]
        catalogue[f'pred_err_{name}'] = labels_pred_err[:, m]
    catalogue['spec_chi2'] = spec_chi2
    catalogue['spec_n_pix'] = spec_n_good
    # the P latent dims are spent as fitted parameters
    catalogue['spec_rchi2'] = spec_chi2 / (spec_n_good - model.P)

    # Linear age with the log-normal error propagated: for x = log10(age),
    # sigma_age = age * ln(10) * sigma_x.
    log_age = labels_pred[:, age_index]
    log_age_err = labels_pred_err[:, age_index]
    age_gyr = 10.0 ** log_age
    catalogue['age_gyr'] = age_gyr
    catalogue['age_gyr_err'] = age_gyr * np.log(10.0) * log_age_err
    catalogue['flag_unphysical_age'] = ~((age_gyr > 0) & (age_gyr < 20))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    catalogue.to_parquet(out, index=False)

    physical = ~catalogue['flag_unphysical_age']
    print(f'\nwrote {len(catalogue)}-star Lux age catalogue to {out}')
    print(f'  median age {np.median(age_gyr[physical]):.2f} Gyr | '
          f'16th-84th {np.percentile(age_gyr[physical], 16):.2f}-'
          f'{np.percentile(age_gyr[physical], 84):.2f} Gyr | '
          f'{int((~physical).sum())} unphysical | '
          f"median spec_rchi2 {np.median(catalogue['spec_rchi2']):.2f}")


if __name__ == '__main__':
    main()

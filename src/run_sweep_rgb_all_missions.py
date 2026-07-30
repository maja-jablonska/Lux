"""
GPU P/l2 sweep of the LuxModel on the Willett all-missions RGB sample,
logged to Weights & Biases.

Replicates the data pipeline of ``notebooks/train-rgb-wilett-all-missions.ipynb``
exactly (same quality cuts, thecannon continuum normalization, persisted RGB
selection, label error floors, reliable-pixel mask, and the same seeded 80/20
split), then trains one model per (P, l2_reg_strength) grid point and logs the
training chi2 history plus held-out metrics to one offline wandb run each.

The metric that motivated this sweep is the age residual-vs-truth slope: the
systematics notebook measured -0.39 (ages shrink toward the sample mean) with
~0.13 dex of systematic error beyond the formal + reference error budget, so
each config records per-label slope, robust bias/scatter, and the test/train
overfit ratio alongside the fit diagnostics.

Finished grid points are checkpointed to a results parquet after every fit and
skipped on re-run, so a walltime kill costs only the fit in progress -- just
resubmit. wandb runs default to offline (no network on Gadi compute nodes);
sync with ``wandb sync wandb/offline-run-*`` or let jobs/sync-wandb.pbs do it.

Example (defaults reproduce the grid around the production P=256, l2=1000):

        python run_sweep_rgb_all_missions.py --P 128 256 384 \
                --l2 100 300 1000 3000 --n-iterations 1000
"""

import argparse
import gc
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import wandb

# thecannon is used only for its continuum module, so the normalization is
# identical to the AnniesLasso notebook that defines this dataset
for cand in ['../../AnniesLasso', '/scratch/mk27/mj8805/AnniesLasso']:
    if os.path.exists(os.path.join(cand, 'thecannon', 'continuum.py')):
        sys.path.append(os.path.abspath(cand))
        break
else:
    raise FileNotFoundError('could not locate AnniesLasso (needed for '
                            'thecannon.continuum)')

import jax
import jax.numpy as jnp

from lux_model import LuxModel
from thecannon import continuum

# data root (Gadi first, then a local checkout)
DATA_ROOT_CANDIDATES = ['/home/100/mj8805/scr_mk27',
                        os.path.expanduser('~/scr_mk27')]

LABEL_COLS = ['raw_teff', 'raw_logg', 'raw_fe_h', 'mg_fe', 'c_n', 'log_age_L']
ERR_COLS = ['raw_e_teff', 'raw_e_logg', 'raw_e_fe_h', 'e_mg_fe', 'c_n_err',
            'e_log_age_L']

# ASPCAP formal errors are internal precisions far below the true accuracy;
# floor them before they become 1/err^2 training weights (the c_n floor is
# the quadrature sum of the C and N floors)
ERR_FLOOR = {
    'raw_teff': 30., 'raw_logg': 0.05, 'raw_fe_h': 0.03,
    'mg_fe': 0.03, 'c_n': 0.07, 'log_age_L': 0.05,
}

APOGEE_REGIONS = ([15090, 15822], [15823, 16451], [16452, 16971])


def to_array(x):
    if isinstance(x, str):                       # stringified list
        return np.array(x.strip('[] \n').split(','), dtype=float)
    arr = np.asarray(x, dtype=float)
    return arr[None] if arr.ndim == 0 else arr


def load_dataset(src, seed, test_size):
    """The training notebook's data pipeline, condensed; see
    notebooks/train-rgb-wilett-all-missions.ipynb for the rationale of every
    cut. Returns the Lux arrays and the notebook's exact train/test split."""
    spectra = pd.read_parquet(
        src / 'bulge-ages-and-orbits/data/merged_with_ages_raw.parquet')
    spectra = spectra[np.isfinite(spectra['age_L'])].reset_index(drop=True)
    spectra['log_age_L'] = np.log10(spectra['age_L'])

    if 'mg_fe' not in spectra.columns:
        spectra['mg_fe'] = spectra['raw_mg_h'] - spectra['raw_fe_h']
    spectra['e_mg_fe'] = np.sqrt(spectra['raw_e_mg_h']**2
                                 + spectra['raw_e_fe_h']**2)
    spectra['c_n'] = spectra['raw_c_h'] - spectra['raw_n_h']
    spectra['c_n_err'] = np.sqrt(spectra['raw_e_c_h']**2
                                 + spectra['raw_e_n_h']**2)
    spectra['e_log_age_L'] = 0.5 * (np.log10(spectra['age_84_L'])
                                    - np.log10(spectra['age_16_L']))

    # quality cuts: ASPCAP sentinel rows, unreliable Willett ages, and young
    # alpha-rich impostors (mass-transfer products with wrong reference ages)
    sentinel_cols = ['raw_teff', 'raw_logg', 'raw_fe_h', 'raw_mg_h',
                     'raw_c_h', 'raw_n_h', 'raw_al_h']
    sentinel = (spectra[sentinel_cols] < -100).any(axis=1)
    unreliable_age = ~spectra['RelAge_L'].astype(bool)
    impostor = ((spectra['age_L'] < 4) & (spectra['mg_fe'] > 0.15)
                & (spectra['c_n'] > -0.2))
    spectra = spectra[~(sentinel | unreliable_age
                        | impostor)].reset_index(drop=True)

    # stack the per-row wavelength/flux/ivar arrays onto one grid
    dispersion = to_array(spectra['wavelength'].iloc[0])
    flux_arrays, ivar_arrays, bad_rows = [], [], set()
    for i, (fx, iv) in enumerate(zip(spectra['flux'], spectra['ivar'])):
        f_arr, iv_arr = to_array(fx), to_array(iv)
        if f_arr.size != dispersion.size or iv_arr.size != dispersion.size:
            bad_rows.add(i)
        flux_arrays.append(f_arr)
        ivar_arrays.append(iv_arr)
    if bad_rows:
        keep_rows = [i for i in range(len(spectra)) if i not in bad_rows]
        spectra = spectra.iloc[keep_rows].reset_index(drop=True)
        flux_arrays = [a for i, a in enumerate(flux_arrays)
                       if i not in bad_rows]
        ivar_arrays = [a for i, a in enumerate(ivar_arrays)
                       if i not in bad_rows]
    flux = np.vstack(flux_arrays)
    ivar = np.vstack(ivar_arrays)

    # thecannon continuum normalization, identical pixel list and parameters
    continuum_pixels = np.loadtxt(
        src / 'bulge-ages-and-orbits/data/continuum.list',
        dtype=int, comments='#')
    continuum_pixels = continuum_pixels[continuum_pixels < dispersion.size]
    normalized_flux, normalized_ivar, _, _ = continuum.normalize(
        jnp.array(dispersion), jnp.array(flux), jnp.array(ivar),
        jnp.array(continuum_pixels), L=1400, order=3,
        regions=[jnp.array(r) for r in APOGEE_REGIONS])
    normalized_flux = np.asarray(normalized_flux)
    normalized_ivar = np.asarray(normalized_ivar)

    # RGB selection persisted by the AnniesLasso notebook
    sel = pd.read_parquet(
        src / 'bulge-ages-and-orbits/data/rgb_selection_all_missions.parquet')
    sel = sel[['APOGEE_ID', 'source', 'evo_state_source',
               'rgb_proba']].drop_duplicates(subset=['APOGEE_ID', 'source'])
    spectra = spectra.merge(sel, on=['APOGEE_ID', 'source'], how='left',
                            validate='many_to_one')
    is_rgb = spectra['evo_state_source'].notna().to_numpy()
    spectra = spectra[is_rgb].reset_index(drop=True)
    normalized_flux = normalized_flux[is_rgb]
    normalized_ivar = normalized_ivar[is_rgb]

    finite = np.isfinite(spectra[LABEL_COLS].values).all(axis=1)
    spectra = spectra[finite].reset_index(drop=True)
    normalized_flux = normalized_flux[finite]
    normalized_ivar = normalized_ivar[finite]

    # Lux inputs: bad pixels get flux=1, err=9999; label errors floored,
    # missing-label sentinels preserved
    bad_pix = ~(normalized_ivar > 0)
    safe_ivar = np.where(bad_pix, 1., normalized_ivar)
    fluxes = jnp.array(np.where(bad_pix, 1., normalized_flux))
    fluxes_err = jnp.array(np.where(bad_pix, 9999., 1. / np.sqrt(safe_ivar)))

    labels = jnp.array(spectra[LABEL_COLS].values)
    labels_err_np = np.array(spectra[ERR_COLS].values, dtype=float)
    missing = ~(labels_err_np > 0)
    floor_vec = np.array([ERR_FLOOR[n] for n in LABEL_COLS])
    labels_err_np = np.maximum(labels_err_np, floor_vec)
    labels_err_np[missing] = 9999.
    labels_err = jnp.array(labels_err_np)

    # the notebook's exact split (sklearn, random_state=seed), then its
    # near-100% reliable-pixel cut computed on the train split
    from sklearn.model_selection import train_test_split
    train_mask, test_mask = train_test_split(
        np.arange(labels.shape[0]), test_size=test_size, random_state=seed)
    good_frac = (np.asarray(fluxes_err[train_mask]) < 100).mean(axis=0)
    keep = good_frac > 0.99
    fluxes = fluxes[:, keep]
    fluxes_err = fluxes_err[:, keep]

    print(f'{fluxes.shape[0]} RGB stars ({len(train_mask)} train / '
          f'{len(test_mask)} test), {int(keep.sum())}/{keep.size} pixels kept')
    return labels, labels_err, fluxes, fluxes_err, train_mask, test_mask


def robust_bias_scatter(d):
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return np.nan, np.nan
    b = np.median(d)
    return b, 1.4826 * np.median(np.abs(d - b))


def evaluate(model, labels, labels_err, fluxes, fluxes_err, train_mask,
             test_mask):
    """Held-out metrics per config. slope is the residual-vs-truth linear fit
    (0 = unbiased, -1 = pure shrinkage to the sample mean); overfit is the
    test/train robust-scatter ratio (~1 = no memorisation)."""
    truth = np.asarray(labels[test_mask])
    truth_tr = np.asarray(labels[train_mask])
    pred = np.asarray(model.predict_labels(fluxes[test_mask],
                                           fluxes_err[test_mask]))
    pred_tr = np.asarray(model.predict_labels(fluxes[train_mask],
                                              fluxes_err[train_mask]))

    row = {}
    for m, name in enumerate(LABEL_COLS):
        r = pred[:, m] - truth[:, m]
        b, s = robust_bias_scatter(r)
        row[f'{name}_bias'], row[f'{name}_scatter'] = b, s
        ok = np.isfinite(truth[:, m]) & np.isfinite(r)
        row[f'{name}_slope'] = float(np.polyfit(truth[ok, m], r[ok], 1)[0])
        _, s_tr = robust_bias_scatter(pred_tr[:, m] - truth_tr[:, m])
        row[f'{name}_overfit'] = s / s_tr if s_tr > 0 else np.nan
    age = LABEL_COLS.index('log_age_L')
    b_gyr, s_gyr = robust_bias_scatter(10**pred[:, age] - 10**truth[:, age])
    row['age_gyr_bias'], row['age_gyr_scatter'] = b_gyr, s_gyr
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    parser.add_argument('--data-root', default=None,
                        help='directory containing bulge-ages-and-orbits/data '
                             '(default: search the notebook candidate paths)')
    parser.add_argument('--P', type=int, nargs='+', default=[128, 256, 384],
                        help='latent dimensionalities to sweep')
    parser.add_argument('--l2', type=float, nargs='+',
                        default=[100., 300., 1000., 3000.],
                        help='L2 regularisation strengths to sweep')
    parser.add_argument('--n-iterations', type=int, default=1000,
                        help='coordinate-descent iterations per fit')
    parser.add_argument('--test-size', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42,
                        help='split seed (42 = the training notebook split)')
    parser.add_argument('--results',
                        default='lux-sweep-p-l2-rgb-wilett-all-missions.parquet',
                        help='results checkpoint; finished (P, l2) points in '
                             'it are skipped on re-run')
    parser.add_argument('--outdir', default=None,
                        help='save each trained model as a .dill here '
                             '(default: models are not saved)')
    parser.add_argument('--project', default='lux-sweep-rgb-all-missions')
    parser.add_argument('--wandb-mode', default='offline',
                        choices=['offline', 'online', 'disabled'])
    args = parser.parse_args()

    devices = jax.devices()
    print(f'JAX devices: {devices}')
    if all(d.platform == 'cpu' for d in devices):
        print('WARNING: no GPU visible to JAX; the sweep will run on CPU',
              file=sys.stderr)

    data_root = args.data_root
    if data_root is None:
        for cand in DATA_ROOT_CANDIDATES:
            if os.path.exists(os.path.join(cand, 'bulge-ages-and-orbits',
                                           'data')):
                data_root = cand
                break
        else:
            raise FileNotFoundError('could not locate the '
                                    'bulge-ages-and-orbits data root; '
                                    'pass --data-root')
    print('data root:', data_root)

    labels, labels_err, fluxes, fluxes_err, train_mask, test_mask = \
        load_dataset(Path(data_root), args.seed, args.test_size)

    done = set()
    if os.path.exists(args.results):
        prev = pd.read_parquet(args.results)
        done = {(int(p), float(l)) for p, l in zip(prev['P'], prev['l2'])}
        print(f'resuming: {len(done)} finished configs in {args.results}')
    grid = [(int(P), float(l2)) for P in args.P for l2 in args.l2
            if (int(P), float(l2)) not in done]
    print(f'{len(grid)} configs to run: {grid}')

    if args.outdir:
        os.makedirs(args.outdir, exist_ok=True)

    for P, l2 in grid:
        run = wandb.init(
            project=args.project,
            mode=args.wandb_mode,
            name=f'P{P}-l2{l2:g}',
            group=f'rgb-all-missions-seed{args.seed}',
            config={
                'P': P,
                'l2_reg_strength': l2,
                'n_iterations': args.n_iterations,
                'test_size': args.test_size,
                'seed': args.seed,
                'n_stars': int(labels.shape[0]),
                'n_train': int(len(train_mask)),
                'n_pixels': int(fluxes.shape[1]),
                'label_names': LABEL_COLS,
                'data_root': os.path.abspath(data_root),
                'device': devices[0].platform,
            },
            reinit=True,
        )

        t0 = time.time()
        model = LuxModel(P=P)
        model.fit(labels[train_mask], labels_err[train_mask],
                  fluxes[train_mask], fluxes_err[train_mask],
                  n_iterations=args.n_iterations, l2_reg_strength=l2,
                  label_names=LABEL_COLS, verbose=False)

        # offline mode has no live dashboard, so replaying the per-iteration
        # chi2 history after the fit loses nothing
        for step, chi2 in enumerate(model.chi2_history):
            run.log({'train/chi2': chi2}, step=step)

        row = {'P': P, 'l2': l2, 'n_iterations': args.n_iterations,
               'nll': float(model.nll),
               'final_chi2': float(model.chi2_history[-1]),
               'minutes': (time.time() - t0) / 60.}
        row.update(evaluate(model, labels, labels_err, fluxes, fluxes_err,
                            train_mask, test_mask))
        run.log({f'test/{k}': v for k, v in row.items()
                 if k not in ('P', 'l2', 'n_iterations')})

        if args.outdir:
            model_path = os.path.join(args.outdir, f'lux-rgb-P{P}-l2{l2:g}.dill')
            model.save(model_path)
            run.summary['model_path'] = os.path.abspath(model_path)
        run.finish()

        # append + rewrite = one checkpoint per config
        frames = ([pd.read_parquet(args.results)]
                  if os.path.exists(args.results) else [])
        pd.concat(frames + [pd.DataFrame([row])],
                  ignore_index=True).to_parquet(args.results, index=False)
        print(f"P={P} l2={l2:g}: age slope {row['log_age_L_slope']:+.3f}, "
              f"age scatter {row['log_age_L_scatter']:.3f} dex, "
              f"overfit {row['log_age_L_overfit']:.2f}, "
              f"{row['minutes']:.1f} min")

        # free GPU buffers and compiled kernels before the next (differently
        # shaped) fit; without this the sweep accumulates memory across configs
        del model
        gc.collect()
        jax.clear_caches()

    sweep = pd.read_parquet(args.results).sort_values(['P', 'l2'])
    cols = ['P', 'l2', 'log_age_L_slope', 'log_age_L_scatter',
            'age_gyr_scatter', 'log_age_L_overfit', 'c_n_slope', 'nll',
            'minutes']
    with pd.option_context('display.float_format', '{:.4g}'.format,
                           'display.width', 200):
        print(sweep[cols].to_string(index=False))
    best_scatter = sweep.loc[sweep['log_age_L_scatter'].idxmin()]
    best_slope = sweep.loc[sweep['log_age_L_slope'].abs().idxmin()]
    print(f"lowest age scatter:  P={best_scatter['P']:g}, "
          f"l2={best_scatter['l2']:g} "
          f"({best_scatter['log_age_L_scatter']:.3f} dex, "
          f"slope {best_scatter['log_age_L_slope']:+.3f})")
    print(f"slope closest to 0:  P={best_slope['P']:g}, "
          f"l2={best_slope['l2']:g} ({best_slope['log_age_L_slope']:+.3f}, "
          f"scatter {best_slope['log_age_L_scatter']:.3f} dex)")


if __name__ == '__main__':
    main()

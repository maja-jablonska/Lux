"""
GPU hyperparameter sweep over the LuxModel, logged to Weights & Biases.

Replicates the data pipeline of ``notebooks/example-lux-api.ipynb`` (APOGEE
giant spectra with asteroseismic ages: clean, continuum-normalise, train/test
split), then trains one model per (P, l2_reg_strength) grid point and logs the
training chi2 history, final negative log-likelihood, and held-out label
metrics to wandb.

wandb runs default to offline mode (no network needed on compute nodes); a
proper ``wandb.agent`` sweep requires the cloud controller, so the grid is
enumerated locally instead — one offline run per config. Sync afterwards with:

        wandb sync wandb/offline-run-*

Example (one V100, full notebook-sized fit per grid point):

        python run_sweep.py --data /path/to/merged_with_ages.parquet \
                --P 32 64 128 256 --l2 1000 5000 10000 --n-iterations 2000
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import wandb

from lux_model import LuxModel

# same candidate locations as the example notebook (local checkout and Gadi)
DATA_CANDIDATES = [
    '../../bulge-ages-and-orbits/data/merged_with_ages.parquet',
    '/home/100/mj8805/scr_mk27/bulge-ages-and-orbits/data/merged_with_ages.parquet',
]

LABEL_NAMES = ['raw_teff', 'raw_logg', 'raw_fe_h', 'mg_fe', 'c_fe', 'o_fe',
               'n_fe', 'log_age_Dnu']
LABEL_ERRS = ['raw_e_teff', 'raw_e_logg', 'raw_e_fe_h', 'e_mg_fe', 'e_c_fe',
              'e_o_fe', 'e_n_fe', 'e_log_age_Dnu']


def load_and_clean(path):
    """Load the merged APOGEE parquet and apply the notebook's cleaning cuts."""
    spectra = pd.read_parquet(path)

    spectra = spectra[spectra['spectrum_flags'] == 0]
    warn_cols = [col for col in spectra.columns if col.endswith('_warn')]
    if warn_cols:
        spectra = spectra[~spectra[warn_cols].any(axis=1)].reset_index(drop=True)

    # sentinel / non-physical label values; [Fe/H] is legitimately negative
    for lbl in ['raw_teff', 'raw_logg', 'age_Dnu']:
        spectra = spectra[spectra[lbl] > 0].reset_index(drop=True)
    spectra = spectra[spectra['raw_fe_h'] > -10].reset_index(drop=True)
    spectra = spectra[spectra['snr'] > 100]

    for el in ['mg', 'c', 'o', 'n']:
        spectra[f'{el}_fe'] = spectra[f'raw_{el}_h'] - spectra['raw_fe_h']
        spectra[f'e_{el}_fe'] = np.sqrt(spectra[f'raw_e_{el}_h']**2
                                        + spectra['raw_e_fe_h']**2)

    # asteroseismic age errors are roughly fractional, so fit log10(age)
    spectra['e_age_Dnu'] = np.maximum(
        spectra['age_84_Dnu'] - spectra['age_Dnu'],
        spectra['age_Dnu'] - spectra['age_16_Dnu'])
    spectra['log_age_Dnu'] = np.log10(spectra['age_Dnu'])
    spectra['e_log_age_Dnu'] = (spectra['e_age_Dnu']
                                / (spectra['age_Dnu'] * np.log(10)))

    # 3-sigma clip on each label
    clip_labels = ['raw_teff', 'raw_logg', 'raw_fe_h', 'age_Dnu',
                   'mg_fe', 'c_fe', 'o_fe', 'n_fe']
    for lbl in clip_labels:
        values = spectra[lbl]
        if np.all(np.isnan(values)) or len(values.dropna()) <= 1:
            continue
        med, std = np.nanmedian(values), np.nanstd(values)
        if std == 0 or np.isnan(std):
            continue
        spectra = spectra[np.abs(values - med) < 3 * std].reset_index(drop=True)

    return spectra


def build_arrays(spectra):
    """Continuum-normalise the fluxes and assemble the label/flux matrices."""
    raw_flux = np.array(spectra['flux'].tolist())
    raw_ivar = np.array(spectra['ivar'].tolist())
    cont = np.array(spectra['continuum'].tolist())

    bad = (raw_flux <= 0) | (cont <= 0) | (raw_ivar <= 0)
    safe_cont = np.where(cont > 0, cont, 1.)
    safe_ivar = np.where(raw_ivar > 0, raw_ivar, 1.)

    # same sentinel convention as load_data.load_spectra: flux=1, err=9999
    fluxes = jnp.array(np.where(bad, 1., raw_flux / safe_cont))
    fluxes_err = jnp.array(np.where(bad, 9999.,
                                    1. / (np.sqrt(safe_ivar) * safe_cont)))

    labels = jnp.array(spectra[LABEL_NAMES].values)
    labels_err_np = np.array(spectra[LABEL_ERRS].values, dtype=float)
    labels_err = jnp.array(np.where(labels_err_np > 0, labels_err_np, 9999.))

    return labels, labels_err, fluxes, fluxes_err, float(bad.mean())


def evaluate(model, labels, labels_err, fluxes, fluxes_err, test_mask):
    """Held-out metrics: per-label bias/scatter, age scatter, flux RMS."""
    labels_pred = np.asarray(model.predict_labels(fluxes[test_mask],
                                                  fluxes_err[test_mask]))
    truth = np.asarray(labels[test_mask])

    metrics = {}
    for m, name in enumerate(LABEL_NAMES):
        resid = labels_pred[:, m] - truth[:, m]
        metrics[f'test/{name}_bias'] = float(np.mean(resid))
        metrics[f'test/{name}_scatter'] = float(np.std(resid))

    age_idx = LABEL_NAMES.index('log_age_Dnu')
    age_true, age_pred = 10**truth[:, age_idx], 10**labels_pred[:, age_idx]
    metrics['test/age_frac_scatter'] = float(np.std(age_pred / age_true - 1))

    fluxes_pred = np.asarray(model.predict_fluxes(labels[test_mask],
                                                  labels_err[test_mask]))
    good = np.asarray(fluxes_err[test_mask]) < 100
    flux_resid = (fluxes_pred - np.asarray(fluxes[test_mask]))[good]
    metrics['test/flux_rms'] = float(np.sqrt(np.mean(flux_resid**2)))
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    parser.add_argument('--data', default=None,
                        help='path to merged_with_ages.parquet '
                             '(default: search the notebook candidate paths)')
    parser.add_argument('--P', type=int, nargs='+', default=[32, 64, 128, 256],
                        help='latent dimensionalities to sweep')
    parser.add_argument('--l2', type=float, nargs='+',
                        default=[1000., 5000., 10000.],
                        help='L2 regularisation strengths to sweep')
    parser.add_argument('--n-iterations', type=int, default=2000,
                        help='coordinate-descent iterations per fit')
    parser.add_argument('--test-size', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--outdir', default='sweep-models',
                        help='where to save the trained .dill models')
    parser.add_argument('--project', default='lux-sweep')
    parser.add_argument('--wandb-mode', default='offline',
                        choices=['offline', 'online', 'disabled'])
    args = parser.parse_args()

    devices = jax.devices()
    print(f'JAX devices: {devices}')
    if all(d.platform == 'cpu' for d in devices):
        print('WARNING: no GPU visible to JAX; the sweep will run on CPU',
              file=sys.stderr)

    data_path = args.data
    if data_path is None:
        for cand in DATA_CANDIDATES:
            if os.path.exists(cand):
                data_path = cand
                break
        else:
            raise FileNotFoundError(
                'could not locate merged_with_ages.parquet; pass --data')

    spectra = load_and_clean(data_path)
    labels, labels_err, fluxes, fluxes_err, bad_frac = build_arrays(spectra)
    print(f'{fluxes.shape[0]} stars, {fluxes.shape[1]} pixels, '
          f'bad pixel fraction: {bad_frac:.3f}')

    perm = np.random.default_rng(args.seed).permutation(labels.shape[0])
    n_test = int(round(args.test_size * labels.shape[0]))
    test_mask, train_mask = perm[:n_test], perm[n_test:]

    os.makedirs(args.outdir, exist_ok=True)
    grid = [(P, l2) for P in args.P for l2 in args.l2]
    print(f'sweeping {len(grid)} configs: P in {args.P}, l2 in {args.l2}')

    for P, l2 in grid:
        run = wandb.init(
            project=args.project,
            mode=args.wandb_mode,
            name=f'P{P}-l2{l2:g}',
            group=f'seed{args.seed}',
            config={
                'P': P,
                'l2_reg_strength': l2,
                'n_iterations': args.n_iterations,
                'test_size': args.test_size,
                'seed': args.seed,
                'n_stars': int(labels.shape[0]),
                'n_train': int(len(train_mask)),
                'n_pixels': int(fluxes.shape[1]),
                'label_names': LABEL_NAMES,
                'data': os.path.abspath(data_path),
                'device': devices[0].platform,
            },
            reinit=True,
        )

        model = LuxModel(P=P)
        model.fit(labels[train_mask], labels_err[train_mask],
                  fluxes[train_mask], fluxes_err[train_mask],
                  n_iterations=args.n_iterations, l2_reg_strength=l2,
                  label_names=LABEL_NAMES, verbose=False)

        # offline mode has no live dashboard, so replaying the per-iteration
        # chi2 history after the fit loses nothing
        for step, chi2 in enumerate(model.chi2_history):
            run.log({'train/chi2': chi2}, step=step)

        metrics = evaluate(model, labels, labels_err, fluxes, fluxes_err,
                           test_mask)
        metrics['train/nll'] = model.nll
        metrics['train/final_chi2'] = model.chi2_history[-1]
        run.log(metrics)

        model_path = os.path.join(args.outdir, f'lux-P{P}-l2{l2:g}.dill')
        model.save(model_path)
        run.summary['model_path'] = os.path.abspath(model_path)
        run.finish()
        print(f'P={P} l2={l2:g}: nll={model.nll:.1f}, '
              f'age scatter={metrics["test/age_frac_scatter"]:.0%}, '
              f'saved {model_path}')


if __name__ == '__main__':
    main()

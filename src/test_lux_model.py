"""
Smoke test for the LuxModel API on synthetic data drawn from the model's
own generative process. Run from the src/ directory:

        python test_lux_model.py
"""

import numpy as np
import jax.numpy as jnp

from lux_model import LuxModel, NotTrainedError


def make_synthetic(n_stars, n_labels, n_wavelengths, P, seed=42):
    rng = np.random.default_rng(seed)

    zetas_true = np.hstack([np.ones((n_stars, 1)),
                            rng.normal(0., 1., (n_stars, P - 1))])
    # labels load only on the non-continuum latent dimensions, so that
    # inferring zetas from labels is well determined (with P = M + 1 the
    # continuum dimension is unconstrained by labels and must stay at its
    # initial value of one)
    alphas_true = rng.normal(0., 1., (n_labels, P))
    alphas_true[:, 0] = 0.
    betas_true = rng.normal(0., 0.1, (n_wavelengths, P))
    betas_true[:, 0] = 1.  # continuum

    labels_err = np.full((n_stars, n_labels), 0.05)
    fluxes_err = np.full((n_stars, n_wavelengths), 0.01)
    labels = zetas_true @ alphas_true.T + rng.normal(0., 1., (n_stars, n_labels)) * labels_err
    fluxes = zetas_true @ betas_true.T + rng.normal(0., 1., (n_stars, n_wavelengths)) * fluxes_err

    return labels, labels_err, fluxes, fluxes_err


def main():
    n_stars, n_labels, n_wavelengths, P = 40, 3, 60, 4
    labels, labels_err, fluxes, fluxes_err = make_synthetic(
        n_stars, n_labels, n_wavelengths, P)

    train = slice(0, 30)
    test = slice(30, 40)

    # --- validation errors ---------------------------------------------
    try:
        LuxModel(P=2).fit(labels, labels_err, fluxes, fluxes_err, verbose=False)
        raise AssertionError("expected ValueError for P < M + 1")
    except ValueError:
        pass

    try:
        LuxModel(P=P).predict_labels(fluxes, fluxes_err)
        raise AssertionError("expected NotTrainedError before fit")
    except NotTrainedError:
        pass

    # --- training -------------------------------------------------------
    model = LuxModel(P=P)
    model.fit(labels[train], labels_err[train], fluxes[train], fluxes_err[train],
              n_iterations=5, l2_reg_strength=1.0,
              label_names=['teff', 'logg', 'fe_h'])

    assert model.is_trained
    assert model.alphas.shape == (n_labels, P)
    assert model.betas.shape == (n_wavelengths, P)
    assert model.zetas.shape == (30, P)
    assert model.ln_noise_fluxes.shape == (n_wavelengths,)
    assert model.chi2_history[-1] <= model.chi2_history[0]

    # --- label prediction from spectra ----------------------------------
    labels_pred, zetas_test = model.predict_labels(
        fluxes[test], fluxes_err[test], return_zetas=True)
    assert labels_pred.shape == (10, n_labels)
    assert zetas_test.shape == (10, P)
    label_scatter = np.std(np.asarray(labels_pred) - labels[test], axis=0)
    print("label prediction scatter:", label_scatter)
    assert np.all(label_scatter < 0.5), "label predictions are far off truth"

    # --- flux prediction from labels -------------------------------------
    fluxes_pred = model.predict_fluxes(labels[test], labels_err[test])
    assert fluxes_pred.shape == (10, n_wavelengths)
    flux_rms = np.sqrt(np.mean((np.asarray(fluxes_pred) - fluxes[test])**2))
    print("flux prediction rms:", flux_rms)
    assert flux_rms < 0.1, "flux predictions are far off truth"

    # --- save / load roundtrip -------------------------------------------
    import tempfile, os
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'lux-model.dill')
        model.save(path)
        restored = LuxModel.load(path)
        assert restored.label_names == ['teff', 'logg', 'fe_h']
        labels_pred2 = restored.predict_labels(fluxes[test], fluxes_err[test])
        assert np.allclose(np.asarray(labels_pred), np.asarray(labels_pred2))

    # --- no-scatter training path ----------------------------------------
    model_ns = LuxModel(P=P)
    model_ns.fit(labels[train], labels_err[train], fluxes[train], fluxes_err[train],
                 n_iterations=3, fit_scatters=False, verbose=False)
    assert model_ns.ln_noise_fluxes is None
    labels_pred_ns = model_ns.predict_labels(fluxes[test], fluxes_err[test])
    assert labels_pred_ns.shape == (10, n_labels)

    print("All LuxModel API tests passed.")


if __name__ == '__main__':
    main()

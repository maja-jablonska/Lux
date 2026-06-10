"""
High-level API for the Lux model.

Wraps the lower-level routines in ``init_latents``, ``optimise``, and
``scatters`` into a single scikit-learn-style model object:

        model = LuxModel(P=16)
        model.fit(labels, labels_err, fluxes, fluxes_err)
        labels_pred = model.predict_labels(test_fluxes, test_fluxes_err)
        fluxes_pred = model.predict_fluxes(test_labels, test_labels_err)
        model.save('lux-model.dill')
        model = LuxModel.load('lux-model.dill')

Shapes follow the paper conventions: N stars, M labels, Lambda wavelengths,
P latent dimensions. alphas are M x P, betas are Lambda x P, zetas are N x P.
"""

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import dill as pickle

import jaxopt

import init_latents as il
import optimise as opt
import scatters as opt_sc

# effectively zero extra pixel noise: exp(2 * -20) ~ 4e-18
_NEGLIGIBLE_LN_NOISE = -20.0


def _labels_map_objective(params, data):
    """
    Negative log-posterior for the test-set zetas given labels only: the
    Gaussian label likelihood plus an empirical Gaussian prior on the zetas
    estimated from the training set. The prior pins latent directions the
    labels cannot constrain (the null space of the alphas), which otherwise
    drift arbitrarily and corrupt synthesised fluxes.
    """
    z = params['zetas']
    chi2 = jnp.nansum((data['labels'] - z @ data['alphas'].T)**2 * data['labels_ivars'])
    dz = z - data['zeta_mean']
    prior = jnp.sum((dz @ data['zeta_prec']) * dz)
    return 0.5 * (chi2 + prior)


@jax.jit
def _run_labels_map(zetas_init, labels, labels_ivars, alphas, zeta_mean, zeta_prec):
    # jitted so repeated calls with the same shapes reuse the compiled program
    optimizer = jaxopt.LBFGS(fun=_labels_map_objective, tol=1e-6,
                             maxiter=3000, max_stepsize=1e3)
    res = optimizer.run(
        init_params={'zetas': zetas_init},
        data={'labels': labels, 'labels_ivars': labels_ivars,
              'alphas': alphas, 'zeta_mean': zeta_mean, 'zeta_prec': zeta_prec})
    return res.params['zetas']


class NotTrainedError(RuntimeError):
    """Raised when prediction is attempted before the model is trained."""


class LuxModel:
    """
    Generative, multi-output, latent-variable model for spectra with noisy labels.

    INPUT:
            P: size of the latent parameter space. Must be at least M + 1
               (one continuum dimension plus one per label) for the zeta
               initialisation to be well defined
            omega: weight of the label term in the joint likelihood
                   (kept at 1 in the paper)
    """

    def __init__(self, P, omega=1.0):
        if int(P) < 1:
            raise ValueError("P must be a positive integer")
        self.P = int(P)
        self.omega = float(omega)

        # trained latents
        self.alphas = None            # M x P
        self.betas = None             # Lambda x P
        self.zetas = None             # N_train x P
        self.ln_noise_fluxes = None   # Lambda (only when scatters are fitted)

        # training metadata
        self.label_names = None
        self.chi2_history = []
        self.nll = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_trained(self):
        return self.alphas is not None and self.betas is not None

    @property
    def n_labels(self):
        return None if self.alphas is None else self.alphas.shape[0]

    @property
    def n_wavelengths(self):
        return None if self.betas is None else self.betas.shape[0]

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, labels, labels_err, fluxes, fluxes_err, n_iterations=5,
            fit_scatters=True, l2_reg_strength=1.0, ln_noise_fluxes_init=-8.0,
            label_names=None, verbose=True):
        """
        Train the model latents on a set of stars with known labels and spectra.

        INPUT:
                labels: stellar labels; size N x M
                labels_err: label errors; size N x M
                fluxes: stellar fluxes; size N x Lambda
                fluxes_err: flux errors; size N x Lambda
                n_iterations: number of passes of the (un-regularised)
                              alpha/beta/zeta coordinate-descent agenda
                fit_scatters: if True, follow up with the regularised agenda
                              that also fits per-pixel noise (scatters)
                l2_reg_strength: L2 regularisation strength on the zetas
                                 (only used when fit_scatters is True)
                ln_noise_fluxes_init: initial value of the logarithmic
                                      per-pixel scatters
                label_names: optional list of M label names, stored for
                             bookkeeping
                verbose: print chi2 progress
        OUTPUT:
                self, with alphas, betas, zetas (and optionally
                ln_noise_fluxes) populated
        """
        labels = jnp.atleast_2d(jnp.asarray(labels, dtype=jnp.float64))
        labels_err = jnp.atleast_2d(jnp.asarray(labels_err, dtype=jnp.float64))
        fluxes = jnp.atleast_2d(jnp.asarray(fluxes, dtype=jnp.float64))
        fluxes_err = jnp.atleast_2d(jnp.asarray(fluxes_err, dtype=jnp.float64))

        self._validate_pair(labels, labels_err, 'labels')
        self._validate_pair(fluxes, fluxes_err, 'fluxes')
        if labels.shape[0] != fluxes.shape[0]:
            raise ValueError("labels and fluxes must describe the same N stars: "
                             f"got {labels.shape[0]} and {fluxes.shape[0]}")

        n_stars, n_labels = labels.shape
        if self.P < n_labels + 1:
            raise ValueError(f"P={self.P} is too small for M={n_labels} labels; "
                             "the zeta initialisation requires P >= M + 1")
        if label_names is not None and len(label_names) != n_labels:
            raise ValueError(f"label_names has {len(label_names)} entries "
                             f"but labels has M={n_labels} columns")
        self.label_names = list(label_names) if label_names is not None else None

        alphas, betas, zetas = il.initialise_alphas_betas_zetas(labels, fluxes, self.P)
        alphas = jnp.asarray(alphas)
        betas = jnp.asarray(betas)
        zetas = jnp.asarray(zetas)

        self.chi2_history = []
        for i in range(n_iterations):
            alphas, betas, zetas, diff_chi2, chi2 = opt.run_agenda(
                alphas, betas, zetas, labels, labels_err, fluxes, fluxes_err, self.omega)
            self.chi2_history.append(float(chi2))
            if verbose:
                print(f"iteration {i + 1}/{n_iterations}: chi2 = {float(chi2):.4f} "
                      f"(improvement {float(diff_chi2):.4f})")

        self.nll = None
        self.ln_noise_fluxes = None
        if fit_scatters:
            labels_ivars = 1. / labels_err**2
            fluxes_ivars = 1. / fluxes_err**2
            ln_noise_fluxes = jnp.full(fluxes.shape[1], float(ln_noise_fluxes_init))
            betas, zetas, ln_noise_fluxes, nll = opt_sc.run_agenda(
                alphas, betas, zetas, labels, labels_ivars, fluxes, fluxes_ivars,
                ln_noise_fluxes, float(l2_reg_strength), self.omega)
            self.ln_noise_fluxes = ln_noise_fluxes
            self.nll = float(nll)
            if verbose:
                print(f"scatter fit: negative log-likelihood = {self.nll:.4f}")

        self.alphas = alphas
        self.betas = betas
        self.zetas = zetas
        return self

    # ------------------------------------------------------------------
    # Latent inference for new stars
    # ------------------------------------------------------------------

    def infer_zetas_from_fluxes(self, fluxes, fluxes_err):
        """
        Find the latent zetas of new stars from their spectra, holding the
        trained betas (and pixel scatters, if fitted) fixed.

        INPUT:
                fluxes: fluxes for the new stars; size N x Lambda
                fluxes_err: flux errors for the new stars; size N x Lambda
        OUTPUT:
                zetas: size N x P
        """
        self._require_trained()
        fluxes = jnp.atleast_2d(jnp.asarray(fluxes, dtype=jnp.float64))
        fluxes_err = jnp.atleast_2d(jnp.asarray(fluxes_err, dtype=jnp.float64))
        self._validate_pair(fluxes, fluxes_err, 'fluxes')
        if fluxes.shape[1] != self.n_wavelengths:
            raise ValueError(f"fluxes has {fluxes.shape[1]} wavelengths but the "
                             f"model was trained with Lambda={self.n_wavelengths}")

        fluxes_ivars = 1. / fluxes_err**2
        ln_noise_fluxes = self.ln_noise_fluxes
        if ln_noise_fluxes is None:
            ln_noise_fluxes = jnp.full(self.n_wavelengths, _NEGLIGIBLE_LN_NOISE)

        zetas_init = self._init_test_zetas(fluxes.shape[0])
        res = opt_sc.get_zetas_test_using_fluxes(
            fluxes, fluxes_ivars, self.betas, zetas_init, ln_noise_fluxes)
        return res.params['zetas']

    def infer_zetas_from_labels(self, labels, labels_err, use_prior=True):
        """
        Find the latent zetas of new stars from their labels, holding the
        trained alphas fixed.

        With M labels and P > M latent dimensions this inference is
        under-determined, so by default an empirical Gaussian prior built
        from the training zetas keeps the unconstrained latent directions
        at typical training values (a MAP estimate). Set use_prior=False
        for the original, unregularised maximum-likelihood behaviour.

        INPUT:
                labels: labels for the new stars; size N x M
                labels_err: label errors for the new stars; size N x M
                use_prior: regularise with the training-set zeta
                           distribution (recommended)
        OUTPUT:
                zetas: size N x P
        """
        self._require_trained()
        labels = jnp.atleast_2d(jnp.asarray(labels, dtype=jnp.float64))
        labels_err = jnp.atleast_2d(jnp.asarray(labels_err, dtype=jnp.float64))
        self._validate_pair(labels, labels_err, 'labels')
        if labels.shape[1] != self.n_labels:
            raise ValueError(f"labels has {labels.shape[1]} columns but the "
                             f"model was trained with M={self.n_labels}")

        labels_ivars = 1. / labels_err**2
        zetas_init = self._init_test_zetas(labels.shape[0])

        if not use_prior or self.zetas is None:
            res = opt_sc.get_zetas_test_using_labels(
                labels, labels_ivars, self.alphas, zetas_init)
            return res.params['zetas']

        zeta_mean, zeta_prec = self._zeta_prior()
        return _run_labels_map(zetas_init, labels, labels_ivars,
                               self.alphas, zeta_mean, zeta_prec)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def synthesise_labels(self, zetas):
        """Synthesise labels (N x M) from latent zetas (N x P)."""
        self._require_trained()
        return jnp.atleast_2d(jnp.asarray(zetas)) @ self.alphas.T

    def synthesise_fluxes(self, zetas):
        """Synthesise fluxes (N x Lambda) from latent zetas (N x P)."""
        self._require_trained()
        return jnp.atleast_2d(jnp.asarray(zetas)) @ self.betas.T

    def predict_labels(self, fluxes, fluxes_err, return_zetas=False):
        """
        Predict the labels of new stars from their spectra.

        INPUT:
                fluxes: fluxes for the new stars; size N x Lambda
                fluxes_err: flux errors for the new stars; size N x Lambda
                return_zetas: also return the inferred latent zetas
        OUTPUT:
                labels: size N x M (and zetas, size N x P, if requested)
        """
        zetas = self.infer_zetas_from_fluxes(fluxes, fluxes_err)
        labels = self.synthesise_labels(zetas)
        return (labels, zetas) if return_zetas else labels

    def predict_fluxes(self, labels, labels_err, return_zetas=False, use_prior=True):
        """
        Predict the spectra of new stars from their labels.

        INPUT:
                labels: labels for the new stars; size N x M
                labels_err: label errors for the new stars; size N x M
                return_zetas: also return the inferred latent zetas
                use_prior: see infer_zetas_from_labels
        OUTPUT:
                fluxes: size N x Lambda (and zetas, size N x P, if requested)
        """
        zetas = self.infer_zetas_from_labels(labels, labels_err, use_prior=use_prior)
        fluxes = self.synthesise_fluxes(zetas)
        return (fluxes, zetas) if return_zetas else fluxes

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path):
        """Save the trained model to ``path`` (dill pickle)."""
        state = {
            'P': self.P,
            'omega': self.omega,
            'alphas': None if self.alphas is None else np.asarray(self.alphas),
            'betas': None if self.betas is None else np.asarray(self.betas),
            'zetas': None if self.zetas is None else np.asarray(self.zetas),
            'ln_noise_fluxes': None if self.ln_noise_fluxes is None
                               else np.asarray(self.ln_noise_fluxes),
            'label_names': self.label_names,
            'chi2_history': self.chi2_history,
            'nll': self.nll,
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        return path

    @classmethod
    def load(cls, path):
        """Load a model previously saved with ``save``."""
        with open(path, 'rb') as f:
            state = pickle.load(f)
        model = cls(P=state['P'], omega=state['omega'])
        for key in ('alphas', 'betas', 'zetas', 'ln_noise_fluxes'):
            value = state[key]
            setattr(model, key, None if value is None else jnp.asarray(value))
        model.label_names = state['label_names']
        model.chi2_history = state['chi2_history']
        model.nll = state['nll']
        return model

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _zeta_prior(self):
        # empirical Gaussian prior on the zetas from the training set; the
        # jitter keeps the precision finite along latent directions with
        # (near-)zero variance in training
        zeta_mean = jnp.mean(self.zetas, axis=0)
        centred = self.zetas - zeta_mean
        cov = (centred.T @ centred) / max(self.zetas.shape[0] - 1, 1)
        jitter = 1e-6 * jnp.trace(cov) / self.P
        zeta_prec = jnp.linalg.inv(cov + jitter * jnp.eye(self.P))
        return zeta_mean, zeta_prec

    def _init_test_zetas(self, n_stars):
        # start each test star at the mean of the training zetas: latent
        # directions the data cannot constrain (e.g. the null space of the
        # alphas when inferring from labels alone) then stay at typical
        # training values rather than at an arbitrary point
        if self.zetas is not None:
            return jnp.tile(jnp.mean(self.zetas, axis=0), (n_stars, 1))
        zetas = jnp.zeros((n_stars, self.P))
        return zetas.at[:, 0].set(1.)

    def _require_trained(self):
        if not self.is_trained:
            raise NotTrainedError("model has not been trained; call fit() or load() first")

    @staticmethod
    def _validate_pair(values, errors, name):
        if values.ndim != 2:
            raise ValueError(f"{name} must be a 2D array (stars x {name}), "
                             f"got shape {values.shape}")
        if values.shape != errors.shape:
            raise ValueError(f"{name} and {name}_err must have the same shape: "
                             f"got {values.shape} and {errors.shape}")
        if not bool(jnp.all(errors > 0)):
            raise ValueError(f"{name}_err must be strictly positive everywhere "
                             "(use a large value, e.g. 9999, for missing data)")

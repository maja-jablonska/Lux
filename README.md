# Lux ⚡️

An implementation of the method Lux: A generative, multi-output, latent-variable model for astronomical data with noisy labels

## Python API

`src/lux_model.py` wraps the lower-level routines (`init_latents`, `optimise`, `scatters`)
into a single model object:

```python
from lux_model import LuxModel

# train on N stars with M labels and Lambda wavelengths (P >= M + 1)
model = LuxModel(P=24)
model.fit(labels, labels_err, fluxes, fluxes_err,
          n_iterations=5, l2_reg_strength=1.0,
          label_names=['TEFF', 'LOGG', 'FE_H'])

# predict labels for new stars from their spectra
labels_pred = model.predict_labels(test_fluxes, test_fluxes_err)

# or synthesise spectra for new stars from their labels
fluxes_pred = model.predict_fluxes(test_labels, test_labels_err)

# persistence
model.save('lux-model.dill')
model = LuxModel.load('lux-model.dill')
```

`fit` runs the un-regularised alpha/beta/zeta coordinate-descent agenda for
`n_iterations`, then (by default) the regularised agenda that also fits
per-pixel noise scatters. The latent zetas of new stars can be inferred from
spectra (`infer_zetas_from_fluxes`) or from labels (`infer_zetas_from_labels`);
label-based inference is under-determined when P > M, so by default it uses a
MAP estimate with an empirical Gaussian prior built from the training zetas
(pass `use_prior=False` for the original maximum-likelihood behaviour).

See `notebooks/example-lux-api.ipynb` for a full worked example, and run the
test suite with:

```bash
cd src && python test_lux_model.py
```

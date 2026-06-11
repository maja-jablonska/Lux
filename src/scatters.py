# import necessary modules
import jax.numpy as jnp
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
from astropy.io import fits
import os
import jaxopt
from functools import partial
import tqdm
import dill as pickle

                        
##################################################################################################################
# FUNCTIONS TO GET THE DATA IN THE RIGHT FORMAT FOR OPTIMISATION INCLUDING NOISE and regularisation in z latents
##################################################################################################################

def get_data_beta_step_fixed_betas_opt_fluxnoise(betas, zetas, fluxes, fluxes_ivars, ln_noise_fluxes):
                
        """
                Function to get the data and parameters for the wavelength plate stored in dictionary format to optimise the flux scatters
                INPUT: 
                        fluxes: array of flux values; size N x Lambda 
                        fluxes_ivars: array of inverse flux variances; size N x Lambda
                        betas: latent parameters for the wavelengths, size Lambda x P
                        zetas: latent parameters for the stars, size N x P
                        ln_noise_fluxes: logarithmic scatters in the pixels, Lambda
                OUTPUT:
                        params: ln_noise_fluxes
                        data: fluxes, ivars, betas, zetas
        """
             
        params = {'ln_noise_fluxes': ln_noise_fluxes}
        data = {'betas': betas, 'zetas': zetas, 'fluxes': fluxes, 'fluxes_ivars': fluxes_ivars}

        return params, data

def get_data_beta_step_fixed_fluxnoise_opt_betas(betas, zetas, fluxes, fluxes_ivars, ln_noise_fluxes):
                
        """
                Function to get the data and parameters for the wavelength plate stored in dictionary format to optimise betas
                INPUT: 
                        fluxes: array of flux values; size N x Lambda 
                        fluxes_ivars: array of inverse flux variances; size N x Lambda
                        betas: latent parameters for the wavelengths, size Lambda x P
                        zetas: latent parameters for the stars, size N x P
                        ln_noise_fluxes: logarithmic scatters in the pixels, Lambda
                OUTPUT:
                        params: betas
                        data: fluxes, ivars, zetas, ln_noise_fluxes
        """
             
        params = {'betas': betas}
        data = {'zetas': zetas, 'fluxes': fluxes, 'fluxes_ivars': fluxes_ivars, 'ln_noise_fluxes': ln_noise_fluxes}

        return params, data

def get_data_zeta_step(alphas, betas, zetas, labels, labels_ivars, fluxes, fluxes_ivars, ln_noise_fluxes):
                
        """
                Function to get the data and parameters for the star plate stored in dictionary format to optimise zetas and scatters in pixels
                INPUT:
                        labels: array of flux values; size N x Lambda 
                        labels_ivars: array of flux values; size N x Lambda 
                        fluxes: array of flux values; size N x Lambda 
                        fluxes_ivars: array of inverse flux variances; size N x Lambda
                        alphas: latent parameters for the labels, size M x P
                        betas: latent parameters for the wavelengths, size Lambda x P
                        zetas: latent parameters for the stars, size N x P
                        ln_noise_fluxes: logarithmic scatters in the pixels, Lambda
                OUTPUT: 
                        params: zetas, ln_noise_fluxes
                        data: alphas, betas, labels, labels_ivars, fluxes, fluxes_ivars
        """
             
        params = {'zetas': zetas, 'ln_noise_fluxes': ln_noise_fluxes}
        data = {'alphas': alphas, 'betas': betas,\
                'labels': labels, 'labels_ivars': labels_ivars, \
                'fluxes': fluxes, 'fluxes_ivars': fluxes_ivars}

        return params, data


############################################################
# FUNCTIONS TO ESTIMATE THE LATENT PARAMETERS
############################################################

################# FOR ALPHAS

def synthesise_all_labels_all_stars(zeta, alpha):
        """
                Function to use in the loss function for optimising all labels and all stars. For now, this is treated as linear model, but
                could in principle be replaced with any functional form (e.g., quadratic, CNN, GP)

                INPUT: 
                        zeta: latent parameter zetas
                        alpha: latent parameter alphas

                OUTPUT:
                        dot product of zeta and alpha
        """

        return zeta @ alpha.T

def all_labels_all_stars_Gaussian_likelihood(params, data):
        """
                Gaussian log-likelihood function for all labels and all stars
                INPUT: 
                        params and data from get_data_zeta_step function
                OUTPUT:
                        log-likelihood 
        """
        
        model_label = synthesise_all_labels_all_stars(data['zetas'], params['alphas'])
        noise = 1./data['labels_ivars'] # here the scatter values are the variances
        # compute the chi
        return -0.5 * jnp.nansum((data['labels'] - model_label)**2 / (noise)) 

def one_label_all_stars_objective(params, data):
        """
                Objective function for all labels and all stars
                INPUT: 
                        params and data from get_data_zeta_step function
                OUTPUT:
                        -log-likelihood 
        """
        return - (all_labels_all_stars_Gaussian_likelihood(params, data))

################# FOR BETAS

def synthesise_all_wavelengths_all_stars(zeta, beta):
        """
                Function to use in the loss function for optimising all wavelengths and all stars. For now, this is treated as linear model, but
                could in principle be replaced with any functional form (e.g., quadratic, CNN, GP)

                INPUT: 
                        zeta: latent parameter zetas
                        beta: latent parameter betas

                OUTPUT:
                        dot product of zeta and beta
        """

        return zeta @ beta.T
    
def all_wavelengths_all_stars_Gaussian_likelihood_fixed_fluxnoise_opt_betas(params, data):
        """
                Gaussian log-likelihood function for all wavelengths and all stars to optimise betas
                INPUT: 
                        params and data from get_data_beta_step_fixed_fluxnoise_opt_betas function
                OUTPUT:
                        log-likelihood 
        """
        model_flux = synthesise_all_wavelengths_all_stars(data['zetas'], params['betas'])
        V = jnp.exp(2 * data['ln_noise_fluxes'])
        noise = 1./ data['fluxes_ivars'] + V 
          
        return -0.5 * jnp.nansum((data['fluxes'] - model_flux)**2 / (noise)) -0.5 * jnp.nansum(jnp.log(noise))

def all_wavelengths_all_stars_objective_fixed_fluxnoise_opt_betas(params, data):
        """
                Objective function for all labels and all stars to optimise betas
                INPUT: 
                        params and data from get_data_beta_step_fixed_fluxnoise_opt_betas function
                OUTPUT:
                        -log-likelihood 
        """
        return - all_wavelengths_all_stars_Gaussian_likelihood_fixed_fluxnoise_opt_betas(params, data)

def beta_step_opt_betas(betas, zetas, fluxes, fluxes_ivar, ln_noise_fluxes):

        """
                Function to optimise the beta parameters at fixed scatters and zetas using the fluxes and flux ivars
                INPUT: 
                        betas: latent parameters for wavelengths, Lambda x P
                        zetas: latent parameters, N x P
                        fluxes: fluxes for all stars, N x Lambda
                        fluxes_ivar: flux inverse variances for all stars, N x Lambda
                        ln_noise_fluxes: logarithmic scatters in the pixels, Lambda
                OUTPUT:
                        optimised betas
        """

        params, data = get_data_beta_step_fixed_fluxnoise_opt_betas(betas, zetas, fluxes, fluxes_ivar, ln_noise_fluxes)

        optimizer = jaxopt.LBFGS(fun=all_wavelengths_all_stars_objective_fixed_fluxnoise_opt_betas, tol=1e-6, maxiter=1000, max_stepsize=1e3) # Magic numbers

        res = optimizer.run(init_params = params, data = data) 

        return res

################# FOR NOISE FLUXES
    
def all_wavelengths_all_stars_Gaussian_likelihood_fixed_betas_opt_fluxnoise(params, data):
        """
                Gaussian log-likelihood function for all wavelengths and all stars to optimise scatters in pixels
                INPUT: 
                        params and data from get_data_beta_step_fixed_betas_opt_fluxnoise function
                OUTPUT:
                        log-likelihood 
        """
        model_flux = synthesise_all_wavelengths_all_stars(data['zetas'], data['betas'])
        V = jnp.exp(2 * params['ln_noise_fluxes'])
        noise = 1./ data['fluxes_ivars'] + V 
          
        return -0.5 * jnp.nansum((data['fluxes'] - model_flux)**2 / (noise)) -0.5 * jnp.nansum(jnp.log(noise))

def all_wavelengths_all_stars_objective_fixed_betas_opt_fluxnoise(params, data):
        """
                Objective function for all labels and all stars to optimise scatters
                INPUT: 
                        params and data from get_data_beta_step_fixed_betas_opt_fluxnoise function
                OUTPUT:
                        -log-likelihood 
        """
        return - all_wavelengths_all_stars_Gaussian_likelihood_fixed_betas_opt_fluxnoise(params, data)

def beta_step_opt_fluxnoise(betas, zetas, fluxes, fluxes_ivar, ln_noise_fluxes):

        """
                Function to optimise the beta parameters at fixed scatters and zetas using the fluxes and flux ivars
                INPUT:
                        betas: latent parameters for wavelengths, Lambda x P
                        zetas: latent parameters, N x P
                        fluxes: fluxes for all stars, N x Lambda
                        fluxes_ivar: flux inverse variances for all stars, N x Lambda
                        ln_noise_fluxes: logarithmic scatters in the pixels, Lambda
                OUTPUT:
                        optimised scatters
        """

        params, data = get_data_beta_step_fixed_betas_opt_fluxnoise(betas, zetas, fluxes, fluxes_ivar, ln_noise_fluxes)

        optimizer = jaxopt.LBFGS(fun=all_wavelengths_all_stars_objective_fixed_betas_opt_fluxnoise, tol=1e-6, maxiter=1000, max_stepsize=1e3) # Magic numbers

        res = optimizer.run(init_params = params, data = data)

        return res

def one_pixel_objective_opt_fluxnoise(params, data):
        """
                Negative Gaussian log-likelihood for the scatter of a single pixel across all stars.
                The full noise objective is separable per pixel at fixed betas and zetas, so each
                pixel's scalar scatter can be optimised independently (and in parallel via vmap)
                INPUT:
                        params: ln_noise for one pixel (scalar)
                        data: beta_h (P), flux_h (N), ivars_h (N), zetas (N x P) for that pixel
                OUTPUT:
                        -log-likelihood for that pixel
        """
        model_flux = data['zetas'] @ data['beta']
        V = jnp.exp(2 * params['ln_noise'])
        noise = 1. / data['fluxes_ivars'] + V
        return 0.5 * jnp.nansum((data['fluxes'] - model_flux)**2 / noise) + 0.5 * jnp.nansum(jnp.log(noise))

def one_pixel_step_opt_fluxnoise(beta_h, fluxes_h, fluxes_ivar_h, ln_noise_h, zetas):
        """
                Optimise the scatter of one pixel at fixed betas and zetas
                INPUT:
                        beta_h: latent parameters for this wavelength, P
                        fluxes_h: fluxes of all stars in this pixel, N
                        fluxes_ivar_h: flux inverse variances in this pixel, N
                        ln_noise_h: initial logarithmic scatter for this pixel (scalar)
                        zetas: latent parameters, N x P
                OUTPUT:
                        optimised scatter for this pixel
        """
        params = {'ln_noise': ln_noise_h}
        data = {'beta': beta_h, 'fluxes': fluxes_h, 'fluxes_ivars': fluxes_ivar_h, 'zetas': zetas}

        optimizer = jaxopt.LBFGS(fun=one_pixel_objective_opt_fluxnoise, tol=1e-6, maxiter=1000, max_stepsize=1e3) # Magic numbers

        res = optimizer.run(init_params = params, data = data)

        return res.params['ln_noise']

# vmap to optimise every pixel's scatter independently; ~2x faster than the joint
# LBFGS over all pixels and equivalent in the fitted variances (pixels whose optimal
# scatter is zero converge to different, equally flat, very negative ln values)
fluxnoise_step = jax.vmap(one_pixel_step_opt_fluxnoise, in_axes=(0, 1, 1, 0, None))

################# FOR ZETAS (not jitted to save memory)

def all_stars_all_wavelengths_all_labels_Gaussian_likelihood(params, data, reg_std, omega):
        """
                Gaussian log-likelihood function for all labels, all wavelengths, and all stars to optimise zetas
                INPUT: 
                        params and data from get_data_zeta_step function
                OUTPUT:
                        log-likelihood 
        """

        model_labels = synthesise_all_labels_all_stars(params['zetas'], data['alphas'])
        noise_labels = 1./ data['labels_ivars'] 
        loglike_labels = -0.5 * jnp.nansum((data['labels'] - model_labels)**2 / (noise_labels))
        
        model_fluxes = synthesise_all_wavelengths_all_stars(params['zetas'], data['betas'])
        V =  jnp.exp(2 * params['ln_noise_fluxes'])
        noise_fluxes = 1./ data['fluxes_ivars'] + V 
        loglike_fluxes = -0.5 * jnp.nansum((data['fluxes'] - model_fluxes)**2 / (noise_fluxes)) -0.5 * jnp.nansum(jnp.log(noise_fluxes))
        l2_reg = - 0.5 * jnp.sum(params['zetas']**2 / reg_std**2)

        return omega * loglike_labels + loglike_fluxes + l2_reg

def all_stars_all_wavelengths_all_labels_objective(params, data, reg_std, omega):
        """
                Objective function for all labels, all wavelengths, and all stars to optimise zetas
                INPUT: 
                        params and data from get_data_zeta_step function
                OUTPUT:
                        -log-likelihood 
        """
        return - all_stars_all_wavelengths_all_labels_Gaussian_likelihood(params, data, reg_std, omega)

def zeta_step(alphas, betas, zetas, labels, labels_ivars, fluxes, fluxes_ivars, ln_noise_fluxes, reg_std, omega):

        """
                Function to optimise the beta parameters at fixed scatters and zetas using the fluxes and flux ivars
                INPUT: 
                        alphas: latent parameters for labels, M x P
                        betas: latent parameters for wavelengths, Lambda x P
                        zetas: latent parameters, N x P
                        labels: labels for all stars, N x Lambda
                        labels_ivar: labels inverse variances for all stars, N x Lambda
                        fluxes: fluxes for all stars, N x Lambda
                        fluxes_ivar: flux inverse variances for all stars, N x Lambda
                        ln_noise_fluxes: logarithmic scatters in the pixels, Lambda
                        reg_std: L2 regularisation strength
                        omega: dummy variable set to 1, but initially included to allow the labels to weigh more in the likelihood
                OUTPUT:
                        optimised zetas
        """

        params, data = get_data_zeta_step(alphas, betas, zetas, labels, labels_ivars, fluxes, fluxes_ivars, ln_noise_fluxes)

        optimizer = jaxopt.LBFGS(fun=all_stars_all_wavelengths_all_labels_objective, tol = 1e-6, maxiter = 1000, max_stepsize=1e3) # Magic numbers

        res = optimizer.run(init_params = params, data = data, reg_std = reg_std, omega = omega) 

        return res      


############################################################
# DIRECT (CLOSED-FORM) BLOCK SOLVERS
# At fixed noise the model is linear in the betas and in the zetas, so both
# block updates are weighted linear least-squares problems with exact
# solutions; iterating them with LBFGS (which previously hit maxiter=1000
# without converging) is unnecessary. The per-pixel noise update is a 1-D
# root find solved by bisection. NaNs in the data are given zero weight,
# matching the nansum semantics of the likelihood functions above.
############################################################

# scatters below exp(_LN_NOISE_FLOOR) add a variance of exp(2 * -20) ~ 4e-18,
# i.e. effectively zero; pixels whose optimal scatter is zero are pinned here
# instead of drifting to arbitrary very negative values as under LBFGS
_LN_NOISE_FLOOR = -20.0

def _masked(values, weights):
        """Zero out the weight (and value) of non-finite data entries."""
        mask = jnp.isfinite(values)
        return jnp.where(mask, values, 0.), jnp.where(mask, weights, 0.)

def beta_step_direct(zetas, fluxes, fluxes_ivars, ln_noise_fluxes):
        """
                Exact betas at fixed zetas and scatters: for each pixel h, solve the
                weighted normal equations (Z^T W_h Z) beta_h = Z^T W_h f_h
                INPUT:
                        zetas: latent parameters, N x P
                        fluxes: fluxes for all stars, N x Lambda
                        fluxes_ivars: flux inverse variances for all stars, N x Lambda
                        ln_noise_fluxes: logarithmic scatters in the pixels, Lambda
                OUTPUT:
                        optimised betas, Lambda x P
        """
        V = jnp.exp(2 * ln_noise_fluxes)
        f, w = _masked(fluxes, 1. / (1. / fluxes_ivars + V[None, :]))
        zz = zetas[:, :, None] * zetas[:, None, :]              # N x P x P
        ZtWZ = jnp.einsum('npq,nh->hpq', zz, w)                 # Lambda x P x P
        ZtWf = jnp.einsum('np,nh->hp', zetas, w * f)            # Lambda x P
        # the jitter only matters for pathological all-masked pixels (beta -> 0)
        return jnp.linalg.solve(ZtWZ + 1e-12 * jnp.eye(zetas.shape[1]),
                                ZtWf[..., None])[..., 0]

def zeta_step_direct(alphas, betas, labels, labels_ivars, fluxes, fluxes_ivars, ln_noise_fluxes, reg_std, omega):
        """
                Exact zetas at fixed alphas, betas, and scatters: for each star n the
                objective is quadratic, so solve the ridge-regularised normal equations
                (omega A^T W_l A + B^T W_f B + I / reg_std^2) zeta_n = rhs_n
                INPUT:
                        alphas: latent parameters for labels, M x P
                        betas: latent parameters for wavelengths, Lambda x P
                        labels: labels for all stars, N x M
                        labels_ivars: label inverse variances for all stars, N x M
                        fluxes: fluxes for all stars, N x Lambda
                        fluxes_ivars: flux inverse variances for all stars, N x Lambda
                        ln_noise_fluxes: logarithmic scatters in the pixels, Lambda
                        reg_std: L2 regularisation strength
                        omega: weight of the label term in the likelihood
                OUTPUT:
                        optimised zetas, N x P
        """
        P = alphas.shape[1]
        V = jnp.exp(2 * ln_noise_fluxes)
        f, wf = _masked(fluxes, 1. / (1. / fluxes_ivars + V[None, :]))
        l, wl = _masked(labels, labels_ivars)
        aa = alphas[:, :, None] * alphas[:, None, :]            # M x P x P
        bb = betas[:, :, None] * betas[:, None, :]              # Lambda x P x P
        G = omega * jnp.einsum('mpq,nm->npq', aa, wl) \
            + jnp.einsum('hpq,nh->npq', bb, wf) \
            + jnp.eye(P) / reg_std**2                           # N x P x P
        rhs = omega * (wl * l) @ alphas + (wf * f) @ betas      # N x P
        return jnp.linalg.solve(G, rhs[..., None])[..., 0]

def _pixel_noise_score(V, resid2, var_data, weights):
        """
                Per-pixel derivative score h(V) = sum_n (r_n^2 - (c_n + V)) / (c_n + V)^2;
                the per-pixel likelihood is stationary where h crosses zero, and the
                optimal extra variance is zero when h(0) <= 0
        """
        t = var_data + V[None, :]
        return jnp.sum(weights * (resid2 - t) / t**2, axis=0)

def fluxnoise_step_direct(betas, fluxes, fluxes_ivars, zetas):
        """
                Exact per-pixel scatters at fixed betas and zetas, via bisection on the
                stationarity condition in the extra-variance V = exp(2 ln_noise). V = 0
                (no extra scatter) is detected from the score at V = 0 and mapped to
                _LN_NOISE_FLOOR
                INPUT:
                        betas: latent parameters for wavelengths, Lambda x P
                        fluxes: fluxes for all stars, N x Lambda
                        fluxes_ivars: flux inverse variances for all stars, N x Lambda
                        zetas: latent parameters, N x P
                OUTPUT:
                        optimised logarithmic scatters, Lambda
        """
        resid2, weights = _masked((fluxes - zetas @ betas.T)**2, jnp.ones_like(fluxes))
        var_data = 1. / fluxes_ivars
        # the optimum satisfies V <= max_n r_n^2, so [0, max r^2] brackets the root
        lo = jnp.zeros(fluxes.shape[1])
        hi = jnp.max(weights * resid2, axis=0)
        score0 = _pixel_noise_score(lo, resid2, var_data, weights)

        def body(_, bracket):
                lo, hi = bracket
                mid = 0.5 * (lo + hi)
                above = _pixel_noise_score(mid, resid2, var_data, weights) > 0
                return jnp.where(above, mid, lo), jnp.where(above, hi, mid)

        lo, hi = jax.lax.fori_loop(0, 64, body, (lo, hi))
        V = 0.5 * (lo + hi)
        ln_noise = jnp.where(score0 > 0,
                             0.5 * jnp.log(jnp.maximum(V, jnp.exp(2 * _LN_NOISE_FLOOR))),
                             _LN_NOISE_FLOOR)
        return jnp.maximum(ln_noise, _LN_NOISE_FLOOR)


###################### RUN FULL AGENDA

def run_agenda(alphas, betas, zetas, labels, labels_ivars, fluxes, fluxes_ivars, ln_noise_fluxes, reg_std, omega):

        """
        One exact coordinate-descent sweep over the scatters, betas, and zetas
        blocks (alphas stay fixed). Each block update is solved in closed form
        (or by per-pixel bisection for the scatters), so a sweep is exact and
        cheap; call repeatedly until the returned likelihood stops improving.

            INPUT:
                alphas: latent parameters for labels, M x P
                betas: latent parameters for wavelengths, Lambda x P
                zetas: latent parameters, N x P
                labels: labels for all stars, N x M
                labels_ivars: labels inverse variances for all stars, N x M
                fluxes: fluxes for all stars, N x Lambda
                fluxes_ivars: flux inverse variances for all stars, N x Lambda
                ln_noise_fluxes: logarithmic scatters in the pixels, Lambda
                        (unused: the scatter update is exact and needs no initial value;
                        kept for backwards compatibility of the call signature)
                reg_std: L2 regularisation strength
                omega: dummy variable set to 1, but initially included to allow the labels to weigh more in the likelihood
            OUTPUT:
                updated optimised betas, zetas, ln_noise_fluxes, and the negative
                log-likelihood evaluated at the returned parameters
        """

        # exact per-pixel scatters at the current betas and zetas
        ln_noise_fluxes_updated = fluxnoise_step_direct(betas, fluxes, fluxes_ivars, zetas)

        # closed-form betas at the updated scatters
        betas_updated = beta_step_direct(zetas, fluxes, fluxes_ivars, ln_noise_fluxes_updated)

        # closed-form zetas at the updated betas and scatters (each block update
        # sees the latest values of the others, unlike the previous LBFGS agenda
        # which optimised the zetas against the stale betas and scatters)
        zetas_updated = zeta_step_direct(alphas, betas_updated, labels, labels_ivars,
                                         fluxes, fluxes_ivars, ln_noise_fluxes_updated, reg_std, omega)

        params = {'zetas': zetas_updated, 'ln_noise_fluxes': ln_noise_fluxes_updated}
        data = {'alphas': alphas, 'betas': betas_updated,
                'labels': labels, 'labels_ivars': labels_ivars,
                'fluxes': fluxes, 'fluxes_ivars': fluxes_ivars}
        nll = all_stars_all_wavelengths_all_labels_objective(params, data, reg_std, omega)

        return betas_updated, zetas_updated, ln_noise_fluxes_updated, nll

# jit the full agenda so repeated sweeps reuse the compiled program
run_agenda = jax.jit(run_agenda)

############################################################
# DIRECT (CLOSED-FORM) TEST-SET ZETA SOLVERS
# At fixed betas (or alphas) and scatters the test-set objectives are
# quadratic in the zetas, so they too have exact solutions. A tiny ridge
# anchored at zetas_init keeps latent directions the data cannot constrain
# (e.g. the null space of the alphas when P > M) at their initial values,
# matching the behaviour of the LBFGS solvers below, which leave such
# directions untouched
############################################################

def _anchored_solve(G, rhs, zetas_init):
        """
                Solve G z = rhs per star with a negligible ridge towards zetas_init:
                exact where the data constrain z, and z = zetas_init along directions
                of zero curvature
        """
        P = G.shape[-1]
        # the ridge must sit far below the smallest genuine eigenvalue of G,
        # not just below its trace (which the largest eigenvalue dominates
        # when G is ill-conditioned, e.g. via the continuum direction)
        eps = 1e-12 * jnp.trace(G, axis1=-2, axis2=-1)[:, None] / P + 1e-300
        G = G + eps[..., None] * jnp.eye(P)
        rhs = rhs + eps * zetas_init
        return jnp.linalg.solve(G, rhs[..., None])[..., 0]

def get_zetas_test_using_fluxes_direct(fluxes, fluxes_ivars, betas, zetas_init, ln_noise_fluxes):
        """
                Exact test-set zetas from spectra at fixed betas and scatters: for each
                star n, solve the weighted normal equations (B^T W_n B) zeta_n = B^T W_n f_n
                INPUT:
                        fluxes: fluxes for the new stars, N x Lambda
                        fluxes_ivars: flux inverse variances, N x Lambda
                        betas: trained latent parameters for wavelengths, Lambda x P
                        zetas_init: anchor for unconstrained latent directions, N x P
                        ln_noise_fluxes: trained logarithmic scatters in the pixels, Lambda
                OUTPUT:
                        optimised zetas, N x P
        """
        V = jnp.exp(2 * ln_noise_fluxes)
        f, w = _masked(fluxes, 1. / (1. / fluxes_ivars + V[None, :]))
        bb = betas[:, :, None] * betas[:, None, :]              # Lambda x P x P
        G = jnp.einsum('hpq,nh->npq', bb, w)                    # N x P x P
        rhs = (w * f) @ betas                                   # N x P
        return _anchored_solve(G, rhs, zetas_init)

get_zetas_test_using_fluxes_direct = jax.jit(get_zetas_test_using_fluxes_direct)

def get_zetas_test_using_labels_direct(labels, labels_ivars, alphas, zetas_init):
        """
                Exact test-set zetas from labels at fixed alphas: for each star n,
                solve the weighted normal equations (A^T W_n A) zeta_n = A^T W_n l_n.
                With P > M this is under-determined and the null-space directions
                stay at zetas_init
                INPUT:
                        labels: labels for the new stars, N x M
                        labels_ivars: label inverse variances, N x M
                        alphas: trained latent parameters for labels, M x P
                        zetas_init: anchor for unconstrained latent directions, N x P
                OUTPUT:
                        optimised zetas, N x P
        """
        l, w = _masked(labels, labels_ivars)
        aa = alphas[:, :, None] * alphas[:, None, :]            # M x P x P
        G = jnp.einsum('mpq,nm->npq', aa, w)                    # N x P x P
        rhs = (w * l) @ alphas                                  # N x P
        return _anchored_solve(G, rhs, zetas_init)

get_zetas_test_using_labels_direct = jax.jit(get_zetas_test_using_labels_direct)

########################################################################################################################
# TESTING THE MODEL
########################################################################################################################

# NOTE: Idea is, you have now optimised for alphas, betas, and zetas (of the training set). You now need to find the 
# zetas corresponding to the test set (or new stars). Once you have those, you can estimate the labels

################### FOR ZETAS USING FLUX
def get_data_zeta_test_using_fluxes(fluxes, fluxes_ivars, betas, zetas, ln_noise_fluxes):
        """
                Function to get the data and parameters for the star wavelength plate stored in dictionary format to optimise zetas for test set using betas, scatters, and fluxes
                INPUT:
                        fluxes: array of flux values; size N x Lambda 
                        fluxes_ivars: array of inverse flux variances; size N x Lambda
                        betas: latent parameters for the wavelengths, size Lambda x P
                        zetas: latent parameters for the stars, size N x P
                        ln_noise_fluxes: logarithmic scatters in the pixels, Lambda
                OUTPUT: 
                        params: zetas
                        data: betas, fluxes, fluxes_ivars, ln_noise_fluxes
        """
             
        params = {'zetas': zetas}
        data = {'fluxes': fluxes, 'fluxes_ivars': fluxes_ivars,\
                 'betas': betas, 'ln_noise_fluxes': ln_noise_fluxes}

        return params, data

def all_wavelengths_all_stars_Gaussian_likelihood_test(params, data):
        """
                Gaussian log-likelihood function for all wavelengths and all stars to optimise zetas in the test set
                INPUT: 
                        params and data from get_data_zeta_test_using_fluxes function
                OUTPUT:
                        log-likelihood 
        """

        model_flux = synthesise_all_wavelengths_all_stars(params['zetas'], data['betas'])
        V = jnp.exp(2 * data['ln_noise_fluxes'])
        noise = 1./ data['fluxes_ivars'] + V 
          
        return -0.5 * jnp.nansum((data['fluxes'] - model_flux)**2 / (noise)) -0.5 * jnp.nansum(jnp.log(noise))

def all_wavelengths_all_stars_objective_test(params, data):
        """
                Objective function for all wavelengths and all stars to optimise zetas in the test set
                INPUT: 
                        params and data from get_data_zeta_test_using_fluxes function
                OUTPUT:
                        -log-likelihood 
        """
        return -(all_wavelengths_all_stars_Gaussian_likelihood_test(params, data))
        
def get_zetas_test_using_fluxes(fluxes, fluxes_ivars, betas, zetas, ln_noise_fluxes):
        """
                Function to optimise the zetas latents parameters at fixed scatters and betas using the fluxes and flux ivars
                INPUT: 
                        betas: latent parameters for wavelengths, Lambda x P
                        zetas: latent parameters, N x P
                        fluxes: fluxes for all stars, N x Lambda
                        fluxes_ivar: flux inverse variances for all stars, N x Lambda
                        ln_noise_fluxes: logarithmic scatters in the pixels, Lambda
                OUTPUT:
                        optimised zetas
        """

        params, data = get_data_zeta_test_using_fluxes(fluxes, fluxes_ivars, betas, zetas, ln_noise_fluxes)

        optimizer = jaxopt.LBFGS(fun=all_wavelengths_all_stars_objective_test, tol=1e-6, maxiter=3000, max_stepsize=1e3) # magic numbers

        res = optimizer.run(init_params = params, data = data)

        return res

# jit so repeated test-set calls with the same shapes reuse the compiled program
get_zetas_test_using_fluxes = jax.jit(get_zetas_test_using_fluxes)


################### FOR ZETAS USING LABELS
def get_data_zeta_test_using_labels(labels, labels_ivars, alphas, zetas):
        """
                Function to get the data and parameters for the star labels plate stored in dictionary format to optimise zetas for test set using alphas and labels
                INPUT:
                        labels: array of label values; size N x M 
                        labels_ivars: array of inverse label variances; size N x M
                        alphas: latent parameters for the wavelengths, size M x P
                        zetas: latent parameters for the stars, size N x P
                OUTPUT: 
                        params: zetas
                        data: alphas, labels, labels_ivars
        """
             
        params = {'zetas': zetas}
        data = {'labels': labels, 'labels_ivars': labels_ivars, 'alphas': alphas}

        return params, data

def all_labels_all_stars_Gaussian_likelihood_test(params, data):
        """
                Gaussian log-likelihood function for all labels and all stars to optimise zetas in the test set
                INPUT: 
                        params and data from get_data_zeta_test_using_labels function
                OUTPUT:
                        log-likelihood 
        """
        
        model_label = synthesise_all_labels_all_stars(params['zetas'], data['alphas'])
        noise = 1./data['labels_ivars'] # here the scatter values are the variances

        return -0.5 * jnp.nansum((data['labels'] - model_label)**2 / (noise)) 

def all_labels_all_stars_objective_test(params, data):
        """
                Objective function for all labels and all stars to optimise zetas in the test set
                INPUT: 
                        params and data from get_data_zeta_test_using_labels function
                OUTPUT:
                        -log-likelihood 
        """
        return -(all_labels_all_stars_Gaussian_likelihood_test(params, data))

def get_zetas_test_using_labels(labels, labels_ivars, alphas, zetas):
        """
                Function to optimise the zetas latents parameters at fixes alphas using the labels and labels ivars
                INPUT: 
                        alphas: latent parameters for labels, M x P
                        zetas: latent parameters, N x P
                        labels: fluxes for all stars, N x M
                        labels_ivar: label inverse variances for all stars, N x M
                OUTPUT:
                        optimised zetas
        """

        params, data = get_data_zeta_test_using_labels(labels, labels_ivars, alphas, zetas)

        optimizer = jaxopt.LBFGS(fun=all_labels_all_stars_objective_test, tol=1e-6, maxiter=3000, max_stepsize=1e3) # magic numbers

        res = optimizer.run(init_params = params, data = data)

        return res

# jit so repeated test-set calls with the same shapes reuse the compiled program
get_zetas_test_using_labels = jax.jit(get_zetas_test_using_labels)
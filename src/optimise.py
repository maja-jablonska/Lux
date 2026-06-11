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


#################################################################
# FUNCTIONS TO GET THE DATA IN THE RIGHT FORMAT FOR OPTIMISATION
# NOTE: alphas, betas, zetas are equal to A, B, Z in the paper
#################################################################

def get_data_alpha_step(alphas, zetas, labels, labels_err):
                
        """
                Function to get the data and parameters for the label plate stored in dictionary format
                INPUT: 
                        alphas: latent parameters for the labels, size M x P
                        zetas: latent parameters for the stars, size N x P
                        labels: array of labels; size N x M 
                        labels_err: array of label errors; size N x M
                OUTPUT:
                        params: alphas
                        data: labels, labels_err, labels_var, zetas
        """

        params = {'alphas': alphas}
        data = {'labels': labels, 'labels_err': labels_err, 'zetas': zetas}

        return params, data

def get_data_beta_step(betas, zetas, fluxes, fluxes_err):
                
        """
                Function to get the data and parameters for the wavelength plate stored in dictionary format
                INPUT: 
                        fluxes: array of flux values; size N x Lambda 
                        fluxes_err: array of flux errors; size N x Lambda
                        betas: latent parameters for the wavelengths, size Lambda x P
                        zetas: latent parameters for the stars, size N x P
                OUTPUT:
                        params: betas
                        data: fluxes, fluxes_err, zetas
        """
             
        params = {'betas': betas}
        data = {'fluxes': fluxes, 'fluxes_err': fluxes_err, 'zetas': zetas}

        return params, data

def get_data_zeta_step(zetas, alphas, betas, labels, labels_err, fluxes, fluxes_err):
                
        """
                Function to get the data and parameters for the star plate stored in dictionary format
                INPUT
                        alphas: latent parameters for the labels, size M x P
                        betas: latent parameters for the wavelengths, size Lambda x P
                        zetas: latent parameters for the stars, size N x P
                        labels: array of labels; size N x M 
                        labels_err: array of label errors; size N x M
                        fluxes: array of flux values; size N x Lambda 
                        fluxes_err: array of flux errors; size N x Lambda
                OUTPUT
                        params: zetas
                        data: alphas, betas, labels, labels_err, fluxes, fluxes_err
        """
             
        params = {'zetas': zetas}
        data = {'alphas': alphas, 'betas': betas,\
                'labels': labels, 'labels_err': labels_err,\
                'fluxes': fluxes, 'fluxes_err': fluxes_err}

        return params, data


############################################################
# FUNCTIONS TO ESTIMATE THE LATENT PARAMETERS
############################################################

################# FOR ALPHAS

def synthetise_one_label_one_star(zeta_n, alpha_m):
        """
                Function to use in the loss function for optimising one label and one star. For now, this is treated as linear model, but
                could in principle be replaced with any functional form (e.g., quadratic, CNN, GP)

                INPUT: 
                        zeta_n: latent parameter zeta for nth star
                        alpha_m: latent parameter alphas for mth label

                OUTPUT:
                        dot product of zeta_n and alpha_m
        """

        return zeta_n @ alpha_m

# vmaps to loop over either labels or stars using the synthetise_one_label_one_star function
#  NOTE: CAUTION with this, if using more complex synthesizer, then vmap and optimisation routine may have issues
synthetise_one_label_all_stars =  jax.vmap(synthetise_one_label_one_star, in_axes = (0, None)) # all stars
synthetise_all_labels_one_star =  jax.vmap(synthetise_one_label_one_star, in_axes = (None, 0)) # all labels

def one_label_all_stars_chi(params, data):
        """
                Chi value for one alpha and all stars.
                INPUT:
                        params: latent parameters obtained from ``get_data_alpha_step''
                        data: label data obtained from ``get_data_alpha_step''
                OUTPUT:
                        Chi for one alpha parameter for all stars. Sum of this gives the total chi summed over all stars
        """
        return (data['labels'] - synthetise_one_label_all_stars(data['zetas'], params['alphas'])) / data['labels_err']
                
def one_label_alpha_step(alphas_m, zetas, labels_m, labels_err_m):

        """
                Function used inside the run of alpha_step, where we fit for one alpha (i.e., one label) for all stars
                INPUT:  
                        labels_m: array of a label for all stars; size N x 1
                        labels_err_m: array of a label error for all stars; size N x 1
                        alphas_m: latent parameters for a given label, size M x 1
                        zetas: latent parameters for the stars, size N x P
                OUTPUT:
                        optimised label parameters: alpha_m          
        """
        params, data = get_data_alpha_step(alphas_m, zetas, labels_m, labels_err_m)
                
        optimizer = jaxopt.GaussNewton(residual_fun=one_label_all_stars_chi, maxiter=30) # Maxiter is magic number; tried varying this value and didn't change much
        
        res = optimizer.run(init_params = params, data = data) 
        return res

# vmap to loop over all labels using the one_label_alpha_step function
alpha_step = jax.vmap(one_label_alpha_step, in_axes=(0, None, 1, 1)) # over all labels

################# FOR BETAS
def synthetise_one_wavelength_one_star(zeta_i, beta_h):
        """
                Function to use in the loss function for optimising one wavelength and one star. For now, this is treated as linear model

                INPUT: 
                        zeta_i: latent parameter zeta for ith star
                        beta_h: latent parameter beta for hth wavelength

                OUTPUT:
                        dot product of zeta_i and beta_h
        """

        return zeta_i @ beta_h
    
# vmaps to loop over either wavelengths or stars using the synthetise_one_wavelength_one_star function
synthetise_one_wavelength_all_stars =  jax.vmap(synthetise_one_wavelength_one_star, in_axes = (0, None)) # all stars
synthetise_all_wavelengths_one_star =  jax.vmap(synthetise_one_wavelength_one_star, in_axes = (None, 0)) # all wavelengths

def one_wavelength_all_stars_chi(params, data):
        """
                Chi for one beta and all stars

                INPUT:  
                        params: latent parameters obtained from ``get_data_beta_step'': betas, zetas
                        data: flux data obtained from ``get_data_beta_step''
                OUTPUT:
                        Log-likelihood (chi2) for one beta parameter summed up over all stars
        """
        return (data['fluxes'] - synthetise_one_wavelength_all_stars(data['zetas'], params['betas'])) / data['fluxes_err']

def one_wavelength_beta_step(betas_lambda, zetas, fluxes_lambda, fluxes_err_lambda):

        """
                Function used inside the run of beta_step, where we fit for one beta (i.e., one wavelength) for all stars
                INPUT:  
                        fluxes_lambda: array of one flux for all stars; size N x 1
                        fluxes_err_lambda: array of one flux error for all stars; size N x 1
                        betas: latent parameters for a wavelength, size Lambda x 1
                        zetas: latent parameters for the stars, size N x P
                OUTPUT:
                        optimised label parameters: betas
        """

        params, data = get_data_beta_step(betas_lambda, zetas, fluxes_lambda, fluxes_err_lambda)

        optimizer = jaxopt.GaussNewton(residual_fun=one_wavelength_all_stars_chi, maxiter=30) # Magic number

        res = optimizer.run(init_params = params, data = data) 

        return res

# vmap to loop over all wavelengths using the one_wavelength_beta_step function
beta_step = jax.vmap(one_wavelength_beta_step, in_axes=(0, None, 1, 1)) # over all wavelengths

################# FOR ZETAS

def one_star_all_wavelengths_all_labels_chi(params, data):
        """
                Chi for one star (one zeta) and all labels and wavelengths 

                INPUT:  
                        params: latent parameters obtained from ``get_data_zeta_step'': zetas
                        data: label and flux data obtained from ``get_data_zeta_step'', with alphas and betas
                OUTPUT:
                        Log-likelihood (chi2) for one zeta parameter summed up over all wavelengths and labels
        """
        chi_labels = (data['labels'] - synthetise_all_labels_one_star(params['zetas'], data['alphas'])) /data['labels_err']
        chi_wavelengths = (data['fluxes'] - synthetise_all_wavelengths_one_star(params['zetas'], data['betas']))/ data['fluxes_err']
        return jnp.concatenate((chi_labels, chi_wavelengths))

def one_star_zeta_step(zetas_n, alphas, betas, labels, labels_err, fluxes, fluxes_err):

        """
                Function used inside the run of zeta_step to optimise all the labels and all wavelengths for one star
                INPUT:  
                        labels: array of labels; size N x M
                        labels_err: array of label errors; size N x M
                        fluxes: array of flux; size N x Lambda
                        fluxes_err: array of flux errors; size N x Lambda
                        alphas: latent parameters for the labels, size M x P
                        betas: latent parameters for the wavelengths, size Lambda x P
                        zetas_n: latent parameters for a star, size 1 x P
                OUTPUT:
                        optimised label parameters: zetas
        """

        params, data = get_data_zeta_step(zetas_n, alphas, betas, labels, labels_err, fluxes, fluxes_err)

        optimizer = jaxopt.GaussNewton(residual_fun=one_star_all_wavelengths_all_labels_chi, maxiter=30) # Magic number

        res = optimizer.run(init_params = params, data = data) 

        return res        

# vmap to loop over all stars using the one_star_zeta_step function
zeta_step = jax.vmap(one_star_zeta_step, in_axes=(0, None, None, 0, 0, 0, 0))

# #######################################################################
# DIRECT (CLOSED-FORM) BLOCK SOLVERS
# The model is linear in each of the alphas, betas, and zetas blocks, so
# every block update is a weighted linear least-squares problem with an
# exact solution. The Gauss-Newton solvers above reach the same optimum
# (Gauss-Newton is exact for linear residuals) but materialise the full
# Jacobian of every vmapped sub-problem at once -- e.g. Lambda x N x P for
# the beta step -- which exhausts device memory for survey-sized data.
# NaNs in the data are given zero weight, matching the nansum semantics
# of the chi2 functions below.
# #######################################################################

def _masked(values, weights):
        """Zero out the weight (and value) of non-finite data entries."""
        mask = jnp.isfinite(values)
        return jnp.where(mask, values, 0.), jnp.where(mask, weights, 0.)

def _anchored_solve(G, rhs, params_init):
        """
                Solve G x = rhs per batch element with a negligible ridge towards
                params_init: exact where the data constrain x, and x = params_init
                along directions of zero curvature (mirroring Gauss-Newton, which
                leaves such directions at their initial values)
        """
        P = G.shape[-1]
        # the ridge must sit far below the smallest genuine eigenvalue of G,
        # not just below its trace (which the largest eigenvalue dominates
        # when G is ill-conditioned, e.g. via the continuum direction)
        eps = 1e-12 * jnp.trace(G, axis1=-2, axis2=-1)[:, None] / P + 1e-300
        G = G + eps[..., None] * jnp.eye(P)
        rhs = rhs + eps * params_init
        return jnp.linalg.solve(G, rhs[..., None])[..., 0]

def alpha_step_direct(alphas, zetas, labels, labels_err):
        """
                Exact alphas at fixed zetas: for each label m, solve the weighted
                normal equations (Z^T W_m Z) alpha_m = Z^T W_m l_m
                INPUT:
                        alphas: latent parameters for the labels (anchor for
                                unconstrained directions), M x P
                        zetas: latent parameters for the stars, N x P
                        labels: array of labels; size N x M
                        labels_err: array of label errors; size N x M
                OUTPUT:
                        optimised alphas, M x P
        """
        l, w = _masked(labels, 1. / labels_err**2)
        zz = zetas[:, :, None] * zetas[:, None, :]              # N x P x P
        G = jnp.einsum('npq,nm->mpq', zz, w)                    # M x P x P
        rhs = (w * l).T @ zetas                                 # M x P
        return _anchored_solve(G, rhs, alphas)

def beta_step_direct(betas, zetas, fluxes, fluxes_err):
        """
                Exact betas at fixed zetas: for each wavelength h, solve the
                weighted normal equations (Z^T W_h Z) beta_h = Z^T W_h f_h
                INPUT:
                        betas: latent parameters for the wavelengths (anchor for
                               unconstrained directions), Lambda x P
                        zetas: latent parameters for the stars, N x P
                        fluxes: array of flux values; size N x Lambda
                        fluxes_err: array of flux errors; size N x Lambda
                OUTPUT:
                        optimised betas, Lambda x P
        """
        f, w = _masked(fluxes, 1. / fluxes_err**2)
        zz = zetas[:, :, None] * zetas[:, None, :]              # N x P x P
        G = jnp.einsum('npq,nh->hpq', zz, w)                    # Lambda x P x P
        rhs = (w * f).T @ zetas                                 # Lambda x P
        return _anchored_solve(G, rhs, betas)

def zeta_step_direct(zetas, alphas, betas, labels, labels_err, fluxes, fluxes_err, omega=1.):
        """
                Exact zetas at fixed alphas and betas: for each star n the joint
                label-and-flux objective is quadratic, so solve the weighted normal
                equations (omega A^T W_l A + B^T W_f B) zeta_n = rhs_n
                INPUT:
                        zetas: latent parameters for the stars (anchor for
                               unconstrained directions), N x P
                        alphas: latent parameters for the labels, M x P
                        betas: latent parameters for the wavelengths, Lambda x P
                        labels: array of labels; size N x M
                        labels_err: array of label errors; size N x M
                        fluxes: array of flux values; size N x Lambda
                        fluxes_err: array of flux errors; size N x Lambda
                        omega: weight of the label term in the joint likelihood
                OUTPUT:
                        optimised zetas, N x P
        """
        l, wl = _masked(labels, 1. / labels_err**2)
        f, wf = _masked(fluxes, 1. / fluxes_err**2)
        aa = alphas[:, :, None] * alphas[:, None, :]            # M x P x P
        bb = betas[:, :, None] * betas[:, None, :]              # Lambda x P x P
        G = omega * jnp.einsum('mpq,nm->npq', aa, wl) \
            + jnp.einsum('hpq,nh->npq', bb, wf)                 # N x P x P
        rhs = omega * (wl * l) @ alphas + (wf * f) @ betas      # N x P
        return _anchored_solve(G, rhs, zetas)

# #######################################################################
# FUNCTIONS TO RUN THE OPTIMISATION AND CALCULATE CHI2
# #######################################################################

def all_labels_all_stars_chi2(alphas, zetas, labels, labels_err):
        """
                INPUT:
                        alphas: latent parameters for the labels
                        zetas: latent parameters 
                        labels: stellar labels
                        labels_err: stellar label errors
                OUTPUT:
                        Chi value for all labels and all stars.
        """
        return jnp.nansum((labels - zetas @ alphas.T)**2 / labels_err**2)

def all_wavelengths_all_stars_chi2(betas, zetas, fluxes, fluxes_err):
        """
                INPUT:
                        betas: latent parameters for the wavelengths
                        zetas: latent parameters 
                        fluxes: stellar fluxes
                        fluxes_err: stellar fluxes errors
                OUTPUT:
                        Chi value for all wavelengths and all stars.
        """
        return jnp.nansum((fluxes - zetas @ betas.T)**2 / fluxes_err**2)

def all_wavelengths_all_labels_all_stars_chi2(alphas, betas, zetas, labels, labels_err, fluxes, fluxes_err, omega=1.):
        """
                Chi value for all wavelengths, all labels, and all stars
                INPUT:
                        alphas: latent parameters for the labels
                        betas: latent parameters for the wavelengths
                        zetas: latent parameters 
                        labels: stellar labels
                        labels_err: stellar label errors
                        fluxes: stellar fluxes
                        fluxes_err: stellar fluxes errors
                OUTPUT:
                        Chi value for all wavelengths, all labels, and all stars
        """
        chi2_labels = all_labels_all_stars_chi2(alphas, zetas, labels, labels_err)
        chi2_wavelength = all_wavelengths_all_stars_chi2(betas, zetas, fluxes, fluxes_err)

        return omega * chi2_labels + chi2_wavelength

def run_agenda(alphas, betas, zetas, labels, label_err, fluxes, fluxes_err, omega):

        """
            Agenda function to run the alpha_step, beta_step, and zeta_step

            INPUT:  
                alphas: latent parameters for labels, M x P
                betas: latent parameters for wavelengths, Lambda x P
                zetas: latent parameters, N x P
                labels: labels for all stars, N x Lambda
                label_err: labels errors for all stars, N x Lambda
                fluxes: fluxes for all stars, N x Lambda
                fluxes_err: flux errors for all stars, N x Lambda
                ln_noise_fluxes: logarithmic scatters in the pixels, Lambda
                omega: dummy variable set to 1, but initially included to allow the labels to weigh more in the likelihood
            OUTPUT:
                optimised alphas, betas, zetas, difference in chi2 between initial and current chi2, and chi2 at each step
        """

        # run the alpha step (exact, at the current zetas)
        alphas_updated = alpha_step_direct(alphas, zetas, labels, label_err)

        # run the beta step (exact, at the current zetas)
        betas_updated = beta_step_direct(betas, zetas, fluxes, fluxes_err)

        # run the zeta step (exact, at the updated alphas and betas; omega
        # weighs the label term, matching the chi2 reported below)
        zetas_updated = zeta_step_direct(zetas, alphas_updated, betas_updated, labels, label_err, fluxes, fluxes_err, omega)

        #check that this step improved chi2 and throw an error if diff_chi2 is negative
        chi2_init = all_wavelengths_all_labels_all_stars_chi2(alphas_updated, betas_updated, zetas, labels, label_err, fluxes, fluxes_err, omega)
        chi2_step = all_wavelengths_all_labels_all_stars_chi2(alphas_updated, betas_updated, zetas_updated, labels, label_err, fluxes, fluxes_err, omega)

        return alphas_updated, betas_updated, zetas_updated, chi2_init - chi2_step, chi2_step

# jit the full agenda so repeated training iterations reuse the compiled program
run_agenda = jax.jit(run_agenda)



########################################################################################################################
########################################################################################################################
########################################################################################################################
# TESTING THE MODEL
# NOTE: SINCE WE HAVE MOVED AWAY FROM USING AN UN-REGULARISED VERSION OF THE MODEL, WE DON'T USE CODE BELOW. HOWEVER
# KEPT IT FOR CONVENIENCE JUST IN CASE PEOPLE WOULD LIKE TO RUN THIS QUICKLY WITHOUT REGULARISATION
########################################################################################################################
########################################################################################################################
########################################################################################################################

# NOTE: Idea is, you have now optimised for alphas, betas, and zetas (of the training set). You now need to find the 
# zetas corresponding to the test set (or new stars). Once you have those, you can estimate the labels/spectra

################### FOR ZETAS FROM SPECTRA
# NOTE: reuses synthetise_one_wavelength_one_star and its vmaps defined above

def all_wavelengths_one_star_chi(params, data):
        """
                Chi for one zeta and all wavelength
                INPUT:  
                        params: zeta_n
                        data: flux data and optimised betas
                OUTPUT:
                        Chi for one beta parameter summed up over all stars
        """
        return (data['fluxes'] - synthetise_all_wavelengths_one_star(params['zetas'], data['betas'])) / data['fluxes_err']

def find_one_zeta_test_step(zetas_init, betas, fluxes, fluxes_err):

        """
                Function used inside the run of alpha_step, where we fit for one alpha (i.e., one label) for all stars
                INPUT:  
                        fluxes: array of flux; size Lambda
                        fluxes_err: array of flux errors; size Lambda
                        betas: latent parameters for the wavelengths, size Lambda x K
                        zetas: latent parameters for the stars, size N x K
                OUTPUT:
                        optimised label parameters: zetas
        """

        params = {'zetas': zetas_init}
        data = {'betas': betas, 'fluxes': fluxes, 'fluxes_err': fluxes_err}

        # run the least squares optimisation using the Gauss-Newton solver
        optimizer = jaxopt.GaussNewton(residual_fun=all_wavelengths_one_star_chi, maxiter=30) # magic 30

        res = optimizer.run(init_params = params, data = data)

        return res

zetas_test_step = jax.vmap(find_one_zeta_test_step, in_axes=(0, None, 0, 0))

################### FOR ZETAS FROM LABELS
# NOTE: reuses synthetise_one_label_one_star and its vmaps defined above

def all_labels_one_star_chi(params, data):
        """
                Chi for one zeta and all wavelength
                INPUT:  
                        params: zeta_n
                        data: flux data and optimised betas
                OUTPUT:
                        Chi for one beta parameter summed up over all stars
        """
        return (data['labels'] - synthetise_all_labels_one_star(params['zetas'], data['alphas'])) / data['labels_err']

def find_one_zeta_test_step_labels(zetas_init, alphas, labels, labels_err):

        """
                Function used inside the run of alpha_step, where we fit for one alpha (i.e., one label) for all stars
                INPUT:  
                        labels: array of labels; size M
                        fluxes_err: array of label errors; size M
                        alphas: latent parameters for the labels, size M x K
                        zetas: latent parameters for the stars, size N x K
                OUTPUT:
                        optimised label parameters: zetas
        """

        params = {'zetas': jnp.zeros((zetas_init.shape))}
        data = {'alphas': alphas, 'labels': labels, 'labels_err': labels_err}

        # run the least squares optimisation using the Gauss-Newton solver
        optimizer = jaxopt.GaussNewton(residual_fun=all_labels_one_star_chi, maxiter=30) # magic 30

        res = optimizer.run(init_params = params, data = data)

        return res

zetas_test_step_labels = jax.vmap(find_one_zeta_test_step_labels, in_axes=(0, None, 0, 0))

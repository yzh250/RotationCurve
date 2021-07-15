################################################################################
# All the libraries used & constant values
# -------------------------------------------------------------------------------
import numpy as np

import matplotlib.pyplot as plt

from astropy.table import QTable

from scipy.optimize import minimize

from rotation_curve_functions import nloglike_Bur, v_co_Burket, v_co_Burket_nb,v_d, vel_h_Burket, vel_b, loglike_Bur_nb,nloglike_Bur_nb, RC_fitting_Bur, RC_plotting_Bur

import dynesty
from dynesty import plotting as dyplot

################################################################################

################################################################################
# data from Di Paolo 2019
r = [0.202169625, 0.601577909, 1.000986193, 1.400394477, 1.804733728, 2.24852071, 2.75147929, 3.264299803, 3.75739645, 4.26035503, 4.758382643, 5.25147929]
r_denorm = 2.2*np.array(r)
print(r_denorm)
v = [24.89465154, 39.93517018, 51.99351702, 56.66126418, 62.36628849, 64.95948136, 70.79416532, 74.42463533, 76.23987034, 78.57374392, 81.29659643, 81.68557536]
v_err = [2,1,3,1,1,2,2,1,3,1,2,1]
################################################################################

################################################################################
# Fitting and Plotting
v_fit = RC_fitting_Bur(r_denorm,3.8E9,v,v_err,90)
RC_plotting_Bur(r_denorm,v,v_err,v_fit,'Di Paolo 2019')
#################################################################################

#################################################################################
# Corner Plot Functions ()
def uniform(a, b, u):
    """Given u in [0,1], return a uniform number in [a,b]."""
    return a + (b-a)*u
def jeffreys(a, b, u):
    #"""Given u in [0,1], return a Jeffreys random number in [a,b]."""
    return a**(1-u) * b**u
def prior_xforBB(u):
    """
    Priors for the 3 parameters of the BB velocity curve model.
    Required by the dynesty sampler.
    Parameters
    ----------
    u : ndarray
        Array of uniform random numbers between 0 and 1.
    Returns
    -------
    priors : ndarray
        Transformed random numbers giving prior ranges on model parameters.
    """
    M_disk = uniform(8,12,u[0])
    R_disk = uniform(0.1, 20,u[1])
    rho_halo = uniform(0.0001,1,u[2])
    R_halo = uniform(0.1,100,u[3])
    return M_disk, R_disk, rho_halo, R_halo

dsampler = dynesty.DynamicNestedSampler(loglike_Bur_nb, prior_xforBB, ndim=4,
                                        logl_args=(np.array(r_denorm),
                                                   np.array(v),
                                                   np.array(v_err),
                                                   90),
                                        nlive=2000,
                                        bound='multi',
                                        sample='auto')
dsampler.run_nested()
dres1 = dsampler.results

labels = ['$M_{disk}$', '$R_{disk}$', '$Rho_{halo}$','$R_{halo}$']
nParams = len(labels)
fig, axes = dyplot.cornerplot(dres1, smooth=0.03,
                              labels=labels,
                              show_titles=True,
                              quantiles_2d=[1-np.exp(-0.5*r**2) for r in [1.,2.,3]],
                              quantiles=[0.16, 0.5, 0.84],
                              fig=plt.subplots(nParams, nParams, figsize=(2.5*nParams, 2.6*nParams)),
                              color='#1f77d4')
plt.show()
#################################################################################


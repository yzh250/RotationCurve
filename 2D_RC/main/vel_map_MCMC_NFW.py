####################################################################
import matplotlib.pyplot as plt

import numpy as np
import numpy.ma as ma

from astropy.io import fits
from astropy.table import QTable

from scipy.optimize import minimize

import numdifftools as ndt
from numpy import log as ln
from scipy.special import kn
from scipy.special import iv

from scipy import integrate as inte
import emcee
import corner

import pickle

from galaxy_component_functions import vel_tot_iso,\
                                       vel_tot_NFW,\
                                       vel_tot_bur


from Velocity_Map_Functions import loglikelihood_iso,\
                                   loglikelihood_NFW, \
                                   loglikelihood_bur,\
                                   loglikelihood_iso_flat,\
                                   loglikelihood_NFW_flat, \
                                   loglikelihood_bur_flat,\
                                   find_phi

from RC_2D_Fit_Functions import Galaxy_Data    

G = 6.674E-11  # m^3 kg^-1 s^-2
Msun = 1.989E30  # kg
scale = 0.46886408261217366                                                                    
####################################################################

####################################################################
# 7443-12705
r_band, Ha_vel, Ha_vel_ivar, Ha_vel_mask, Ha_flux, Ha_flux_ivar, Ha_flux_mask, vmasked, Ha_flux_masked, ivar_masked, gshape, x_center_guess, y_center_guess = Galaxy_Data('7443-12705','bluehive')
####################################################################

####################################################################
# loglikelihood
def log_prior(params):
    rho_b,Rb,SigD,Rd,rho_h,Rh,inclination,phi,center_x,center_y,vsys= params
    logP = 0
    if 0 < rho_b < 100 and 0 < Rb < 5 and 100 < SigD < 3000 and 1 < Rd < 30\
     and 1e-5 < rho_h < 0.1 and 0.01 < Rh< 500 and 0 < inclination < 0.436*np.pi and 0 < phi < 2*np.pi\
     and 20 < center_x < 40 and 20 < center_y < 40 and -100 < vsys < 100:
        logP = 0
    # setting constraints on the radii
    elif Rh < Rb or Rh < Rd or Rd < Rd:
        logP = -np.inf
    else:
        logP = -np.inf
    return logP

def log_prob_iso(params, scale, shape, vdata, ivar, mask):
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + loglikelihood_iso_flat(params, scale, shape, vdata.compressed(), ivar.compressed(), mask)

def log_prob_NFW(params, scale, shape, vdata, ivar, mask):
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + loglikelihood_NFW_flat(params, scale, shape, vdata.compressed(), ivar.compressed(), mask)

def log_prob_bur(params, scale, shape, vdata, ivar, mask):
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + loglikelihood_bur_flat(params, scale, shape, vdata.compressed(), ivar.compressed(), mask)
####################################################################

####################################################################
# NFW

pos = np.random.uniform(low=[0,1e-4,300,2,0.0001,0.1,0,0.01,30,30,-20], high=[50,5,2000,20,0.01,300,np.pi/2,2*np.pi,40,40,20], size=(64,11))
nwalkers, ndim = pos.shape

bad_sampler_NFW = emcee.EnsembleSampler(nwalkers, ndim, log_prob_NFW, args=(scale, gshape, vmasked, ivar_masked, Ha_vel_mask))
bad_sampler_NFW.run_mcmc(pos, 10000, progress=True)

good_walkers_NFW = bad_sampler_NFW.acceptance_fraction > 0

fig_NFW, axes_NFW = plt.subplots(11,1, figsize=(20, 14), sharex=True,
                         gridspec_kw={'hspace':0.1})
bad_samples_NFW = bad_sampler_NFW.get_chain()[:,good_walkers_NFW,:]

labels = ['rho_b','R_b', 'Sigma_d','R_d','rho_h','R_h','i','phi','x','y','vsys']

for i in range(ndim):
    ax = axes_NFW[i]
    ax.plot(bad_samples_NFW[:10000,:,i], 'k', alpha=0.3)
    ax.set(xlim=(0,10000), ylabel=labels[i])
    ax.yaxis.set_label_coords(-0.11, 0.5)

axes_NFW[-1].set_xlabel('step number')
#fig_NFW.tight_layout()
plt.savefig('mcmc_NFW.png',format='png')
plt.close()
####################################################################

####################################################################
bad_samples_NFW = bad_sampler_NFW.get_chain(discard=500)[:,good_walkers_NFW,:]
ns_NFW, nw_NFW, nd_NFW = bad_samples_NFW.shape
flat_bad_samples_NFW = bad_samples_NFW.reshape(ns_NFW*nw_NFW, nd_NFW)
flat_bad_samples_NFW.shape
####################################################################

####################################################################
corner.corner(flat_bad_samples_NFW, labels=labels,
                    range=[(0,100), (0,5), (0,2000),(1,20),(0.0001,0.01),(5,200),(0,np.pi/2),(0,1.5),(30,40),(30,40),(-100,100)], bins=30, #smooth=1,
                    truths=[0.05812451,3.601276359,385.2756031,6.748078457,0.002449669,30.24921674,1.080172553,0.69825044,36.61004742,37.67680252,11.81343922], truth_color='#ff4444',
                    levels=(1-np.exp(-0.5), 1-np.exp(-2)), quantiles=(0.16, 0.84),
                    hist_kwargs={'histtype':'stepfilled', 'alpha':0.3, 'density':True},
                    color='blue', plot_datapoints=False,
                    fill_contours=True)
plt.savefig('corner_NFW.png',format='png')
plt.close()
####################################################################

soln = [0.05812451,3.601276359,385.2756031,6.748078457,0.002449669,30.24921674,1.080172553,0.69825044,36.61004742,37.67680252,11.81343922]

for i, label in enumerate(labels):
    x = soln.x[i]
    x16, x84 = np.percentile(flat_bad_samples_iso[:,i], [16,84])
    dlo = x - x16
    dhi = x84 - x
    print('{:3s} = {:5.2f} + {:4.2f} - {:4.2f}'.format(label, x, dhi, dlo))
    print('    = ({:5.2f}, {:5.2f})'.format(x16, x84))

####################################################################
# Dumping out put
#out_directory = '/Users/richardzhang/Documents/UR_Stuff/Research_UR/2D_RC/MCMC_folder/'
#temp_outfile = open(out_directory + 'results.pickle', 'wb')
#pickle.dump((flat_bad_samples_iso, flat_bad_samples_NFW, flat_bad_samples_bur), temp_outfile)
#temp_outfile.close()




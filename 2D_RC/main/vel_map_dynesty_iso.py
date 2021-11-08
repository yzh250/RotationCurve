####################################################################
import matplotlib.pyplot as plt

import numpy as np
import numpy.ma as ma

from astropy.io import fits
from astropy.table import QTable

from scipy.optimize import minimize

#import numdifftools as ndt
from numpy import log as ln
from scipy.special import kn
from scipy.special import iv

from scipy import integrate as inte

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

import dynesty
from dynesty import plotting as dyplot  

G = 6.674E-11  # m^3 kg^-1 s^-2
Msun = 1.989E30  # kg
scale = 0.46886408261217366   
####################################################################

# 7443-12705
r_band, Ha_vel, Ha_vel_ivar, Ha_vel_mask, Ha_flux, Ha_flux_ivar, Ha_flux_mask, vmasked, Ha_flux_masked, ivar_masked, gshape, x_center_guess, y_center_guess = Galaxy_Data('7443-12705')

minimize_best_fit = [0.048688757,2.549862293,748.5940907,5.617303041,0.002927534,0.100051148,1.070928683,0.699892835,36.61461409,37.68004929,11.37083843]

def uniform(a,b,u):
	return a + (b - a)*u

def prior_isothermal(u):
	log_rhob0 = uniform(-7,2,u[0])
	Rb = uniform(0,5,u[1])
	SigD = uniform(100,3000,u[2])
	Rd = uniform(1,30,u[3])
	rho_h = uniform(1e-5,0.1,u[4])
	Rh = uniform(0.01,500,u[5])
	incl = uniform(0,0.436*np.pi,u[6])
	phi = uniform(0,2*np.pi,u[7])
	cen_x = uniform(20,40,u[8])
	cen_y = uniform(20,40,u[9])
	vsys = uniform(-100,100,u[10])
	return log_rhob0, Rb, SigD, Rd, rho_h, Rh, incl, phi, cen_x, cen_y, vsys

labels = ['log_rhob0','R_b', 'Sigma_d','R_d','rho_h','R_h','i','phi','x','y','vsys']

dsampler = dynesty.DynamicNestedSampler(loglikelihood_iso_flat, prior_isothermal, 
												 logl_args = (scale,gshape,vmasked.compressed(),ivar_masked.compressed(),Ha_vel_mask),
												 ndim = 11,
										 		 nlive = 2000,
										 		 bound = 'multi',
										 		 sample = 'auto')

dsampler.run_nested()

d_result = dsampler.results

truths = [0.048688757,2.549862293,748.5940907,5.617303041,0.002927534,0.100051148,1.070928683,0.699892835,36.61461409,37.68004929,11.37083843]

fig, axes = dyplot.cornerplot(d_result, 
							  truths=truths,
							  show_titles=True, 
                              title_kwargs={'y': 1.04}, labels=labels,
                              quantiles_2d=[1-np.exp(-0.5*r**2) for r in [1.,2.,3.]],
                              quantiles=(0.16, 0.84),
                              fig=plt.subplots(3, 3, figsize=(9,10)),
                              color='#1f77d4')

fig.savefig('dynesty_corner_iso.png',format='png')




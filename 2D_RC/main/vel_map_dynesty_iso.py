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


from Velocity_Map_Functions import loglikelihood_iso_flat

from RC_2D_Fit_Functions import Galaxy_Data  

import dynesty
from dynesty import plotting as dyplot  

G = 6.674E-11  # m^3 kg^-1 s^-2
Msun = 1.989E30  # kg
scale = 0.136270089 
####################################################################

# 7495-12704
gal_ID = '7495-12704'
data_maps, gshape, x_center_guess, y_center_guess = Galaxy_Data(gal_ID, 
                                                                manga)

initial_guesses = [-1, 1, 1000, 4, -3, 25, 0.8725795257390155, 6.05728734209396, 38, 16, 0]

model_guesses = [-1, 1, 1000, 4, -3, 25]

geo_guesses =  [0.8725795257390155, 6.05728734209396, 38, 16, 0]

def uniform(a,b,u):
	return a + (b - a)*u

def prior_isothermal(u):
	log_rhob0 = uniform(-7,2,u[0])
	Rb = uniform(0.00001,5,u[1])
	SigD = uniform(200,3000,u[2])
	Rd = niform(0.1,25,u[3])
	rho_h = uniform(2e-5,0.1,u[4])
	Rh = uniform(0.1,500,u[5])
	incl = geo_guesses[0] + uniform(-1e-3,1e-3,u[6])
	phi = geo_guesses[1] + uniform(-1e-3,1e-3,u[7])
	cen_x = geo_guesses[2] + uniform(-1e-3,1e-3,u[8])
	cen_y = geo_guesses[3] + uniform(-1e-3,1e-3,u[9])
	vsys = geo_guesses[4] + uniform(-1e-3,1e-3,u[10])
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

truths = [0.879029737,4.998864286,2058.505154,19.85417227,1.062853006,0.484093619,0.584995383,2.077845538,43.68893541,6.00004005,99.99993356]

fig, axes = dyplot.cornerplot(d_result, 
							  truths=truths,
							  show_titles=True, 
                              title_kwargs={'y': 1.04}, labels=labels,
                              quantiles_2d=[1-np.exp(-0.5*r**2) for r in [1.,2.,3.]],
                              quantiles=(0.16, 0.84),
                              fig=plt.subplots(3, 3, figsize=(9,10)),
                              color='#1f77d4')

fig.savefig('dynesty_corner_iso.png',format='png')




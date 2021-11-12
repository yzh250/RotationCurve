################################################################################
'''
import matplotlib.pyplot as plt
'''
import numpy as np
'''
import numpy.ma as ma

from astropy.io import fits
from astropy.table import QTable

from scipy.optimize import minimize

import numdifftools as ndt
from numpy import log as ln
from scipy.special import kn, k0, k1
from scipy.special import iv, i0, i1

from scipy import integrate as inte
import emcee
import corner

import pickle
'''

from galaxy_component_functions import bulge_vel,\
                                       disk_vel,\
                                       halo_vel_NFW,\
                                       vel_tot_iso,\
                                       vel_tot_NFW,\
                                       vel_tot_bur

from Velocity_Map_Functions import loglikelihood_iso,\
                                   loglikelihood_NFW, \
                                   loglikelihood_bur,\
                                   loglikelihood_iso_flat,\
                                   loglikelihood_NFW_flat, \
                                   loglikelihood_bur_flat,\
                                   find_phi,\
                                   rot_incl_NFW,\
                                   rot_incl_iso,\
                                   rot_incl_bur

from RC_2D_Fit_Functions import Galaxy_Data    


#from galaxy_component_functions_cython import bulge_vel, disk_vel, halo_vel_NFW
#from galaxy_component_functions_cython import vel_tot_NFW
#from Velocity_Map_Functions_cython import rot_incl_NFW

G = 6.674E-11  # m^3 kg^-1 s^-2
Msun = 1.989E30  # kg
scale = 0.46886408261217366                                                                    
################################################################################



################################################################################
# 7443-12705
#-------------------------------------------------------------------------------
MANGA_FOLDER = '/Users/richardzhang/Documents/UR_Stuff/Research_UR/SDSS/dr16/manga/spectro/'
#MANGA_FOLDER = '/Users/kellydouglass/Documents/Research/data/SDSS/dr16/manga/spectro/'

maps, gshape, x_center_guess, y_center_guess = Galaxy_Data('7443-12705', 
                                                           MANGA_FOLDER)
################################################################################

max_likelihood_params = [np.log10(0.05812451), 
                         3.601276359, 
                         385.2756031, 
                         6.748078457, 
                         0.002449669, 
                         30.24921674, 
                         1.080172553, 
                         0.69825044, 
                         36.61004742, 
                         37.67680252, 
                         11.81343922]

"""
@profile
def vel_tot_iso_profile(r, params):
    vel_tot_iso(r, params[:6])

vel_tot_iso_profile(13.5, max_likelihood_params)
"""

"""
@profile
def vel_tot_NFW_profile(r, params):
    vel_tot_NFW(r, params[:6])

vel_tot_NFW_profile(13.5, max_likelihood_params)
"""

"""
@profile
def vel_tot_bur_profile(r, params):
    vel_tot_bur(r, params[:6])

vel_tot_bur_profile(13.5, max_likelihood_params)
"""

"""
@profile
def rot_incl_iso_profile(shape, scale, params):
    rot_incl_iso(shape, scale, params)

rot_incl_iso_profile(gshape, scale, max_likelihood_params)
"""

"""
@profile
def rot_incl_NFW_profile(shape, scale, params):
    rot_incl_NFW(shape, scale, params)

rot_incl_NFW_profile(gshape, scale, max_likelihood_params)
"""

"""
@profile
def rot_incl_bur_profile(shape, scale, params):
    rot_incl_bur(shape, scale, params)

rot_incl_bur_profile(gshape, scale, max_likelihood_params)
"""

'''
@profile
def disk_vel_profile():
    disk_vel(13.5, 385.2756031, 6.748078457)

disk_vel_profile()
'''
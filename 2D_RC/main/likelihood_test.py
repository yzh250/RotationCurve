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
                                       vel_tot_bur,\
                                       bulge_vel,\
                                       disk_vel,\
                                       halo_vel_NFW


from Velocity_Map_Functions import loglikelihood_iso,\
                                   loglikelihood_NFW, \
                                   loglikelihood_bur,\
                                   loglikelihood_iso_flat,\
                                   loglikelihood_NFW_flat, \
                                   loglikelihood_bur_flat,\
                                   find_phi,\
                                   rot_incl_NFW

from RC_2D_Fit_Functions import Galaxy_Data    

G = 6.674E-11  # m^3 kg^-1 s^-2
Msun = 1.989E30  # kg
scale = 0.46886408261217366                                                                    
####################################################################

####################################################################
# 7443-12705
r_band, Ha_vel, Ha_vel_ivar, Ha_vel_mask, Ha_flux, Ha_flux_ivar, Ha_flux_mask, vmasked, Ha_flux_masked, ivar_masked, gshape, x_center_guess, y_center_guess = Galaxy_Data('7443-12705')#,'bluehive')
####################################################################

max_likelihood_params = [np.log10(0.05812451),3.601276359,385.2756031,6.748078457,0.002449669,30.24921674,1.080172553,0.69825044,36.61004742,37.67680252,11.81343922]

'''
def rot_incl_NFW(shape,scale,params):

    log_rhob0, Rb, SigD, Rd, rho0_h, Rh, inclination, phi, center_x, center_y, vsys = params
    rotated_inclined_map = np.zeros(shape)
    
    for i in range(shape[0]):
        for j in range(shape[1]):
            x =  ((j-center_x)*np.cos(phi) + np.sin(phi)*(i-center_y))/np.cos(inclination)
            y =  (-(j-center_x)*np.sin(phi) + np.cos(phi)*(i-center_y))
            r = np.sqrt(x**2 + y**2)
            theta = np.arctan2(-x,y)
            r_in_kpc = r*scale
            if i == 25 and j == 30:
                print(r_in_kpc)
            v = vel_tot_NFW(r_in_kpc,[log_rhob0,Rb,SigD,Rd,rho0_h,Rh])*np.sin(inclination)*np.cos(theta)
            rotated_inclined_map[i,j] = v + vsys

    return rotated_inclined_map
'''

'''
def loglikelihood_NFW_flat(params, scale, shape, vdata_flat, ivar_flat, mask):
    # Construct the model
    #print(len(vdata_flat))
    #print(len(ivar_flat))
    model = rot_incl_NFW(shape, scale, params)
    model_masked = ma.array(model, mask=mask)
    model_flat = model_masked.compressed()
    #print(len(model_flat))
    logL = -0.5 * np.sum((vdata_flat - model_flat) ** 2 * ivar_flat - np.log(ivar_flat))
    if params[3] >= params[5]:
        logL += 1e7
    elif params[1] >= params[5]:
        logL += 1e7
    return logL
'''

'''
def vel_tot_NFW(r, params):
    log_rhob0, Rb, SigD, Rd, rho0_h, Rh = params

    r_pc = r * 1000
    Rb_pc = Rb * 1000
    Rd_pc = Rd * 1000
    Rh_pc = Rh * 1000

    Vbulge = bulge_vel(r_pc, log_rhob0, Rb_pc)
    Vdisk = disk_vel(r_pc, SigD, Rd_pc)
    Vhalo = halo_vel_NFW(r_pc, rho0_h, Rh_pc)
    v2 = Vbulge ** 2 + Vdisk ** 2 + Vhalo ** 2

    return np.sqrt(v2)
'''

@profile
def disk_vel(r, SigD, Rd):
    '''
    :param SigD: Central surface density for the disk [M_sol/pc^2]
    :param Rd: The scale radius of the disk [pc]
    :r: The distance from the centre [pc]
    :return: The rotational velocity of the disk [km/s]
    '''
    # SigD, Rd = params

    y = r / (2 * Rd)

    bessel_component = (iv(0, y) * kn(0, y) - iv(1, y) * kn(1, y))
    vel2 = (4 * np.pi * G * SigD * y ** 2 * (Rd / (3.086e16)) * Msun) * bessel_component

    return np.sqrt(vel2) / 1000

print('Likelihood: ' + str(disk_vel(13.5,385.2756031,6.748078457)))


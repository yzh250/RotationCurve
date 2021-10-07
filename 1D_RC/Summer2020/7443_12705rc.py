################################################################################
# All the libraries used & constant values
# -------------------------------------------------------------------------------
import numpy as np

import matplotlib.pyplot as plt

from astropy.table import QTable

from scipy.optimize import minimize

from rotation_curve_functions import nloglike_Bur, v_co_Burket, v_d, vel_h_Burket, vel_b

import dynesty
from dynesty import plotting as dyplot

################################################################################


################################################################################
# Reading in all the objects with Plate No. 7443
# -------------------------------------------------------------------------------
galaxy_ID = '7443-12705'

rot_curve_data_filename = galaxy_ID + '_rot_curve_data.txt'

DTable = QTable.read(rot_curve_data_filename, format='ascii.ecsv')
################################################################################


################################################################################
# Reading in the radii, velocity data
# -------------------------------------------------------------------------------
r = DTable['deprojected_distance'].data
av = DTable['rot_vel_avg'].data
mav = DTable['max_velocity'].data
miv = DTable['min_velocity'].data
av_err = DTable['rot_vel_avg_error'].data
mav_err = DTable['max_velocity_error'].data
miv_err = DTable['min_velocity_error'].data
################################################################################


################################################################################
# Plotting all the data
# -------------------------------------------------------------------------------
plt.errorbar(r, av, yerr=av_err, fmt='g.', label='Average')
plt.errorbar(r, mav, yerr=mav_err, fmt='r*', label='Maximum')
plt.errorbar(r, np.abs(miv), yerr=miv_err, fmt='b^', label='Minimum')
plt.legend()
plt.xlabel('$r_{dep}$ [kpc]')
plt.ylabel('$v_{rot}$ [km/s]')
plt.title(galaxy_ID)
plt.show()
################################################################################


################################################################################
# Fit rotation curves
# -------------------------------------------------------------------------------
# Average rotation curve
# -------------------------------------------------------------------------------
# Initial guesses
p0 = [0.2,0.1,10.77897, 4, 0.02, 20]

# Bounds
'''
param_bounds = [[8, 12],  # Disk mass [log(Msun)]
                [0.1, 100],  # Disk radius [kpc]
                [0.001, 10],  # Halo density [Msun/pc^2]
                [0.1, 10000]]  # Halo radius [kpc]
'''
param_bounds = [[0.2,1], # Scale Factor [unitless]
                [0,1000], # Bulge Scale Velocity [km/s]
                [0, 12],  # Disk mass [log(Msun)]
                [0, 10],  # Disk radius [kpc]
                [0, 1],  # Halo density [Msun/pc^2]
                [0, 100]]  # Halo radius [kpc]

bestfit_av = minimize(nloglike_Bur, p0, args=(r, av, av_err, 250),
                      bounds=param_bounds)
print('---------------------------------------------------')
print('Average curve')
print(bestfit_av)

r_normalized = r / bestfit_av.x[1]

plt.errorbar(r_normalized, av, yerr=av_err, fmt='g*', label='Average')
plt.plot(r_normalized, v_co_Burket(np.array(r), bestfit_av.x), '--', label='fit')
plt.plot(r_normalized, vel_b(np.array(r) * 1000, bestfit_av.x[0], bestfit_av.x[1], bestfit_av.x[3] * 1000), label='bulge')
plt.plot(r_normalized, v_d(np.array(r) * 1000, 10 ** bestfit_av.x[2], bestfit_av.x[3] * 1000), label='disk')
plt.plot(r_normalized, vel_h_Burket(np.array(r) * 1000, bestfit_av.x[4], bestfit_av.x[5] * 1000), label='Burket halo')
plt.legend()
plt.xlabel('$r_{dep}$/$R_d$')
plt.ylabel('$v_{rot}$ [km/s]')
plt.title(galaxy_ID)
plt.show()
# -------------------------------------------------------------------------------

# -------------------------------------------------------------------------------
# Positive rotation curve
# -------------------------------------------------------------------------------
# Initial guesses

bestfit_mav = minimize(nloglike_Bur, p0, args=(r, mav, mav_err, 250),
                       bounds=param_bounds)
print('---------------------------------------------------')
print('Positive curve')
print(bestfit_mav)

r_normalized_mav = r / bestfit_mav.x[1]

plt.errorbar(r_normalized_mav, mav, yerr=mav_err, fmt='r*', label='data')
plt.plot(r_normalized_mav, v_co_Burket(np.array(r), bestfit_mav.x), '--', label='fit')
plt.plot(r_normalized_mav, vel_b(np.array(r) * 1000, bestfit_mav.x[0], bestfit_mav.x[1], bestfit_mav.x[3] * 1000), label='bulge')
plt.plot(r_normalized_mav, v_d(np.array(r) * 1000, 10 ** (bestfit_mav.x[2]), bestfit_mav.x[3] * 1000), label='disk')
plt.plot(r_normalized_mav, vel_h_Burket(np.array(r) * 1000, bestfit_mav.x[4], bestfit_mav.x[5] * 1000), label='Burket halo')
plt.legend()
plt.xlabel('$r_{dep}$/$R_d$')
plt.ylabel('$v_{rot}$ [km/s]')
plt.title(galaxy_ID)
plt.show()
# -------------------------------------------------------------------------------

# -------------------------------------------------------------------------------
# Negative rotation curve
# -------------------------------------------------------------------------------
bestfit_miv = minimize(nloglike_Bur, p0, args=(r, np.abs(miv), miv_err, 250),
                       bounds=param_bounds)
print('---------------------------------------------------')
print('Negative curve')
print(bestfit_miv)

r_normalized_miv = r / bestfit_miv.x[1]
plt.errorbar(r_normalized_miv, np.abs(miv), yerr=miv_err, fmt='b*', label='data')
plt.plot(r_normalized_miv, v_co_Burket(np.array(r), bestfit_miv.x), '--', label='fit')
plt.plot(r_normalized_miv, vel_b(np.array(r) * 1000, bestfit_miv.x[0], bestfit_miv.x[1], bestfit_miv.x[3] * 1000), label='bulge')
plt.plot(r_normalized_miv, v_d(np.array(r) * 1000, 10 ** bestfit_miv.x[2], bestfit_miv.x[3] * 1000), label='disk')
plt.plot(r_normalized_miv, vel_h_Burket(np.array(r) * 1000, bestfit_miv.x[4], bestfit_miv.x[5] * 1000), label='Burket halo')
plt.legend()
plt.xlabel('$r_{dep}$/$R_d$')
plt.ylabel('$v_{rot}$ [km/s]')
plt.title(galaxy_ID)
plt.show()
################################################################################

################################################################################
# Random Distribution Functions
def uniform(a, b, u):
    """Given u in [0,1], return a uniform number in [a,b]."""
    return a + (b-a)*u
def jeffreys(a, b, u):
    #"""Given u in [0,1], return a Jeffreys random number in [a,b]."""
    return a**(1-u) * b**u
def ptform_Bur(u):
    """
    Priors for the 4 parameters of Burket rotation curve model. 
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
    M_disk = uniform(8,11,u[0])
    R_disk = uniform(2, 10,u[1])
    Rho_halo = uniform(5E-4,5E-2,u[2])
    R_halo = jeffreys(10,500,u[3])
    return M_disk, R_disk, Rho_halo, R_halo
##################################################################################

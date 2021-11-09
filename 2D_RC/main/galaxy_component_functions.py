################################################################################
# Pacakges
#-------------------------------------------------------------------------------
from scipy import integrate as inte
from scipy.optimize import minimize
from scipy.special import kn, iv

import numpy as np

import matplotlib.pyplot as plt

from numpy import log as ln

from astropy.table import QTable
import astropy.units as u
################################################################################




################################################################################
# Constants
#-------------------------------------------------------------------------------
G = 6.67E-11  # m^3 kg^-1 s^-2
Msun = 1.989E30  # kg
################################################################################




################################################################################
# Disk velocity from Sofue 2013
#
# Fitting for central surface density
#-------------------------------------------------------------------------------
# def disk_vel(params, r):
def disk_vel(r, SigD, Rd):
    '''
    :param SigD: Central surface density for the disk [M_sol/pc^2]
    :param Rd: The scale radius of the disk [kpc]
    :r: The distance from the centre [kpc]
    :return: The rotational velocity of the disk [km/s]
    '''
    # SigD, Rd = params

    y = r / (2 * Rd)

    bessel_component = (iv(0, y) * kn(0, y) - iv(1, y) * kn(1, y))
    vel2 = (4 * np.pi * G * SigD * y ** 2 * (Rd / (3.086E16)) * Msun) * bessel_component

    return np.sqrt(vel2) / 1000
################################################################################




################################################################################
# Circular velocity for isothermal Halo without the complicated integrals
# from eqn (51) & (52) from Sofue 2013.
# integral form can be seen from "rotation_curve_functions.py"
#-------------------------------------------------------------------------------
def halo_vel_iso(r, rho0_h, Rh):
    '''
    :param r: The a distance from the centre (pc)
    :param rho_iso: The central density of the halo (M_sol/pc^3)
    :param Rh: The scale radius of the dark matter halo (pc)
    :return: rotational velocity
    '''

    v_inf = np.sqrt((4 * np.pi * G * rho0_h * (Msun) * Rh ** 2) / (3.03E16))

    if isinstance(r,float):

        # the part in the square root would be unitless
        vel = v_inf * np.sqrt(1 - ((Rh/r)*np.arctan2(r,Rh)))

    else:

        vel = np.zeros(len(r))

        for i in range(len(r)):

            vel[i] = v_inf * np.sqrt(1 - ((Rh/r[i])*np.arctan2(r[i],Rh)))

    return vel/1000
################################################################################





################################################################################
# NFW_halo
# velocity calculated from mass with already evaluated integral
# from density profile (eqn 55, Sofue 2013)
# integral form can be seen from "rotation_curve_functions.py"
#-------------------------------------------------------------------------------
def halo_vel_NFW(r, rho0_h, Rh):

    if isinstance(r, float):

        halo_mass = 4*np.pi*rho0_h*Rh**3*((-r/(Rh+r)) + np.log(Rh + r) - np.log(Rh))

    else:

        halo_mass = np.zeros(len(r))

        for i in range(len(r)):

            halo_mass[i] = 4*np.pi*rho0_h*Rh**3*((-r[i]/(Rh+r[i])) + np.log(Rh + r[i]) - np.log(Rh))

    vel2 = G * (halo_mass * Msun) / (r * 3.08E16)

    return np.sqrt(vel2)/1000
################################################################################





################################################################################
# Burket halo
# velocity calculated from mass with already evaluated integral
# from density profile (eqn 56, Sofue 2013)
# integral form can be seen from "rotation_curve_functions.py"
#-------------------------------------------------------------------------------
def halo_vel_bur(r,rho0_h, Rh):

    if isinstance(r, float):

        halo_mass = np.pi * (-rho0_h) * (Rh**3) * (-np.log(Rh**2 + r**2) \
                                                   - 2*np.log(Rh + r)\
                                                   + 2*np.arctan2(r, Rh)\
                                                   + np.log(Rh**2)\
                                                   + 2*np.log(Rh)\
                                                   - 2*np.arctan2(0, Rh))
    else:

        halo_mass = np.zeros(len(r))

        for i in range(len(r)):

            halo_mass[i] = np.pi * (-rho0_h) * (Rh**3) * (-np.log(Rh**2\
                                                                  + r[i]**2)\
                                                          - 2*np.log(Rh + r[i])\
                                                          + 2*np.arctan2(r[i], Rh)\
                                                          + np.log(Rh**2)\
                                                          + 2*np.log(Rh)\
                                                          - 2*np.arctan2(0, Rh))
    vel2 = G * (halo_mass * Msun) / (r * 3.08E16)

    return np.sqrt(vel2) / 1000
################################################################################






################################################################################
# In lack of a bulge model that does not involve any complicated mathematics 
# (integrals) We only have total velocity functions that exclude the bulge
#-------------------------------------------------------------------------------
# Isothermal
#-------------------------------------------------------------------------------
def vel_tot_iso_nb(r, params):

    SigD, Rd, rho0_h, Rh = params

    r_pc = r * 1000
    Rd_pc = Rd * 1000
    Rh_pc = Rh * 1000

    Vdisk = disk_vel(r_pc, SigD, Rd_pc)
    Vhalo = halo_vel_iso(r_pc, rho0_h, Rh_pc)
    v2 = Vdisk ** 2 + Vhalo ** 2

    return np.sqrt(v2)
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
# NFW
#-------------------------------------------------------------------------------
def vel_tot_NFW_nb(r, params):
    SigD, Rd, rho0_h, Rh = params

    r_pc = r * 1000
    Rd_pc = Rd * 1000
    Rh_pc = Rh * 1000

    Vdisk = disk_vel(r_pc, SigD, Rd_pc)
    Vhalo = halo_vel_NFW(r_pc, rho0_h, Rh_pc)
    v2 = Vdisk ** 2 + Vhalo ** 2

    return np.sqrt(v2)
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
# Burket
#-------------------------------------------------------------------------------
def vel_tot_bur_nb(r, params):
    SigD, Rd, rho0_h, Rh = params

    r_pc = r * 1000
    Rd_pc = Rd * 1000
    Rh_pc = Rh * 1000

    Vdisk = disk_vel(r_pc, SigD, Rd_pc)
    Vhalo = halo_vel_bur(r_pc, rho0_h, Rh_pc)
    v2 = Vdisk ** 2 + Vhalo ** 2

    return np.sqrt(v2)
################################################################################



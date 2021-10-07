####################################################################################
# Pacakges
from scipy import integrate as inte
import numpy as np
import matplotlib.pyplot as plt
from numpy import log as ln
from astropy.table import QTable
from scipy.optimize import minimize
import scipy.special as sc
import astropy.units as u
from scipy.special import kn
from scipy.special import iv

# Constants
G = 6.674E-11  # m^3 kg^-1 s^-2
Msun = 1.989E30  # kg
####################################################################################

'''
####################################################################################
# A bulge model (Di Paolo et al 2019)
# With a slight modifcation at low radii (r < Rin)
def bulge_vel(r, A, Vin, Rd):
    if isinstance(r, float):
        if r < 0.2 * Rd:
            v = np.sqrt(A*Vin**2*r/(0.2*Rd))
        else:
            v = np.sqrt(A * Vin **2 * (r/(0.2*Rd)) ** -1)
    else:
        v = np.zeros(len(r))
        for i in range(len(r)):
            if r[i] < 0.2 * Rd:
                v[i] = np.sqrt(A*Vin**2*r[i]/(0.2*Rd))
            else:
                v[i] = np.sqrt(A * Vin **2 * (r[i]/(0.2*Rd)) ** -1)
    return v
####################################################################################
'''



####################################################################################
# Exponential bulge model (Feng2014)
def bulge_vel(r,rho_0,Rb):
    if isinstance(r,float):
        mass_b = 4 * np.pi * rho_0 * ((-1/3*Rb**3*np.exp(-(r/Rb)**3)+(1/3)*(Rb**3)))
    else:
        mass_b = np.zeros(len(r))
        for i in range(len(r)):
            mass_b[i] = 4 * np.pi * rho_0 * ((-(1/3)*(Rb**3)*np.exp(-(r[i]/Rb)**3)+(1/3)*(Rb**3)))
    vel = np.sqrt((G * mass_b * Msun) / (r * 3.086e16))
    return vel/1000
####################################################################################

'''
################################################################################
# de Vaucouleur's bulge model (Integrating volume mass density)
#-------------------------------------------------------------------------------
gamma = 3.3308 # unitless
kappa = gamma*ln(10) # unitless

# surface density - sigma
def sigma_b(x,SigBC,Rb):
    """
    parameters:
    x (projected radius): The projected radius  (pc)
    a (central density): The central density of the bulge (M_sol/pc^2)
    b (central radius): The central radius of the bulge (kpc)

    return: surface density of the bulge (g/pc^2)
    """
    return SigBC*np.exp(-1*kappa*((x/Rb)**0.25-1)) #M_sol/pc^2

# derivative of sigma with respect to r
def dsdx(x,SigBC,Rb):
    """
    parameters:
    x (projected radius): The projected rdius  (pc)
    a (central density): The central density of the bulge (M/pc^2)
    b (central radius): The central radius of the bulge (kpc)

    return: derivative of sigma (g/pc^3)
    """
    return sigma_b(x,SigBC,Rb)*(-0.25*kappa)*(Rb**-0.25)*(x**-0.75) # M_sol/pc^2

# integrand for getting volume density
def density_integrand(x,r,SigBC,Rb):
    """
    parameters:
    x (projected radius): The projected radius  (pc)
    r (radius): The a distance from the centre (pc)
    a (central density): The central density of the bulge (M/pc^2)
    b (central radius): The central radius of the bulge (kpc)

    return: integrand for volume density of the bulge (g/pc^3)
    """
    return -(1/np.pi)*dsdx(x,SigBC,Rb)/np.sqrt(x**2-r**2)

# mass integrand
def mass_integrand(r,SigBC,Rb):
    """
    parameters:
    x (projected radius): The projected rdius  (pc)   
    r (radius): The a distance from the centre (pc)
    a (central density): The central density of the bulge (M/pc^2)
    b (central radius): The central radius of the bulge (kpc)

    return: volume density of the bulge
    """
    # integrating for volume density
    vol_den, vol_den_err = inte.quad(density_integrand, r, np.inf, args=(r,SigBC,Rb))
    return 4*np.pi*vol_den*r**2

# getting a velocity
def bulge_vel_full(r,SigBC,Rb):
    """
    parameters:
    r (radius): The a distance from the centre (pc)
    a (central density): The central density of the bulge (M/pc^2)
    b (central radius): The central radius of the bulge (kpc)

    return: rotational velocity of the bulge (pc/s)
    """
    # integrating to get mass
    if isinstance(r, float):
        bulge_mass, m_err = inte.quad(mass_integrand, 0, r, args=(SigBC, Rb))
    else:
        bulge_mass = np.zeros(len(r))
        err = np.zeros(len(r))

        for i in range(len(r)):
            bulge_mass[i],err[i] = inte.quad(mass_integrand, 0, r[i], args=(SigBC, Rb))

    # v = sqrt(GM/r) for circular velocity
    vel = np.sqrt(G*(bulge_mass*Msun)/(r*3.08E16))
    vel /= 1000

    return vel
################################################################################
'''

####################################################################################
# Disk velocity from Sofue 2013
# -------------------------------------------------------------------------------
# Fitting for central surface density
# def disk_vel(params, r):
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
####################################################################################

####################################################################################
# Circular velocity for isothermal Halo without the complicated integrals
# from eqn (51) & (52) from Sofue 2013.
# integral form can be seen from "rotation_curve_functions.py"
def halo_vel_iso(r, rho0_h, Rh):
    '''
    :param r: The a distance from the centre (pc)
    :param rho_iso: The central density of the halo (M_sol/pc^3)
    :param Rh: The scale radius of the dark matter halo (pc)
    :return: rotational velocity
    '''
    v_inf = np.sqrt((4 * np.pi * G * rho0_h * (Msun) * Rh ** 2) / (3.086e16))/1000 #km/s
    if isinstance(r,float):
        # the part in the square root would be unitless
        vel = v_inf * np.sqrt(1 - ((Rh/r)*np.arctan2(r,Rh)))
    else:
        vel = np.zeros(len(r))
        for i in range(len(r)):
            vel[i] = v_inf * np.sqrt(1 - ((Rh/r[i])*np.arctan2(r[i],Rh)))
    return vel

#####################################################################################

#####################################################################################
# NFW_halo
# velocity calculated from mass with already evaluated integral
# from density profile (eqn 55, Sofue 2013)
# integral form can be seen from "rotation_curve_functions.py"
def halo_vel_NFW(r, rho0_h, Rh):
    if isinstance(r, float):
        halo_mass = 4*np.pi*rho0_h*Rh**3*((-r/(Rh+r)) + np.log(Rh + r) - np.log(Rh))
    else:
        halo_mass = np.zeros(len(r))
        for i in range(len(r)):
            halo_mass[i] = 4*np.pi*rho0_h*Rh**3*((-r[i]/(Rh+r[i])) + np.log(Rh + r[i]) - np.log(Rh))
    vel2 = G * (halo_mass * Msun) / (r * 3.086e16)
    return np.sqrt(vel2)/1000
#####################################################################################

#####################################################################################
# Burket halo
# velocity calculated from mass with already evaluated integral
# from density profile (eqn 56, Sofue 2013)
# integral form can be seen from "rotation_curve_functions.py"
def halo_vel_bur(r,rho0_h, Rh):
    if isinstance(r, float):
        halo_mass = np.pi * (-rho0_h) * (Rh**3) * (-np.log(Rh**2 + r**2) - 2*np.log(Rh + r) + 2*np.arctan2(r, Rh) + np.log(Rh**2)\
                                               + 2*np.log(Rh) - 2*np.arctan2(0, Rh))
    else:
        halo_mass = np.zeros(len(r))
        for i in range(len(r)):
            halo_mass[i] = np.pi * (-rho0_h) * (Rh**3) * (-np.log(Rh**2 + r[i]**2) - 2*np.log(Rh + r[i]) + 2*np.arctan2(r[i], Rh) + np.log(Rh**2)\
                                               + 2*np.log(Rh) - 2*np.arctan2(0, Rh))
    vel2 = G * (halo_mass * Msun) / (r * 3.086e16)
    return np.sqrt(vel2)/1000
#####################################################################################

#####################################################################################
# Isothermal
def vel_tot_iso(r, params):
    rho_0, Rb, SigD, Rd, rho0_h, Rh = params

    r_pc = r * 1000
    Rb_pc = Rb * 1000
    Rd_pc = Rd * 1000
    Rh_pc = Rh * 1000

    Vbulge = bulge_vel(r_pc, rho_0, Rb_pc)
    Vdisk = disk_vel(r_pc, SigD, Rd_pc)
    Vhalo = halo_vel_iso(r_pc, rho0_h, Rh_pc)
    v2 = Vbulge ** 2 + Vdisk ** 2 + Vhalo ** 2

    return np.sqrt(v2)
#------------------------------------------------------------------------------------
# NFW
def vel_tot_NFW(r, params):
    rho_0, Rb, SigD, Rd, rho0_h, Rh = params

    r_pc = r * 1000
    Rb_pc = Rb * 1000
    Rd_pc = Rd * 1000
    Rh_pc = Rh * 1000

    Vbulge = bulge_vel(r_pc, rho_0, Rb_pc)
    Vdisk = disk_vel(r_pc, SigD, Rd_pc)
    Vhalo = halo_vel_NFW(r_pc, rho0_h, Rh_pc)
    v2 = Vbulge ** 2 + Vdisk ** 2 + Vhalo ** 2

    return np.sqrt(v2)
#------------------------------------------------------------------------------------
# Burket
def vel_tot_bur(r, params):
    rho_0, Rb, SigD, Rd, rho0_h, Rh = params

    r_pc = r * 1000
    Rb_pc = Rb * 1000
    Rd_pc = Rd * 1000
    Rh_pc = Rh * 1000

    Vbulge = bulge_vel(r_pc, rho_0, Rb_pc)
    Vdisk = disk_vel(r_pc, SigD, Rd_pc)
    Vhalo = halo_vel_bur(r_pc, rho0_h, Rh_pc)
    v2 = Vbulge ** 2 + Vdisk ** 2 + Vhalo ** 2

    return np.sqrt(v2)
######################################################################################

'''
######################################################################################
# Isothermal (no bulge)
def vel_tot_iso_nb(r, params):
    SigD, Rd, rho0_h, Rh = params

    r_pc = r * 1000
    Rd_pc = Rd * 1000
    Rh_pc = Rh * 1000

    Vdisk = disk_vel(r_pc, SigD, Rd_pc)
    Vhalo = halo_vel_iso(r_pc, rho0_h, Rh_pc)
    v2 = Vdisk ** 2 + Vhalo ** 2

    return np.sqrt(v2)
#------------------------------------------------------------------------------------
# NFW (no bulge)
def vel_tot_NFW_nb(r, params):
    SigD, Rd, rho0_h, Rh = params

    r_pc = r * 1000
    Rd_pc = Rd * 1000
    Rh_pc = Rh * 1000

    Vdisk = disk_vel(r_pc, SigD, Rd_pc)
    Vhalo = halo_vel_NFW(r_pc, rho0_h, Rh_pc)
    v2 = Vdisk ** 2 + Vhalo ** 2

    return np.sqrt(v2)
#------------------------------------------------------------------------------------
# Burket (no bulge)
def vel_tot_bur_nb(r, params):
    SigD, Rd, rho0_h, Rh = params

    r_pc = r * 1000
    Rd_pc = Rd * 1000
    Rh_pc = Rh * 1000

    Vdisk = disk_vel(r_pc, SigD, Rd_pc)
    Vhalo = halo_vel_bur(r_pc, rho0_h, Rh_pc)
    v2 = Vdisk ** 2 + Vhalo ** 2

    return np.sqrt(v2)
######################################################################################
'''
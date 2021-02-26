################################################################################
# All the libraries used & constant values
# -------------------------------------------------------------------------------
from scipy import integrate as inte
import numpy as np
import matplotlib.pyplot as plt
from numpy import log as ln

G = 6.67E-11  # m^3 kg^-1 s^-2
Msun = 1.989E30  # kg

from astropy.table import QTable
from scipy.optimize import minimize
import astropy.units as u
from scipy.special import kn
from scipy.special import iv


################################################################################

################################################################################
# Initial Guess
################################################################################

################################################################################
# Bounds
param_bounds = [[0.2,1], # Scale Factor [unitless]
                [0,1000], # Bulge Scale Velocity [km/s]
                [0, 12],  # Disk mass [log(Msun)]
                [0, 10],  # Disk radius [kpc]
                [0, 1],  # Halo density [Msun/pc^3]
                [0, 100]]  # Halo radius [kpc]
################################################################################

################################################################################
# bulge (Simpler Model)
# -------------------------------------------------------------------------------
def vel_b(r, A, Vin, Rd):
    '''
    :param r: The projected radius (pc)
    :param A: Scale factor [unitless]
    :param Vin: the scale velocity in the bulge (km/s)
    :param Rd: The scale radius of the disk (pc)
    :return: The rotational velocity of the bulge (km/s)
    '''
    v = A * (Vin ** 2) * ((r / (0.2 * Rd)) ** -1)
    return np.sqrt(v)


################################################################################


'''
################################################################################
# bulge (Not the simplest model)
#-------------------------------------------------------------------------------
gamma = 3.3308 # unitless
kappa = gamma*ln(10) # unitless
def sigma_b(x,a,b):
    """
    parameters:
    x (projected radius): The projected radius  (pc)
    a (central density): The central density of the bulge (M_sol/pc^2)
    b (central radius): The central radius of the bulge (kpc)

    return: surface density of the bulge (g/pc^2)
    """
    return a*np.exp(-1*kappa*((x/b)**0.25-1)) #M_sol/pc^2

# derivative of sigma with respect to r
def dsdx(x,a,b):
    """
    parameters:
    x (projected radius): The projected rdius  (pc)
    a (central density): The central density of the bulge (M/pc^2)
    b (central radius): The central radius of the bulge (kpc)

    return: derivative of sigma (g/pc^3)
    """
    return sigma_b(x,a,b)*(-0.25*kappa)*(b**-0.25)*(x**-0.75) # M_sol/pc^2
# integrand for getting denisty
def density_integrand(x,r,a,b):
    """
    parameters:
    x (projected radius): The projected rdius  (pc)   
    r (radius): The a distance from the centre (pc)
    a (central density): The central density of the bulge (M/pc^2)
    b (central radius): The central radius of the bulge (kpc)

    return: integrand for volume density of the bulge (g/pc^3)
    """
    return -(1/np.pi)*dsdx(x,a,b)/np.sqrt(x**2-r**2)
def mass_integrand(r,a,b):
    """
    parameters:
    x (projected radius): The projected rdius  (pc)   
    r (radius): The a distance from the centre (pc)
    a (central density): The central density of the bulge (M/pc^2)
    b (central radius): The central radius of the bulge (kpc)

    return: volume density of the bulge
    """
    vol_den, vol_den_err = inte.quad(density_integrand, r, np.inf, args=(r,a,b))
    return 4*np.pi*vol_den*r**2

# getting a velocity
def vel_b(r,a,b):
    """
    parameters:
    r (radius): The a distance from the centre (pc)
    a (central density): The central density of the bulge (M/pc^2)
    b (central radius): The central radius of the bulge (kpc)

    return: rotational velocity of the bulge (pc/s)
    """
    if isinstance(r, float):
        bulge_mass, m_err = inte.quad(mass_integrand, 0, r, args=(Sigma_be, Rb))
    else:
        bulge_mass = np.zeros(len(r))
        err = np.zeros(len(r))

        for i in range(len(r)):
            bulge_mass[i],err[i] = inte.quad(mass_integrand, 0, r[i], args=(a,b))
    vel = np.sqrt(G*(bulge_mass*1.988E30)/(r*3.08E16))
    vel /= 1000
    return vel
################################################################################
'''


################################################################################
# Disk velocity from Paolo et al. 2019
# -------------------------------------------------------------------------------
# Fitting for disk mass
def v_d(r, Mdisk, Rd):
    '''
    :param r: The a distance from the centre [pc]
    :param Mdisk: The total mass of the disk [M_sun]
    :param Rd: The scale radius of the disk [pc]
    :return: The rotational velocity of the disk [km/s]
    '''
    # Unit conversion
    Mdisk_kg = Mdisk * Msun

    bessel_component = (iv(0, r / (2 * Rd)) * kn(0, r / (2 * Rd)) - iv(1, r / (2 * Rd)) * kn(1, r / (2 * Rd)))
    vel2 = ((0.5) * G * Mdisk_kg * (r / Rd) ** 2 / (Rd * 3.08E16)) * bessel_component

    return np.sqrt(vel2) / 1000


################################################################################


################################################################################
# Disk velocity from Paolo et al. 2019
# -------------------------------------------------------------------------------
# Fitting for central surface density
def diak_vel(params, r):
    '''
    :param SigD: Central surface density for the disk [M_sol/pc^2]
    :param Rd: The scale radius of the disk [pc]
    :param r: The a distance from the centre [pc]
    :return: The rotational velocity of the disk [km/s]
    '''
    SigD, Rd = params
    
    y = r / (2 * Rd)

    bessel_component = (iv(0, y) * kn(0, y) - iv(1, y) * kn(1, y))
    vel2 = (4 *np.pi * G * SigD * y ** 2 * (Rd/(3.086E16))*Msun) * bessel_component

    return np.sqrt(vel2) / 1000


################################################################################

################################################################################
# halo (isothermal)
# -------------------------------------------------------------------------------
# e = rho_0_iso
# f = h

def rho0_iso(Vinf, Rh):
    '''
    parameters:
    Vinf (rotational velocity): The rotational velocity when r approaches infinity (km/s)
    Rh (scale radius): The scale radius of the dark matter halo [kpc]

    return: volume density of the isothermal halo (g/pc^3)
    '''
    return 0.740 * (Vinf / 200) * (Rh) ** (-2)


def rho_iso(r, Vinf, Rh):
    '''
    parameters:
    r (radius): The a distance from the centre (pc)
    Vinf (rotational velocity): The rotational velocity when r approaches infinity (km/s)
    Rh (scale radius): The scale radius of the dark matter halo (pc)

    return: volume density of the isothermal halo (g/pc^3)
    '''
    rho_0 = rho0_iso(Vinf, Rh / 1000)
    return rho_0 / (1 + (r / Rh) ** 2)


def integrand_h_iso(r, Vinf, Rh):
    '''
    parameters:
    r (radius): The a distance from the centre (pc)
    Vinf (rotational velocity): The rotational velocity when r approaches infinity (km/s)
    Rh (scale radius): The scale radius of the dark matter halo (pc)
    return: integrand for getting the mass of the isothermal halo
    '''

    return 4 * np.pi * (rho_iso(r, Vinf, Rh)) * r ** 2


def mass_h_iso(r, Vinf, Rh):
    '''
    parameters:
    r (radius): The a distance from the centre (pc)
    e (rotational velocity): The rotational velocity when r approaches infinity (km/s)
    f (scale radius): The scale radius of the dark matter halo (pc)

    return: mass of the isothermal halo (g)
    '''
    halo_mass, m_err = inte.quad(integrand_h_iso, 0, r, args=(Vinf, Rh))
    return halo_mass


def vel_h_iso(r, Vinf, Rh):
    '''
    parameters:
    r (radius): The a distance from the centre (pc)
    Vinf (rotational velocity): The rotational velocity when r approaches infinity (km/s)
    Rh (scale radius): The scale radius of the dark matter halo (pc)

    return: rotational velocity of the isothermal halo (pc/s)
    '''
    if isinstance(r, float):
        halo_mass = mass_h_iso(r, Vinf, Rh)
    else:
        halo_mass = np.zeros(len(r))
        for i in range(len(r)):
            halo_mass[i] = mass_h_iso(r[i], Vinf, Rh)

    vel = np.sqrt(G * (halo_mass * Msun) / (r * 3.08E16))
    vel /= 1000
    return vel
################################################################################


################################################################################
# halo (NFW)
# -------------------------------------------------------------------------------
# e = rho_0_NFW
# f = h

def rho_NFW(r, rho0_h, Rh):
    '''
    parameters:
    r (radius): The a distance from the centre (pc)
    rho0_h (central density): The central density of the halo (M_sol/pc^3)
    Rh (scale radius): The scale radius of the dark matter halo (pc)

    return: volume density of the isothermal halo (M/pc^3)
    '''
    return rho0_h / ((r / Rh) * ((1 + (r / Rh)) ** 2))


def integrand_h_NFW(r, rho0_h, Rh):
    '''
    parameters:
    r (radius): The a distance from the centre (pc)
    rho0_h (central density): The central density of the halo (M_sol/pc^3)
    Rh (scale radius): The scale radius of the dark matter halo (pc)

    return: integrand for getting the mass of the isothermal halo
    '''

    return 4 * np.pi * (rho_NFW(r, rho0_h, Rh)) * r ** 2


def mass_h_NFW(r, rho0_h, Rh):
    '''
    parameters:
    r (radius): The a distance from the centre (pc)
    rho0_h (central density): The central density of the halo (M_sol/pc^3)
    Rh (scale radius): The scale radius of the dark matter halo (pc)

    return: mass of the isothermal halo (g)
    '''
    halo_mass, m_err = inte.quad(integrand_h_NFW, 0, r, args=(rho0_h, Rh))
    return halo_mass


def vel_h_NFW(r, rho0_h, Rh):
    '''
    parameters:
    r (radius): The a distance from the centre (pc)
    rho0_h (central density): The central density of the halo (M_sol/pc^3)
    Rh (scale radius): The scale radius of the dark matter halo (pc)

    return: rotational velocity of the NFW halo (pc/s)
    '''
    if isinstance(r, float):
        halo_mass = mass_h_NFW(r, rho0_h, Rh)
    else:
        halo_mass = np.zeros(len(r))
        for i in range(len(r)):
            halo_mass[i] = mass_h_NFW(r[i], rho0_h, Rh)

    vel = np.sqrt(G * (halo_mass * Msun) / (r * 3.0857E16))
    vel /= 1000
    return vel


################################################################################


################################################################################
# halo (Burket)
# -------------------------------------------------------------------------------
# e = rho_0_Bur
# f = h

def rho_Burket(r, rho0_h, Rh):
    '''
    :param r: The distance from the centre (pc)
    :param rho0_h: The central density of the halo (M_sol/pc^3)
    :param Rh: The scale radius of the dark matter halo (pc)
    :return: volume density of the isothermal halo (M/pc^3)
    '''
    return (rho0_h * Rh ** 3) / ((r + Rh) * (r ** 2 + Rh ** 2))


def integrand_h_Burket(r, rho0_h, Rh):
    '''
    :param r: The a distance from the centre (pc)
    :param rho0_h: The central density of the halo (M_sol/pc^3)
    :param Rh: The scale radius of the dark matter halo (pc)
    :return: integrand for getting the mass of the isothermal halo
    '''
    return 4 * np.pi * (rho_Burket(r, rho0_h, Rh)) * r ** 2


def mass_h_Burket(r, rho0_h, Rh):
    '''
    :param r: The a distance from the centre (pc)
    :param rho0_h: The central density of the halo (M_sol/pc^3)
    :param Rh: The scale radius of the dark matter halo (pc)
    :return: mass of the isothermal halo (g)
    '''
    halo_mass, m_err = inte.quad(integrand_h_Burket, 0, r, args=(rho0_h, Rh))
    return halo_mass


def vel_h_Burket(r, rho0_h, Rh):
    '''
    r (radius): The a distance from the centre [pc]
    rho0_h (central density): The central density of the halo [M_sol/pc^3]
    Rh (scale radius): The scale radius of the dark matter halo [pc]
    :return: rotational velocity of the Burket halo [km/s]
    '''
    if isinstance(r, float):
        halo_mass = mass_h_Burket(r, rho0_h, Rh)
    else:
        halo_mass = np.zeros(len(r))
        for i in range(len(r)):
            halo_mass[i] = mass_h_Burket(r[i], rho0_h, Rh)

    # Unit conversion
    halo_mass_kg = halo_mass * Msun

    vel = np.sqrt(G * halo_mass_kg / (r * 3.0857E16))
    vel /= 1000
    return vel


################################################################################


################################################################################
# Total Velocity
# -------------------------------------------------------------------------------
# Isothermal Model
def v_co_iso(r, params):
    '''
    r (radius): The a distance from the centre (kpc)
    params:
      - (scale factor): Scale factor for the bulge [unitless]
      - (interior velocity): The velocity in the bulge [km/s]
      - (central radius): The central radius of the bulge (kpc)
      - (mass of disk): The total mass of the disk [log(Msun)]
      - (disk radius): The central radius of the disk (kpc)
      - (rotational velocity): The rotational velocity when r approaches infinity (km/s)
      - (scale radius): The scale radius of the dark matter halo (kpc)
    '''

    A, Vin, logMdisk, Rd, Vinf, Rh = params

    # Unit conversion
    r_pc = r * 1000
    Mdisk = 10 ** logMdisk
    Rd_pc = Rd * 1000
    Rh_pc = Rh * 1000

    Vbulge = vel_b(r_pc, A, Vin, Rd_pc)
    Vdisk = v_d(r_pc, Mdisk, Rd_pc)
    Vhalo = vel_h_iso(r_pc, Vinf, Rh_pc)

    return np.sqrt(Vbulge ** 2 + Vdisk ** 2 + Vhalo ** 2)  # km/s


# Isothermal Model (No Bulge)
def v_co_iso_nb(r, params):
    '''
    r (radius): The a distance from the centre (kpc)
    params:
      - (mass of disk): The total mass of the disk [log(Msun)]
      - (disk radius): The central radius of the disk (kpc)
      - (rotational velocity): The rotational velocity when r approaches infinity (km/s)
      - (scale radius): The scale radius of the dark matter halo (kpc)
    '''
    logMdisk, Rd, Vinf, Rh = params

    # Unit conversion
    r_pc = r * 1000
    Mdisk = 10 ** logMdisk
    Rd_pc = Rd * 1000
    Rh_pc = Rh * 1000

    Vdisk = v_d(r_pc, Mdisk, Rd_pc)
    Vhalo = vel_h_iso(r_pc, Vinf, Rh_pc)

    return np.sqrt(Vdisk ** 2 + Vhalo ** 2)  # km/s


# NFW Model
def v_co_NFW(r, params):
    '''
    r (radius): The a distance from the centre (kpc)
    params:
      - (scale factor): Scale factor for the bulge [unitless]
      - (interior velocity): The velocity in the bulge [km/s]
      - (central radius): The central radius of the bulge (kpc)
      - (mass of disk): The total mass of the disk [log(Msun)]
      - (disk radius): The central radius of the disk (kpc)
      - (central density): The central density of the halo (M_sol/pc^3)
      - (scale radius): The scale radius of the dark matter halo (kpc)
    '''

    A, Vin, logMdisk, Rd, rho0_h, Rh = params

    # Unit conversion
    r_pc = r * 1000
    Mdisk = 10 ** logMdisk
    Rd_pc = Rd * 1000
    Rh_pc = Rh * 1000

    Vbulge = vel_b(r_pc, A, Vin, Rd_pc)
    Vdisk = v_d(r_pc, Mdisk, Rd_pc)
    Vhalo = vel_h_NFW(r_pc, rho0_h, Rh_pc)

    return np.sqrt(Vbulge ** 2 + Vdisk ** 2 + Vhalo ** 2)  # km/s


# NFW Model (No Bulge)
def v_co_NFW_nb(r, params):
    '''
    r (radius): The a distance from the centre (kpc)
    params:
      - (mass of disk): The total mass of the disk [log(Msun)]
      - (disk radius): The central radius of the disk (kpc)
      - (central density): The central density of the halo (M_sol/pc^3)
      - (scale radius): The scale radius of the dark matter halo (kpc)
    '''

    logMdisk, Rd, rho0_h, Rh = params

    # Unit conversion
    r_pc = r * 1000
    Mdisk = 10 ** logMdisk
    Rd_pc = Rd * 1000
    Rh_pc = Rh * 1000

    Vdisk = v_d(r_pc, Mdisk, Rd_pc)
    Vhalo = vel_h_NFW(r_pc, rho0_h, Rh_pc)

    return np.sqrt(Vdisk ** 2 + Vhalo ** 2)  # km/s


# Burket Model
def v_co_Burket(r, params):
    '''
    r (radius): The a distance from the centre (kpc)
    params:
      - (scale factor): Scale factor for the bulge [unitless]
      - (interior velocity): The velocity in the bulge [km/s]
      - (central radius): The central radius of the bulge (kpc)
      - (mass of disk): The total mass of the disk [log(Msun)]
      - (disk radius): The central radius of the disk (kpc)
      - (central density): The central density of the halo (M_sol/pc^3)
      - (scale radius): The scale radius of the dark matter halo (kpc)
    '''
    A, Vin, logMdisk, Rd, rho0_h, Rh = params

    # Unit conversion
    r_pc = r * 1000
    Mdisk = 10 ** logMdisk
    Rd_pc = Rd * 1000
    Rh_pc = Rh * 1000

    Vbulge = vel_b(r_pc, A, Vin, Rd_pc)
    Vdisk = v_d(r_pc, Mdisk, Rd_pc)
    Vhalo = vel_h_Burket(r_pc, rho0_h, Rh_pc)

    return np.sqrt(Vbulge ** 2 + Vdisk ** 2 + Vhalo ** 2)  # km/s


# Burket Model (No Bulge)
def v_co_Burket_nb(r, params):
    '''
    r (radius): The a distance from the centre (kpc)
    params:
      - (scale factor): Scale factor for the bulge [unitless]
      - (interior velocity): The velocity in the bulge [km/s]
      - (central radius): The central radius of the bulge (kpc)
      - (mass of disk): The total mass of the disk [log(Msun)]
      - (disk radius): The central radius of the disk (kpc)
      - (central density): The central density of the halo (M_sol/pc^3)
      - (scale radius): The scale radius of the dark matter halo (kpc)
    '''
    logMdisk, Rd, rho0_h, Rh = params

    # Unit conversion
    r_pc = r * 1000
    Mdisk = 10 ** logMdisk
    Rd_pc = Rd * 1000
    Rh_pc = Rh * 1000

    Vdisk = v_d(r_pc, Mdisk, Rd_pc)
    Vhalo = vel_h_Burket(r_pc, rho0_h, Rh_pc)

    return np.sqrt(Vdisk ** 2 + Vhalo ** 2)  # km/s


################################################################################

################################################################################
# Loglike function (Burket w/ bulge)
# -------------------------------------------------------------------------------
def loglike_Iso(theta, r, v, v_err):
    model = v_co_iso(np.array(r), theta)

    inv_sigma2 = 1.0 / (np.array(v_err) ** 2)

    logL = -0.5 * (np.sum((np.array(v) - model) ** 2 * inv_sigma2 - np.log(inv_sigma2)))

    # Additional (physical) penalties
    if theta[3] < theta[1]:
        logL += 1E6

    return logL


# Negative likelihood
def nloglike_Iso(theta, r, v, v_err):
    return -loglike_Iso(theta, r, v, v_err)
#################################################################################

################################################################################
# Loglike function (Burket no bulge)
# -------------------------------------------------------------------------------
def loglike_Iso_nb(theta, r, v, v_err):
    model = v_co_iso_nb(np.array(r), theta)

    inv_sigma2 = 1.0 / (np.array(v_err) ** 2)

    logL = -0.5 * (np.sum((np.array(v) - model) ** 2 * inv_sigma2 - np.log(inv_sigma2)))

    # Additional (physical) penalties
    if theta[3] < theta[1]:
        logL += 1E10

    return logL

# Negative likelihood
def nloglike_Iso_nb(theta, r, v, v_err):
    return -loglike_Iso_nb(theta, r, v, v_err)
#################################################################################

#################################################################################
# Fitting Function (Isothermal)

def RC_fitting_Iso(r,m,v,v_err):
    '''

    :param r: The a distance from the centre (kpc)
    :param m: Mass of the object (M_sol)
    :param v: rotational velocity (km/s)
    :param v_err: error in the rotational velocity (km/s)
    :return: The fitted parameters
    '''
    # variables for initial guesses
    a_guess = 0.2
    v_inf_b_guess = 150
    logM_guess = np.log10(m)+0.5
    r_d_guess = max(np.array(r)) / 3
    v_inf_h_guess = 200
    r_h_guess = max(list(r))*10
    if max(list(r)) < 5:
        logM_guess += 0.5
    half_idx = int(0.5 * len(r))
    if v[half_idx] < max(v[:half_idx]):
            p0 = [a_guess, v_inf_b_guess,logM_guess , r_d_guess , v_inf_h_guess, r_h_guess]
            param_bounds = [[0.2, 1],  # Scale Factor [unitless]
                            [0.001, 1000],  # Bulge Scale Velocity [km/s]
                            [8, 12],  # Disk mass [log(Msun)]
                            [0.1,20],  # Disk radius [kpc]
                            [0.001, 1000],  # Halo density [Msun/pc^2]
                            [0.1, 1000]]  # Halo radius [kpc]

            bestfit = minimize(nloglike_Iso, p0, args=(r, v, v_err, WF50),
                              bounds=param_bounds)
            print('---------------------------------------------------')
            print(bestfit)
    else: # No Bulge
            p0 = [logM_guess, r_d_guess, v_inf_h_guess, r_h_guess]
            param_bounds = [[8, 12],  # Disk mass [log(Msun)]
                            [0.1, 20],  # Disk radius [kpc]
                            [0.001, 1000],  # Halo density [Msun/pc^2]
                            [0.1, 1000]]  # Halo radius [kpc]

            bestfit = minimize(nloglike_Iso_nb, p0, args=(r, v, v_err),
                              bounds=param_bounds)
            print('---------------------------------------------------')
            print(bestfit)
    return bestfit
#################################################################################

#################################################################################
# Plotting (Isohermal)
def RC_plotting_Iso(r, v, v_err, bestfit, ID):
    half_idx = int(0.5 * len(r))
    if v[half_idx] < max(v[:half_idx]):
            if max(list(r)) < bestfit.x[3]:
                r_plot = np.linspace(0,3*bestfit.x[3],100)
            else:
                r_plot = np.linspace(0,3*max(list(r)),100)
            plt.errorbar(r, v, yerr=v_err, fmt='g*', label='data')
            plt.plot(r_plot, v_co_iso(np.array(r_plot), bestfit.x), '--', label='fit')
            plt.plot(r_plot, vel_b(np.array(r_plot) * 1000, bestfit.x[0], bestfit.x[1], bestfit.x[3] * 1000),color='green',
                 label='bulge')
            plt.plot(r_plot, v_d(np.array(r_plot) * 1000, 10 ** bestfit.x[2], bestfit.x[3] * 1000),color='orange',label='disk')
            plt.plot(r_plot, vel_h_iso(np.array(r_plot) * 1000, bestfit.x[4],bestfit.x[5] * 1000),color='blue',
                 label='Isothermal halo')
            plt.legend()
            plt.xlabel('$r_{dep}$ [kpc]')
            plt.ylabel('$v_{rot}$ [km/s]')
            plt.title(ID)
            plt.show()
    else:
            if max(list(r)) < bestfit.x[1]:
                r_plot = np.linspace(0,3*bestfit.x[1],100)
            else:
                r_plot = np.linspace(0,3*max(list(r)),100)
            plt.errorbar(r, v, yerr=v_err, fmt='g*', label='data')
            plt.plot(r_plot, v_co_iso_nb(np.array(r_plot), bestfit.x), '--', label='fit')
            plt.plot(r_plot, v_d(np.array(r_plot) * 1000, 10 ** bestfit.x[0], bestfit.x[1] * 1000),color='orange',
                    label='disk')
            plt.plot(r_plot, vel_h_iso(np.array(r_plot) * 1000, bestfit.x[2], bestfit.x[3] * 1000),color='blue',
                    label='Isothermal halo')
            plt.legend()
            plt.xlabel('$r_{dep}$ [kpc]')
            plt.ylabel('$v_{rot}$ [km/s]')
            plt.title(ID)
            plt.show()
#################################################################################

################################################################################
# Loglike function (Burket w/ bulge)
# -------------------------------------------------------------------------------
def loglike_Bur(theta, r, v, v_err, WF50):
    model = v_co_Burket(np.array(r), theta)

    inv_sigma2 = 1.0 / (np.array(v_err) ** 2)

    logL = -0.5 * (np.sum((np.array(v) - model) ** 2 * inv_sigma2 - np.log(inv_sigma2)))
    return logL


# Negative likelihood
def nloglike_Bur(theta, r, v, v_err,WF50):
    nlogL = -loglike_Bur(theta, r, v, v_err, WF50)
    # Additional (physical) penalties
    # If disk radius greater than halo radius
    if theta[3] < theta[1]:
        nlogL += 1E6
    # If max velocity greater than HI
    if v_co_Burket(5 * max(np.array(r)), theta) > WF50:
        nlogL += 1E3
    return nlogL
#################################################################################

################################################################################
# Loglike function (Burket no bulge)
# -------------------------------------------------------------------------------
def loglike_Bur_nb(theta, r, v, v_err,WF50):
    model = v_co_Burket_nb(np.array(r), theta)

    inv_sigma2 = 1.0 / (np.array(v_err) ** 2)

    logL = -0.5 * (np.sum((np.array(v) - model) ** 2 * inv_sigma2 - np.log(inv_sigma2)))
    return logL

# Negative likelihood
def nloglike_Bur_nb(theta, r, v, v_err,WF50):
    nlogL = -loglike_Bur_nb(theta, r, v, v_err,WF50)
    # Additional (physical) penalties
    # If disk radius greater than halo radius
    if theta[3] < theta[1]:
        nlogL += 1E6
    # If max velocity greater than HI
    if v_co_Burket_nb(5*max(np.array(r)),theta) > WF50:
        nlogL += 1E3
    return nlogL
#################################################################################

#################################################################################
# Fitting Function (Burket)
def RC_fitting_Bur(r,m,v,v_err,WF50):
    '''
    :param r: The a distance from the centre (kpc)
    :param m: Mass of the object (M_sol)
    :param v: rotational velocity (km/s)
    :param v_err: error in the rotational velocity (km/s)
    :param WF50: HI data (km/s)
    :return: The fitted parameters
    '''
    # variables for initial guesses
    a_guess = 0.2
    v_inf_guess = 150
    logM_guess = np.log10(m)
    r_d_guess = max(np.array(r))/5.25
    rho_dc_guess = 0.0051
    r_h_guess = max(list(r))*1.1
    if max(list(r)) < 5:
        rho_dc_guess /= 100
    half_idx = int(0.5 * len(r))
    if v[half_idx] < max(v[:half_idx]):
            p0 = [a_guess, v_inf_guess,logM_guess , r_d_guess , rho_dc_guess, r_h_guess]
            param_bounds = [[0.2, 1],  # Scale Factor [unitless]
                            [0.001, 1000],  # Bulge Scale Velocity [km/s]
                            [8, 12],  # Disk mass [log(Msun)]
                            [0.1,20],  # Disk radius [kpc]
                            [0.0001, 1],  # Halo density [Msun/pc^2]
                            [0.1, 1000]]  # Halo radius [kpc]

            bestfit = minimize(nloglike_Bur, p0, args=(r, v, v_err,WF50),
                              bounds=param_bounds)
            print('---------------------------------------------------')
            print(bestfit)
    else: # No Bulge
            p0 = [logM_guess, r_d_guess, rho_dc_guess, r_h_guess]
            param_bounds = [[8, 12],  # Disk mass [log(Msun)]
                            [0.1, 20],  # Disk radius [kpc]
                            [0.0001, 1],  # Halo density [Msun/pc^2]
                            [0.1, 1000]]  # Halo radius [kpc]

            bestfit = minimize(nloglike_Bur_nb, p0, args=(r, v, v_err,WF50),
                              bounds=param_bounds)
            print('---------------------------------------------------')
            print(bestfit)
    return bestfit
#################################################################################

#################################################################################
# Plotting (Burket)
def RC_plotting_Bur(r,v, v_err, bestfit, ID):
    half_idx = int(0.5 * len(r))
    if v[half_idx] < max(v[:half_idx]):
            if max(list(r)) < bestfit.x[3]:
                r_plot = np.linspace(0,3*bestfit.x[3],100)
            else:
                r_plot = np.linspace(0,3*max(list(r)),100)
            plt.errorbar(r, v, yerr=v_err, fmt='g*', label='data')
            plt.plot(r_plot, v_co_Burket(np.array(r_plot), bestfit.x), '--', label='fit')
            plt.plot(r_plot, vel_b(np.array(r_plot) * 1000, bestfit.x[0], bestfit.x[1], bestfit.x[3] * 1000),color='green',
                 label='bulge')
            plt.plot(r_plot, v_d(np.array(r_plot) * 1000, 10 ** bestfit.x[2], bestfit.x[3] * 1000),color='orange',label='disk')
            plt.plot(r_plot, vel_h_Burket(np.array(r_plot) * 1000, bestfit.x[4],bestfit.x[5] * 1000),color='blue',
                 label='Burket halo')
            plt.legend()
            plt.xlabel('$r_{dep}$ [kpc]')
            plt.ylabel('$v_{rot}$ [km/s]')
            plt.title(ID)
            plt.show()
    else:
            if max(list(r)) < bestfit.x[1]:
                r_plot = np.linspace(0,3*bestfit.x[1],100)
            else:
                r_plot = np.linspace(0,3*max(list(r)),100)
            plt.errorbar(r, v, yerr=v_err, fmt='g*', label='data')
            plt.plot(r_plot, v_co_Burket_nb(np.array(r_plot), bestfit.x), '--', label='fit')
            plt.plot(r_plot, v_d(np.array(r_plot) * 1000, 10 ** bestfit.x[0], bestfit.x[1] * 1000),color='orange',
                    label='disk')
            plt.plot(r_plot, vel_h_Burket(np.array(r_plot) * 1000, bestfit.x[2], bestfit.x[3] * 1000),color='blue',
                    label='Burket halo')
            plt.legend()
            plt.xlabel('$r_{dep}$ [kpc]')
            plt.ylabel('$v_{rot}$ [km/s]')
            plt.title(ID)
            plt.show()
#################################################################################

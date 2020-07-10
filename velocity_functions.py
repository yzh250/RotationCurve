################################################################################
# All the libraries used & constant values
#-------------------------------------------------------------------------------
from scipy import integrate as inte
import numpy as np
import matplotlib.pyplot as plt
from numpy import log as ln

G = 6.67E-11 # m^3 kg^-1 s^-2
Msun = 1.989E30 # kg

from astropy.table import QTable
from scipy.optimize import curve_fit
import astropy.units as u
from scipy.special import kn
from scipy.special import iv
import dynesty
from dynesty import plotting as dyplot
################################################################################



################################################################################
# bulge (Simpler Model)
#-------------------------------------------------------------------------------
def vel_b(r, A, Vin, Rd):
    '''
    :param r: The projected radius (pc)
    :param A: Scale factor [unitless]
    :param Vin: the scale velocity in the bulge (km/s)
    :param Rd: The scale radius of the disk (pc)
    :return: The rotational velocity of the bulge (km/s)
    '''
    v = A*(Vin**2)*((r/(0.2*Rin))**-1)
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
#-------------------------------------------------------------------------------
def v_d(r, Mdisk, Rd):
    '''
    :param r: The a distance from the centre [pc]
    :param Mdisk: The total mass of the disk [M_sun]
    :param Rd: The scale radius of the disk [pc]
    :return: The rotational velocity of the disk [km/s]
    '''
    # Unit conversion
    Mdisk_kg = Mdisk*Msun

    bessel_component = (iv(0,r/(2*Rd))*kn(0,r/(2*Rd)) - iv(1,r/(2*Rd))*kn(1,r/(2*Rd)))
    vel = ((0.5)*G*Mdisk_kg*(r/Rd)**2/(Rd*3.08E16))*bessel_component

    return np.sqrt(vel)/1000
################################################################################


################################################################################
# halo (isothermal)
#-------------------------------------------------------------------------------
# e = rho_0_iso
# f = h

def rho0_iso(Vinf, Rh):
    '''
    parameters:
    Vinf (rotational velocity): The rotational velocity when r approaches infinity (km/s)
    Rh (scale radius): The scale radius of the dark matter halo [kpc]
    
    return: volume density of the isothermal halo (g/pc^3)
    '''
    return 0.740*(Vinf/200)*(Rh)**(-2)

def rho_iso(r, Vinf, Rh):
    '''
    parameters:
    r (radius): The a distance from the centre (pc)
    Vinf (rotational velocity): The rotational velocity when r approaches infinity (km/s)
    Rh (scale radius): The scale radius of the dark matter halo (pc)
    
    return: volume density of the isothermal halo (g/pc^3)
    '''
    rho_0 = rho0_iso(Vinf, Rh/1000)
    return rho_0/(1 + (r/Rh)**2)

def integrand_h_iso(r, Vinf, Rh):
    '''
    parameters:
    r (radius): The a distance from the centre (pc)
    Vinf (rotational velocity): The rotational velocity when r approaches infinity (km/s)
    Rh (scale radius): The scale radius of the dark matter halo (pc)

    return: integrand for getting the mass of the isothermal halo 
    '''
    
    return 4*np.pi*(rho_iso(r, Vinf, Rh))*r**2

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

    vel = np.sqrt(G*(halo_mass*Msun)/(r*3.08E16))
    vel /= 1000
    return vel
################################################################################


################################################################################
# halo (NFW)
#-------------------------------------------------------------------------------
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
    return  rho0_h/((r/Rh)*((1 + (r/Rh))**2))

def integrand_h_NFW(r, rho0_h, Rh):
    '''
    parameters:
    r (radius): The a distance from the centre (pc)
    rho0_h (central density): The central density of the halo (M_sol/pc^3)
    Rh (scale radius): The scale radius of the dark matter halo (pc)
    
    return: integrand for getting the mass of the isothermal halo 
    '''
    
    return 4*np.pi*(rho_NFW(r, rho0_h, Rh))*r**2

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

    vel = np.sqrt(G*(halo_mass*Msun)/(r*3.0857E16))
    vel /= 1000
    return vel
################################################################################


################################################################################
# halo (Burket)
#-------------------------------------------------------------------------------
# e = rho_0_Bur
# f = h

def rho_Burket(r, rho0_h, Rh):
    '''
    :param r: The distance from the centre (pc)
    :param rho0_h: The central density of the halo (M_sol/pc^3)
    :param Rh: The scale radius of the dark matter halo (pc)
    :return: volume density of the isothermal halo (M/pc^3)
    '''
    return  (rho0_h*Rh**3)/((r + Rh)*(r**2 + Rh**2))

def integrand_h_Burket(r, rho0_h, Rh):
    '''
    :param r: The a distance from the centre (pc)
    :param rho0_h: The central density of the halo (M_sol/pc^3)
    :param Rh: The scale radius of the dark matter halo (pc)
    :return: integrand for getting the mass of the isothermal halo
    '''
    return 4*np.pi*(rho_Burket(r, rho0_h, Rh))*r**2

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
    halo_mass_kg = halo_mass*Msun

    vel = np.sqrt(G*halo_mass_kg/(r*3.0857E16))
    vel /= 1000
    return vel
################################################################################


################################################################################
# Total Velocity
#-------------------------------------------------------------------------------
# Isothermal Model
def total_v_iso(r, params):
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
    r_pc = r*1000
    Mdisk = 10**logMdisk
    Rd_pc = Rd*1000
    Rh_pc = Rh*1000

    Vbulge = vel_b(r_pc, A, Vin, Rd_pc)
    Vdisk = v_d(r_pc, Mdisk, Rd_pc)
    Vhalo = vel_h_iso(r_pc, Vinf, Rh_pc)

    return np.sqrt(Vbulge**2 + Vdisk**2 + Vhalo**2) #km/s


# Isothermal Model (No Bulge)
def v_co_iso_nb(r, params):
    '''
    r (radius): The a distance from the centre (kpc)
    params:
      - (mass of disk): The total mass Xof the disk [log(Msun)]
      - (disk radius): The central radius of the disk (kpc)
      - (rotational velocity): The rotational velocity when r approaches infinity (km/s)
      - (scale radius): The scale radius of the dark matter halo (kpc)
    '''
    logMdisk, Rd, Vinf, Rh = params

    # Unit conversion
    r_pc = r*1000
    Mdisk = 10**logMdisk
    Rd_pc = Rd*1000
    Rh_pc = Rh*1000

    Vdisk = v_d(r_pc, Mdisk, Rd_pc)
    Vhalo = vel_h_iso(r_pc, Vinf, Rh_pc)

    return np.sqrt(Vdisk**2 + Vhalo**2) #km/s


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
    r_pc = r*1000
    Mdisk = 10**logMdisk
    Rd_pc = Rd*1000
    Rh_pc = Rh*1000

    Vbulge = vel_b(r_pc, A, Vin, Rd_pc)
    Vdisk = v_d(r_pc, Mdisk, Rd_pc)
    Vhalo = vel_h_NFW(r_pc, rho0_h, Rh_pc)

    return np.sqrt(Vbulge**2 + Vdisk**2 + Vhalo**2) #km/s


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
    r_pc = r*1000
    Mdisk = 10**logMdisk
    Rd_pc = Rd*1000
    Rh_pc = Rh*1000

    Vdisk = v_d(r_pc, Mdisk, Rd_pc)
    Vhalo = vel_h_NFW(r_pc, rho0_h, Rh_pc)

    return np.sqrt(Vdisk**2 + Vhalo**2) #km/s


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
    r_pc = r*1000
    Mdisk = 10**logMdisk
    Rd_pc = Rd*1000
    Rh_pc = Rh*1000

    Vbulge = vel_b(r_pc, A, Vin, Rd_pc)
    Vdisk = v_d(r_pc, Mdisk, Rd_pc)
    Vhalo = vel_h_Burket(r_pc, rho0_h, Rh_pc)

    return np.sqrt(Vbulge**2 + Vdisk**2 + Vhalo**2) #km/s


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
    r_pc = r*1000
    Mdisk = 10**logMdisk
    Rd_pc = Rd*1000
    Rh_pc = Rh*1000

    Vdisk = v_d(r_pc, Mdisk, Rd_pc)
    Vhalo = vel_h_Burket(r_pc, rho0_h, Rh_pc)

    return np.sqrt(Vdisk**2 + Vhalo**2) #km/s
################################################################################


################################################################################
# Loglike function
#-------------------------------------------------------------------------------
def loglike_Bur_nb(theta, r, v, v_err):
    model = v_co_Burket_nb(np.array(r), theta)

    inv_sigma2 = 1.0 / (np.array(v_err)**2)

    logL = -0.5 * (np.sum((np.array(v) - model)**2 * inv_sigma2 - np.log(inv_sigma2)))

    # Additional (physical) penalties
    if theta[3] < theta[1]:
        logL += 1E6
    
    return logL

# Negative likelihood
def nloglike_Bur_nb(theta, r, v, v_err):
    return -loglike_Bur_nb(theta, r, v, v_err)
################################################################################
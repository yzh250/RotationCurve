################################################################################
# Import modules
#-------------------------------------------------------------------------------
import numpy as np
import numpy.ma as ma
from astropy.table import Table, QTable
import matplotlib.pyplot as plt
################################################################################

################################################################################
# Bulge
def bulge_mass(rho_0,Rb):
    Rb = 1000 * Rb
    mass_b = 8 * np.pi * Rb**3 * 10 ** rho_0
    return mass_b

# Disk
def disk_mass(SigD,Rd):
    Rd = 1000 * Rd
    mass_d = 2 * np.pi * SigD * Rd ** 2
    return mass_d

# Isothermal
def halo_mass_iso(rho0_h,r,Rh):
    r = r * 1000
    Rh = 1000 * Rh
    halo_mass = 4 * np.pi * (10 ** rho0_h) * Rh**2 * (r - Rh * np.arctan2(r,Rh))
    return halo_mass

# NFW
def halo_mass_NFW(rho0_h,r,Rh):
    r = r*1000
    Rh = 1000 * Rh
    halo_mass = 4.0 * np.pi * (10 ** rho0_h) * Rh**3 * ((Rh/(Rh + r)) + np.log(Rh + r) - 1.0 - np.log(Rh))
    return halo_mass
# Burket
def halo_mass_bur(rho0_h,r,Rh):
    r = r * 1000
    Rh = 1000 * Rh
    halo_mass = np.pi * (-10 ** rho0_h) * (Rh**3) * (-np.log(Rh**2 + r**2) - 2.0*np.log(Rh + r) + 2*np.arctan2(r, Rh) + np.log(Rh**2) + 2*np.log(Rh))
    return halo_mass

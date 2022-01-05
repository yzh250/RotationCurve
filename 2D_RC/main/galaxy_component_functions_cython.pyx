'''
Cythonized versions of some of the functions found in 
galaxy_component_functions.py
'''

cimport cython

cimport numpy as np

import numpy as np

from libc.math cimport sqrt, exp, log, atan2

from typedefs cimport DTYPE_F64_t

from scipy.special.cython_special cimport i0, i1, k0, k1




################################################################################
# Constants
#-------------------------------------------------------------------------------
cdef DTYPE_F64_t G = 6.674e-11  # m^3 kg^-1 s^-2

cdef DTYPE_F64_t Msun = 1.989e30  # kg

cdef DTYPE_F64_t pi = np.pi

cdef DTYPE_F64_t gamma = 3.3308

cdef DTYPE_F64_t kappa = gamma * log(10.0)
################################################################################




################################################################################
# de Vaucouleurs bulge model (Sofue 2017)
#-------------------------------------------------------------------------------
cpdef sigma_b(DTYPE_F64_t r, 
              DTYPE_F64_t SigBE, 
              DTYPE_F64_t Rb):
    '''
    Function to calculate the surface mass density of the de Vaucouleurs bulge.


    PARAMETERS
    ==========

    r : The distance from the center [pc]

    SigBE : The surface mass density at the scale radius [Msun/pc^2]

    Rb : The scale radius of the bulge [pc]


    RETURNS
    =======

    SigB : The surface mass density at radius r [Msun/pc^2]
    '''

    cdef DTYPE_F64_t SigB

    SigB = SigBE * exp(-kappa * ((r / Rb)**0.25 - 1))

    return SigB



cpdef dSdx(DTYPE_F64_t x, 
           DTYPE_F64_t SigBE, 
           DTYPE_F64_t Rb):
    '''
    Function to calculate the derivative of the surface mass density of the 
    de Vaucouleurs bulge.


    PARAMETERS
    ==========

    x : The distance from the center [pc]

    SigBE : The surface mass density at the scale radius [Msun/pc^2]

    Rb : The scale radius of the bulge [pc]


    RETURNS
    =======

    dSigb_dx : The first derivative of the surface mass density at radius x 
        [Msun/pc^3]
    '''

    cdef DTYPE_F64_t dSigb_dx

    dSigb_dx = sigma_b(x, SigBE, Rb) * (-0.25 * kappa / x) * (x / Rb)**-0.25

    return dSigb_dx



cpdef Sigb_integrand(DTYPE_F64_t x, 
                     DTYPE_F64_t r, 
                     DTYPE_F64_t SigBE, 
                     DTYPE_F64_t Rb):
    '''
    Function to calculate the integrand of the integral to calculate the 
    volume mass density of the de Vaucouleurs bulge.


    PARAMETERS
    ==========

    x : integral variable [pc]

    r : The distance from the center [pc]

    SigBE : The surface mass density at the scale radius [Msun/pc^2]

    Rb : The scale radius of the bulge [pc]


    RETURNS
    =======

    drho_dx : The integrand of the volume mass density integral [Msun/pc^4]
    '''

    cdef DTYPE_F64_t drho_dx

    drho_dx = (1 / pi) * dSdx(x, SigBE, Rb) / sqrt(x**2 - r**2)

    return drho_dx
################################################################################




################################################################################
# Exponential bulge model (Feng2014)
#-------------------------------------------------------------------------------
cpdef DTYPE_F64_t bulge_vel_feng_2014(DTYPE_F64_t r,
                            DTYPE_F64_t log_rhob0, 
                            DTYPE_F64_t Rb):
    '''
    Function to calculate the bulge velocity at a given galactocentric radius.

    NOTE: Not to be used with traditional disk models - this peaks at a much 
    larger radius than is normally expected for a galaxy.


    PARAMETERS
    ==========

    r : The distance from the center [pc]

    log_rhob0 : log10(central bulge density) log([M_sol/pc^2])

    Rb : The scale radius of the bulge [pc]


    RETURNS
    =======

    Vb : The rotational velocity of the bulge [km/s]
    '''
    cdef DTYPE_F64_t rho_0
    cdef DTYPE_F64_t mass_b
    cdef DTYPE_F64_t vel
    cdef DTYPE_F64_t Vb = 0.0

    if r != 0.0:

        rho_0 = 10.0**log_rhob0

        mass_b = (4.0/3.0) * pi * rho_0 * Rb**3 * (1.0 - exp(-(r/Rb)**3) )
        
        vel = sqrt((G * mass_b * Msun) / (r * 3.086e16))

        Vb = vel / 1000.0

    return Vb
################################################################################



################################################################################
# Exponential bulge model (Sofue 2017)
#-------------------------------------------------------------------------------
cpdef DTYPE_F64_t bulge_vel(DTYPE_F64_t r,
                            DTYPE_F64_t log_rhob0, 
                            #DTYPE_F64_t rho_0, 
                            DTYPE_F64_t Rb):
    '''
    Function to calculate the bulge velocity at a given galactocentric radius.


    PARAMETERS
    ==========

    r : The distance from the center [pc]

    log_rhob0 : log10(central bulge density) log([M_sol/pc^3])

    Rb : The scale radius of the bulge [pc]


    RETURNS
    =======

    Vb : The rotational velocity of the bulge [km/s]
    '''
    cdef DTYPE_F64_t rho_0
    cdef DTYPE_F64_t mass_0
    cdef DTYPE_F64_t x
    cdef DTYPE_F64_t F
    cdef DTYPE_F64_t vel = 0.0
    cdef DTYPE_F64_t Vb 

    if r != 0.0:

        rho_0 = 10.0**log_rhob0

        x = r/Rb

        F = 1.0 - exp(-x) * (1.0 + x + 0.5 * x**2)

        mass_0 = 8.0 * pi * Rb**3 * rho_0
            
        vel = sqrt((G * mass_0 * Msun * F) / (r * 3.086e16))

    Vb = vel / 1000.0

    return Vb
################################################################################




################################################################################
# Disk velocity from Sofue (2013)
#
# Fitting for central surface density
#-------------------------------------------------------------------------------
cpdef DTYPE_F64_t disk_vel(DTYPE_F64_t r, 
                           DTYPE_F64_t SigD, 
                           DTYPE_F64_t Rd):
    '''
    Function to calculate the disk velocity at a given galactocentric radius.


    PARAMETERS
    ==========

    r : The distance from the center [pc]

    SigD : Central surface density for the disk [M_sol/pc^2]

    Rd : The scale radius of the disk [pc]


    RETURNS
    =======

    Vd : The rotational velocity of the disk [km/s]
    '''

    cdef DTYPE_F64_t y
    cdef DTYPE_F64_t bessel_component
    cdef DTYPE_F64_t vel2
    cdef DTYPE_F64_t Vd

    y = r / (2.0 * Rd)

    bessel_component = i0(y) * k0(y) - i1(y) * k1(y)

    vel2 = (4.0 * pi * G * SigD * y**2 * (Rd / (3.086e16)) * Msun) * bessel_component

    Vd = sqrt(vel2) / 1000.0

    return Vd
################################################################################




################################################################################
# Isothermal_halo
# velocity calculated from mass with already evaluated integral
# from density profile (eqn 55, Sofue 2013)
# integral form can be seen from "rotation_curve_functions.py"
#-------------------------------------------------------------------------------
cpdef DTYPE_F64_t halo_vel_iso(DTYPE_F64_t r, 
                               DTYPE_F64_t log_rhoh0, 
                               #DTYPE_F64_t rho0_h, 
                               DTYPE_F64_t Rh):
    '''
    Function to calculate the isothermal halo velocity at a given galactocentric 
    radius.


    PARAMETERS
    ==========

    r : The distance from the center [pc]

    log_rhoh0 : The logarithm of the central surface mass density for the halo 
        [log(M_sol/pc^2)]

    Rh : The scale radius of the halo [pc]


    RETURNS
    =======

    Vh : The rotational velocity of the halo [km/s]
    '''

    cdef DTYPE_F64_t rho0_h
    cdef DTYPE_F64_t Vinf
    cdef DTYPE_F64_t sterm = 0.0
    cdef DTYPE_F64_t Vh

    rho0_h = 10**log_rhoh0

    if r != 0.0:
        sterm = sqrt(1.0 - (Rh/r) * atan2(r,Rh))

    Vinf = sqrt((4.0 * pi * G * rho0_h * Msun * Rh**2) / 3.086e16)/1000.0
    
    Vh = Vinf * sterm

    return Vh
################################################################################




################################################################################
# NFW_halo
# velocity calculated from mass with already evaluated integral
# from density profile (eqn 55, Sofue 2013)
# integral form can be seen from "rotation_curve_functions.py"
#-------------------------------------------------------------------------------
cpdef DTYPE_F64_t halo_vel_NFW(DTYPE_F64_t r, 
                               DTYPE_F64_t log_rhoh0, 
                               #DTYPE_F64_t rho0_h, 
                               DTYPE_F64_t Rh):
    '''
    Function to calculate the NFW halo velocity at a given galactocentric 
    radius.


    PARAMETERS
    ==========

    r : The distance from the center [pc]

    log_rhoh0 : The logarithm of the central surface mass density for the halo 
        [log(M_sol/pc^2)]

    Rh : The scale radius of the halo [pc]


    RETURNS
    =======

    Vh : The rotational velocity of the halo [km/s]
    '''

    cdef DTYPE_F64_t rho0_h
    cdef DTYPE_F64_t halo_mass
    cdef DTYPE_F64_t vel2 = 0.0
    cdef DTYPE_F64_t Vh

    rho0_h = 10**log_rhoh0
    
    halo_mass = 4.0 * pi * rho0_h * Rh**3 * ((Rh/(Rh + r)) + log(Rh + r) - 1.0 - log(Rh))

    if r != 0.0:
        vel2 = G * (halo_mass * Msun) / (r * 3.086e16)

    Vh = sqrt(vel2) / 1000.0

    return Vh
################################################################################




################################################################################
# Burket_halo
# velocity calculated from mass with already evaluated integral
# from density profile (eqn 55, Sofue 2013)
# integral form can be seen from "rotation_curve_functions.py"
#-------------------------------------------------------------------------------
cpdef DTYPE_F64_t halo_vel_bur(DTYPE_F64_t r, 
                               DTYPE_F64_t log_rhoh0, 
                               #DTYPE_F64_t rho0_h, 
                               DTYPE_F64_t Rh):
    '''
    Function to calculate the Burket halo velocity at a given galactocentric 
    radius.


    PARAMETERS
    ==========

    r : The distance from the center [pc]

    log_rhoh0 : The logarithm of the central surface mass density for the halo 
        [log(M_sol/pc^3)]

    Rh : The scale radius of the halo [pc]


    RETURNS
    =======

    Vh : The rotational velocity of the halo [km/s]
    '''

    cdef DTYPE_F64_t rho0_h
    cdef DTYPE_F64_t halo_mass
    cdef DTYPE_F64_t vel2 = 0.0
    cdef DTYPE_F64_t Vh

    rho0_h = 10**log_rhoh0
    
    halo_mass = np.pi * (-rho0_h) * (Rh**3) * (-log(Rh**2 + r**2) \
                                               - 2.0*log(Rh + r)\
                                               + 2.0*atan2(r, Rh)\
                                               + log(Rh**2)\
                                               + 2.0*log(Rh)\
                                               - 2.0*atan2(0.0, Rh))

    if r != 0.0:
        vel2 = G * (halo_mass * Msun) / (r * 3.086e16)

    Vh = sqrt(vel2) / 1000.0

    return Vh
################################################################################




################################################################################
# total Isothermal velocity
#-------------------------------------------------------------------------------
cpdef DTYPE_F64_t vel_tot_iso(DTYPE_F64_t r, 
                              DTYPE_F64_t log_rhob0, 
                              DTYPE_F64_t Rb, 
                              DTYPE_F64_t SigD, 
                              DTYPE_F64_t Rd, 
                              DTYPE_F64_t log_rhoh0, 
                              DTYPE_F64_t Rh):
    '''
    Function to calculate the total velocity with an isothermal halo at a given 
    galactocentric radius.


    PARAMETERS
    ==========

    r : The distance from the center [kpc]

    log_rhob0 : The logarithm of the central surface mass density of the bulge 
        [log(M_sol/pc^2)]

    Rb : The scale radius of the bulge [kpc]

    SigD : The surface mass density of the disk [M_sol/pc^2]

    Rd : The scale radius of the disk [kpc]

    log_rhoh0 : The logarithm of the central surface mass density of the 
        isothermal halo [log(M_sol/pc^2)]

    Rh : The scale radius of the isothermal halo [kpc]


    RETURNS
    =======

    Vtot : The total velocity from the bulge, disk, and isothermal halo [km/s]
    '''
    
    cdef DTYPE_F64_t Vbulge
    cdef DTYPE_F64_t Vdisk
    cdef DTYPE_F64_t Vhalo
    cdef DTYPE_F64_t v2
    cdef DTYPE_F64_t Vtot

    Vbulge = bulge_vel(r * 1000.0, log_rhob0, Rb * 1000.0)
    Vdisk = disk_vel(r * 1000.0, SigD, Rd * 1000.0)
    Vhalo = halo_vel_iso(r * 1000.0, log_rhoh0, Rh * 1000.0)

    v2 = Vbulge**2 + Vdisk**2 + Vhalo**2

    Vtot = sqrt(v2)

    return Vtot
################################################################################




################################################################################
# total NFW velocity
#-------------------------------------------------------------------------------
cpdef DTYPE_F64_t vel_tot_NFW(DTYPE_F64_t r, 
                              DTYPE_F64_t log_rhob0, 
                              DTYPE_F64_t Rb, 
                              DTYPE_F64_t SigD, 
                              DTYPE_F64_t Rd, 
                              DTYPE_F64_t log_rhoh0, 
                              DTYPE_F64_t Rh):
    '''
    Function to calculate the total velocity with an NFW halo at a given 
    galactocentric radius.


    PARAMETERS
    ==========

    r : The distance from the center [kpc]

    log_rhob0 : The logarithm of the central surface mass density of the bulge 
        [log(M_sol/pc^2)]

    Rb : The scale radius of the bulge [kpc]

    SigD : The surface mass density of the disk [M_sol/pc^2]

    Rd : The scale radius of the disk [kpc]

    log_rhoh0 : The logarithm of the central surface mass density of the NFW 
        halo [log(M_sol/pc^2)]

    Rh : The scale radius of the NFW halo [kpc]


    RETURNS
    =======

    Vtot : The total velocity from the bulge, disk, and NFW halo [km/s]
    '''
    
    cdef DTYPE_F64_t Vbulge
    cdef DTYPE_F64_t Vdisk
    cdef DTYPE_F64_t Vhalo
    cdef DTYPE_F64_t v2
    cdef DTYPE_F64_t Vtot

    Vbulge = bulge_vel(r * 1000.0, log_rhob0, Rb * 1000.0)
    Vdisk = disk_vel(r * 1000.0, SigD, Rd * 1000.0)
    Vhalo = halo_vel_NFW(r * 1000.0, log_rhoh0, Rh * 1000.0)

    v2 = Vbulge**2 + Vdisk**2 + Vhalo**2

    Vtot = sqrt(v2)

    return Vtot
################################################################################




################################################################################
# total Burket velocity
#-------------------------------------------------------------------------------
cpdef DTYPE_F64_t vel_tot_bur(DTYPE_F64_t r, 
                              DTYPE_F64_t log_rhob0, 
                              DTYPE_F64_t Rb, 
                              DTYPE_F64_t SigD, 
                              DTYPE_F64_t Rd, 
                              DTYPE_F64_t log_rhoh0, 
                              DTYPE_F64_t Rh):
    '''
    Function to calculate the total velocity with a Burket halo at a given 
    galactocentric radius.


    PARAMETERS
    ==========

    r : The distance from the center [kpc]

    log_rhob0 : The logarithm of the central surface mass density of the bulge 
        [log(M_sol/pc^2)]

    Rb : The scale radius of the bulge [kpc]

    SigD : The surface mass density of the disk [M_sol/pc^2]

    Rd : The scale radius of the disk [kpc]

    log_rhoh0 : The logarithm of the central surface mass density of the Burket 
        halo [log(M_sol/pc^2)]

    Rh : The scale radius of the Burket halo [kpc]


    RETURNS
    =======

    Vtot : The total velocity from the bulge, disk, and Burket halo [km/s]
    '''
    
    cdef DTYPE_F64_t Vbulge
    cdef DTYPE_F64_t Vdisk
    cdef DTYPE_F64_t Vhalo
    cdef DTYPE_F64_t v2
    cdef DTYPE_F64_t Vtot

    Vbulge = bulge_vel(r * 1000.0, log_rhob0, Rb * 1000.0)
    Vdisk = disk_vel(r * 1000.0, SigD, Rd * 1000.0)
    Vhalo = halo_vel_bur(r * 1000.0, log_rhoh0, Rh * 1000.0)

    v2 = Vbulge**2 + Vdisk**2 + Vhalo**2

    Vtot = sqrt(v2)

    return Vtot
################################################################################




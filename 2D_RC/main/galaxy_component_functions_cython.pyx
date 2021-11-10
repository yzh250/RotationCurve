'''
Cythonized versions of some of the functions found in 
galaxy_component_functions.py
'''

cimport cython

cimport numpy as np

import numpy as np

from libc.math cimport sqrt, exp, log

from typedefs cimport DTYPE_F32_t

from scipy.special.cython_special cimport i0, i1, k0, k1




################################################################################
# Constants
#-------------------------------------------------------------------------------
cdef DTYPE_F32_t G = 6.674e-11  # m^3 kg^-1 s^-2

cdef DTYPE_F32_t Msun = 1.989e30  # kg

cdef DTYPE_F32_t pi = np.pi
################################################################################



################################################################################
# Exponential bulge model (Feng2014)
#-------------------------------------------------------------------------------
cpdef DTYPE_F32_t bulge_vel(DTYPE_F32_t r,
                            DTYPE_F32_t log_rhob0, 
                            DTYPE_F32_t Rb):
    '''
    Function to calculate the bulge velocity at a given galactocentric radius.


    PARAMETERS
    ==========

    r : The distance from the center [pc]

    log_rhob0 : log10(central bulge density) log([M_sol/pc^2])

    Rb : The scale radius of the bulge [pc]


    RETURNS
    =======

    Vb : The rotational velocity of the bulge [km/s]
    '''
    cdef DTYPE_F32_t rho_0
    cdef DTYPE_F32_t mass_b
    cdef DTYPE_F32_t vel
    cdef DTYPE_F32_t Vb

    rho_0 = 10.0**log_rhob0

    mass_b = 4.0 * pi * rho_0 * (((-1.0/3.0) * Rb**3 * exp(-(r/Rb)**3) + (1.0/3.0) * (Rb**3)))
    
    vel = sqrt((G * mass_b * Msun) / (r * 3.086e16))

    Vb = vel / 1000.0

    return Vb
################################################################################



################################################################################
# Disk velocity from Sofue (2013)
#
# Fitting for central surface density
#-------------------------------------------------------------------------------
cpdef DTYPE_F32_t disk_vel(DTYPE_F32_t r, 
                          DTYPE_F32_t SigD, 
                          DTYPE_F32_t Rd):
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

    cdef DTYPE_F32_t y
    cdef DTYPE_F32_t bessel_component
    cdef DTYPE_F32_t vel2
    cdef DTYPE_F32_t Vd

    y = r / (2.0 * Rd)

    bessel_component = i0(y) * k0(y) - i1(y) * k1(y)

    vel2 = (4.0 * pi * G * SigD * y**2 * (Rd / (3.086e16)) * Msun) * bessel_component

    Vd = sqrt(vel2) / 1000.0

    return Vd
################################################################################




################################################################################
# NFW_halo
# velocity calculated from mass with already evaluated integral
# from density profile (eqn 55, Sofue 2013)
# integral form can be seen from "rotation_curve_functions.py"
#-------------------------------------------------------------------------------
cpdef DTYPE_F32_t halo_vel_NFW(DTYPE_F32_t r, 
                               DTYPE_F32_t rho0_h, 
                               DTYPE_F32_t Rh):
    '''
    Function to calculate the NFW halo velocity at a given galactocentric 
    radius.


    PARAMETERS
    ==========

    r : The distance from the center [pc]

    rho0_h : The central surface mass density for the halo [M_sol/pc^2]

    Rh : The scale radius of the halo [pc]


    RETURNS
    =======

    Vh : The rotational velocity of the halo [km/s]
    '''

    cdef DTYPE_F32_t halo_mass
    cdef DTYPE_F32_t vel2
    cdef DTYPE_F32_t Vh
    
    halo_mass = 4.0 * pi * rho0_h * Rh**3 * ((Rh/(Rh + r)) + log(Rh + r) - 1 - log(Rh))

    vel2 = G * (halo_mass * Msun) / (r * 3.086e16)

    Vh = sqrt(vel2) / 1000.0

    return Vh
################################################################################




################################################################################
# total NFW velocity
#-------------------------------------------------------------------------------
cpdef DTYPE_F32_t vel_tot_NFW(DTYPE_F32_t r, 
                              DTYPE_F32_t log_rhob0, 
                              DTYPE_F32_t Rb, 
                              DTYPE_F32_t SigD, 
                              DTYPE_F32_t Rd, 
                              DTYPE_F32_t rho0_h, 
                              DTYPE_F32_t Rh):
    '''
    Function to calculate the total velocity with an NFW bulge at a given 
    galactocentric radius.


    PARAMETERS
    ==========

    r : The distance from the center [kpc]

    log_rhob0 : The logarithm of the central surface mass density of the bulge 
        [log(M_sol/pc^2)]

    Rb : The scale radius of the bulge [kpc]

    SigD : The surface mass density of the disk [M_sol/pc^2]

    Rd : The scale radius of the disk [kpc]

    rho0_h : The central surface mass density of the NFW halo [M_sol/pc^2]

    Rh : The scale radius of the NFW halo [kpc]


    RETURNS
    =======

    Vtot : The total velocity from the bulge, disk, and NFW halo [km/s]
    '''

    cdef DTYPE_F32_t Vbulge
    cdef DTYPE_F32_t Vdisk
    cdef DTYPE_F32_t Vhalo
    cdef DTYPE_F32_t v2
    cdef DTYPE_F32_t Vtot

    '''
    r_pc = r * 1000
    Rb_pc = Rb * 1000
    Rd_pc = Rd * 1000
    Rh_pc = Rh * 1000
    '''

    Vbulge = bulge_vel(r * 1000.0, log_rhob0, Rb * 1000.0)
    Vdisk = disk_vel(r * 1000.0, SigD, Rd * 1000.0)
    Vhalo = halo_vel_NFW(r * 1000.0, rho0_h, Rh * 1000.0)

    v2 = Vbulge**2 + Vdisk**2 + Vhalo**2

    Vtot = sqrt(v2)

    return Vtot





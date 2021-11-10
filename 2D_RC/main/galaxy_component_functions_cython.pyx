'''
Cythonized versions of some of the functions found in 
galaxy_component_functions.py
'''

cimport cython

cimport numpy as np

import numpy as np

from libc.math cimport sqrt, exp, log

from typedefs cimport DTYPE_CP128_t, \
                      DTYPE_CP64_t, \
                      DTYPE_F64_t, \
                      DTYPE_F32_t, \
                      DTYPE_B_t, \
                      ITYPE_t, \
                      DTYPE_INT32_t, \
                      DTYPE_INT64_t

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
    
    cdef DTYPE_F32_t rho_0
    cdef DTYPE_F32_t mass_b
    cdef DTYPE_F32_t vel

    rho_0 = 10.0**log_rhob0

    mass_b = 4.0 * pi * rho_0 * (((-1.0/3.0) * Rb**3 * exp(-(r/Rb)**3) + (1.0/3.0) * (Rb**3)))
    
    vel = sqrt((G * mass_b * Msun) / (r * 3.086e16))

    return vel / 1000.0
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

    y = r / (2.0 * Rd)

    bessel_component = i0(y) * k0(y) - i1(y) * k1(y)

    vel2 = (4.0 * pi * G * SigD * y**2 * (Rd / (3.086e16)) * Msun) * bessel_component

    return sqrt(vel2) / 1000.0
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

    cdef DTYPE_F32_t halo_mass
    cdef DTYPE_F32_t vel2
    
    halo_mass = 4.0 * pi * rho0_h * Rh**3 * ((Rh/(Rh + r)) + np.log(Rh + r) - 1 - np.log(Rh))

    vel2 = G * (halo_mass * Msun) / (r * 3.086e16)

    return sqrt(vel2) / 1000.0
################################################################################



'''
################################################################################
# total NFW velocity
#-------------------------------------------------------------------------------
cpdef  vel_tot_NFW(r, params):
    log_rhob0, Rb, SigD, Rd, rho0_h, Rh = params

    r_pc = r * 1000
    Rb_pc = Rb * 1000
    Rd_pc = Rd * 1000
    Rh_pc = Rh * 1000

    Vbulge = bulge_vel(r_pc, log_rhob0, Rb_pc)
    Vdisk = disk_vel(r_pc, SigD, Rd_pc)
    Vhalo = halo_vel_NFW(r_pc, rho0_h, Rh_pc)

    v2 = Vbulge**2 + Vdisk**2 + Vhalo**2

    return np.sqrt(v2)
'''




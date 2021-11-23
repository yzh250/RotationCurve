'''
Cythonized versions of some of the functions found in 
Velocity_Map_Functions.py
'''

cimport cython

cimport numpy as np

import numpy as np

from libc.math cimport sqrt, cos, sin, atan2

from typedefs cimport DTYPE_F64_t, DTYPE_INT64_t

from galaxy_component_functions_cython cimport vel_tot_iso,\
                                               vel_tot_NFW,\
                                               vel_tot_bur

################################################################################
# Isothermal model with bulge
#-------------------------------------------------------------------------------
cpdef np.ndarray rot_incl_iso(shape, 
                              DTYPE_F64_t scale, 
                              list params):
    '''
    Function to calculate the model velocity map given a set of parameters.


    PARAMETERS
    ==========

    shape : tuple
        Shape of velocity map array

    scale : float
        Conversion from spaxels to kpc

    params : list
        Model parameter values


    RETURNS
    =======

    rotated_inclined_map : array
        Model velocity map.  Velocities are in units of km/s.
    '''

    cdef DTYPE_F64_t log_rhob0
    cdef DTYPE_F64_t Rb
    cdef DTYPE_F64_t SigD
    cdef DTYPE_F64_t Rd
    cdef DTYPE_F64_t rho0_h
    cdef DTYPE_F64_t Rh
    cdef DTYPE_F64_t inclination
    cdef DTYPE_F64_t phi
    cdef DTYPE_F64_t center_x
    cdef DTYPE_F64_t center_y
    cdef DTYPE_F64_t vsys
    cdef DTYPE_F64_t[:,:] rotated_inclined_map_memview
    cdef DTYPE_INT64_t i
    cdef DTYPE_INT64_t j
    cdef DTYPE_F64_t x
    cdef DTYPE_F64_t y
    cdef DTYPE_F64_t r
    cdef DTYPE_F64_t theta
    cdef DTYPE_F64_t vTot
    cdef DTYPE_F64_t v
    cdef DTYPE_INT64_t num_rows = shape[0]
    cdef DTYPE_INT64_t num_cols = shape[1]

    log_rhob0, Rb, SigD, Rd, rho0_h, Rh, inclination, phi, center_x, center_y, vsys = params

    rotated_inclined_map = np.zeros(shape, dtype=np.float64)
    rotated_inclined_map_memview = rotated_inclined_map
    
    for i in range(num_rows):
        for j in range(num_cols):

            x =  ((j - center_x)*cos(phi) + sin(phi)*(i - center_y)) / cos(inclination)
            y =  (-(j - center_x)*sin(phi) + cos(phi)*(i - center_y))

            r = scale*sqrt(x**2 + y**2)

            theta = atan2(-x, y)

            vTot = vel_tot_iso(r, log_rhob0, Rb, SigD, Rd, rho0_h, Rh)

            v = vTot*sin(inclination)*cos(theta)

            rotated_inclined_map_memview[i,j] = v + vsys

    return rotated_inclined_map
################################################################################




################################################################################
# NFW model with bulge
#-------------------------------------------------------------------------------
cpdef np.ndarray rot_incl_NFW(shape, 
                              DTYPE_F64_t scale, 
                              list params):
    '''
    Function to calculate the model velocity map given a set of parameters.


    PARAMETERS
    ==========

    shape : tuple
        Shape of velocity map array

    scale : float
        Conversion from spaxels to kpc

    params : list
        Model parameter values


    RETURNS
    =======

    rotated_inclined_map : array
        Model velocity map.  Velocities are in units of km/s.
    '''

    cdef DTYPE_F64_t log_rhob0
    cdef DTYPE_F64_t Rb
    cdef DTYPE_F64_t SigD
    cdef DTYPE_F64_t Rd
    cdef DTYPE_F64_t rho0_h
    cdef DTYPE_F64_t Rh
    cdef DTYPE_F64_t inclination
    cdef DTYPE_F64_t phi
    cdef DTYPE_F64_t center_x
    cdef DTYPE_F64_t center_y
    cdef DTYPE_F64_t vsys
    cdef DTYPE_F64_t[:,:] rotated_inclined_map_memview
    cdef DTYPE_INT64_t i
    cdef DTYPE_INT64_t j
    cdef DTYPE_F64_t x
    cdef DTYPE_F64_t y
    cdef DTYPE_F64_t r
    cdef DTYPE_F64_t theta
    cdef DTYPE_F64_t vTot
    cdef DTYPE_F64_t v
    cdef DTYPE_INT64_t num_rows = shape[0]
    cdef DTYPE_INT64_t num_cols = shape[1]

    log_rhob0, Rb, SigD, Rd, rho0_h, Rh, inclination, phi, center_x, center_y, vsys = params

    rotated_inclined_map = np.zeros(shape, dtype=np.float64)
    rotated_inclined_map_memview = rotated_inclined_map
    
    for i in range(num_rows):
        for j in range(num_cols):

            x =  ((j - center_x)*cos(phi) + sin(phi)*(i - center_y)) / cos(inclination)
            y =  (-(j - center_x)*sin(phi) + cos(phi)*(i - center_y))

            r = scale*sqrt(x**2 + y**2)

            theta = atan2(-x, y)

            vTot = vel_tot_NFW(r, log_rhob0, Rb, SigD, Rd, rho0_h, Rh)
            
            v = vTot*sin(inclination)*cos(theta)

            rotated_inclined_map_memview[i,j] = v + vsys

    return rotated_inclined_map
################################################################################





################################################################################
# Burket model with bulge
#-------------------------------------------------------------------------------
cpdef np.ndarray rot_incl_bur(shape, 
                              DTYPE_F64_t scale, 
                              list params):
    '''
    Function to calculate the model velocity map given a set of parameters.


    PARAMETERS
    ==========

    shape : tuple
        Shape of velocity map array

    scale : float
        Conversion from spaxels to kpc

    params : list
        Model parameter values


    RETURNS
    =======

    rotated_inclined_map : array
        Model velocity map.  Velocities are in units of km/s.
    '''

    cdef DTYPE_F64_t log_rhob0
    cdef DTYPE_F64_t Rb
    cdef DTYPE_F64_t SigD
    cdef DTYPE_F64_t Rd
    cdef DTYPE_F64_t rho0_h
    cdef DTYPE_F64_t Rh
    cdef DTYPE_F64_t inclination
    cdef DTYPE_F64_t phi
    cdef DTYPE_F64_t center_x
    cdef DTYPE_F64_t center_y
    cdef DTYPE_F64_t vsys
    cdef DTYPE_F64_t[:,:] rotated_inclined_map_memview
    cdef DTYPE_INT64_t i
    cdef DTYPE_INT64_t j
    cdef DTYPE_F64_t x
    cdef DTYPE_F64_t y
    cdef DTYPE_F64_t r
    cdef DTYPE_F64_t theta
    cdef DTYPE_F64_t vTot
    cdef DTYPE_F64_t v
    cdef DTYPE_INT64_t num_rows = shape[0]
    cdef DTYPE_INT64_t num_cols = shape[1]

    log_rhob0, Rb, SigD, Rd, rho0_h, Rh, inclination, phi, center_x, center_y, vsys = params

    rotated_inclined_map = np.zeros(shape, dtype=np.float64)
    rotated_inclined_map_memview = rotated_inclined_map
    
    for i in range(num_rows):
        for j in range(num_cols):

            x =  ((j - center_x)*cos(phi) + sin(phi)*(i - center_y)) / cos(inclination)
            y =  (-(j - center_x)*sin(phi) + cos(phi)*(i - center_y))

            r = scale*sqrt(x**2 + y**2)

            theta = atan2(-x, y)

            vTot = vel_tot_bur(r, log_rhob0, Rb, SigD, Rd, rho0_h, Rh)
            
            v = vTot*sin(inclination)*cos(theta)

            rotated_inclined_map_memview[i,j] = v + vsys

    return rotated_inclined_map
################################################################################



################################################################################
# Functions for constructing velocity maps
################################################################################

################################################################################
# Importing modules and functions
#-------------------------------------------------------------------------------
import time

import numpy as np
import numpy.ma as ma


from galaxy_component_functions_cython import bulge_vel,\
                                              disk_vel,\
                                              halo_vel_iso,\
                                              halo_vel_NFW,\
                                              halo_vel_bur,\
                                              vel_tot_iso,\
                                              vel_tot_NFW,\
                                              vel_tot_bur

from Velocity_Map_Functions_cython import rot_incl_iso as rot_incl_iso_cython 
from Velocity_Map_Functions_cython import rot_incl_NFW as rot_incl_NFW_cython 
from Velocity_Map_Functions_cython import rot_incl_bur as rot_incl_bur_cython                     
################################################################################

################################################################################
# borrowed from Prof. Kelly Douglass
#-------------------------------------------------------------------------------
def find_phi(center_coords, phi_angle, vel_map):
    '''
    Find a point along the semi-major axis that has data to determine if phi
    needs to be adjusted.  (This is necessary because the positive y-axis is
    defined as being along the semi-major axis of the positive velocity side of
    the velocity map.)


    PARAMETERS
    ==========

    center_coords : tuple
        Coordinates of the center of the galaxy

    phi_angle : float
        Initial rotation angle of the galaxy, E of N.  Units are degrees.

    vel_map : masked ndarray of shape (n,n)
        Masked H-alpha velocity map


    RETURNS
    =======

    phi_adjusted : float
        Rotation angle of the galaxy, E of N, that points along the positive
        velocity sector.  Units are radians.
    '''

    # Convert phi_angle to radians
    phi = phi_angle * np.pi / 180.

    #phi = phi_angle 

    # Extract "systemic" velocity (velocity at center spaxel)
    v_sys = vel_map[center_coords]

    print(center_coords)

    print(phi)

    f = 0.4

    checkpoint_masked = True

    start_time = time.time()

    while checkpoint_masked:
        delta_x = int(center_coords[1] * f)
        delta_y = int(delta_x / np.tan(phi))
        semi_major_axis_spaxel = np.subtract(center_coords, (-delta_y, delta_x))

        '''
        print(center_coords)
        print(semi_major_axis_spaxel)
        '''

        print(semi_major_axis_spaxel)        

        for i in range(len(semi_major_axis_spaxel)):
            print(semi_major_axis_spaxel)
            if semi_major_axis_spaxel[i] < 0:
                semi_major_axis_spaxel[i] = 0
            elif semi_major_axis_spaxel[i] >= vel_map.shape[i]:
                semi_major_axis_spaxel[i] = vel_map.shape[i] - 1
            #elif time.time() - start_time >= 1000:
                #lougoycheckpoint_masked = False


        # Check value along semi-major axis
        if vel_map.mask[tuple(semi_major_axis_spaxel)] == 0:
            checkpoint_masked = False
        #elif time.time() - start_time >= 1000:
            #checkpoint_masked = False
        else:
            f *= 0.9

    print(semi_major_axis_spaxel)

    if vel_map[tuple(semi_major_axis_spaxel)] - v_sys < 0:
        phi_adjusted = phi + np.pi
    else:
        phi_adjusted = phi

    return phi_adjusted
################################################################################


################################################################################
# Isothermal model with bulge
#-------------------------------------------------------------------------------
def rot_incl_iso(shape, scale, params):

    log_rhob0, Rb, SigD, Rd, rho0_h, Rh, inclination, phi, center_x, center_y, vsys = params

    rotated_inclined_map = np.zeros(shape)
    
    for i in range(shape[0]):
        for j in range(shape[1]):

            x =  ((j-center_x)*np.cos(phi) + np.sin(phi)*(i-center_y))/np.cos(inclination)
            y =  (-(j-center_x)*np.sin(phi) + np.cos(phi)*(i-center_y))

            r = np.sqrt(x**2 + y**2)
            
            theta = np.arctan2(-x,y)
            
            r_in_kpc = r*scale
            
            v_rot = vel_tot_iso(r_in_kpc, [log_rhob0, Rb, SigD, Rd, rho0_h, Rh])
            
            v = v_rot*np.sin(inclination)*np.cos(theta)
            
            rotated_inclined_map[i,j] = v + vsys

    return rotated_inclined_map
################################################################################


'''
################################################################################
# Isothermal model without bulge
#-------------------------------------------------------------------------------
def rot_incl_iso_nb(shape,scale,params):

    SigD, Rd, Vinf, Rh,inclination,phi,center_x,center_y,vsys = params
    rotated_inclined_map = np.zeros(shape)
    
    for i in range(shape[0]):
        for j in range(shape[1]):
            x = (-(i - center_x) * np.sin(phi) - np.cos(phi) * (j - center_y)) / np.cos(inclination)
            y = ((i - center_x) * np.cos(phi) - np.sin(phi) * (j - center_y))
            r = np.sqrt(x ** 2 + y ** 2)
            theta = np.arctan2(x, y)
            r_in_kpc = r * scale
            v = vel_tot_iso_nb(r_in_kpc,[SigD, Rd, Vinf, Rh])*np.sin(inclination)*np.cos(theta)
            rotated_inclined_map[i,j] = v + vsys

    return rotated_inclined_map
################################################################################
'''


################################################################################
# NFW model with bulge
#-------------------------------------------------------------------------------
def rot_incl_NFW(shape,scale,params):

    log_rhob0, Rb, SigD, Rd, rho0_h, Rh, inclination, phi, center_x, center_y, vsys = params
    rotated_inclined_map = np.zeros(shape)
    
    for i in range(shape[0]):
        for j in range(shape[1]):
            x =  ((j-center_x)*np.cos(phi) + np.sin(phi)*(i-center_y))/np.cos(inclination)
            y =  (-(j-center_x)*np.sin(phi) + np.cos(phi)*(i-center_y))
            r = np.sqrt(x**2 + y**2)
            theta = np.arctan2(-x,y)
            r_in_kpc = r*scale
            v = vel_tot_NFW(r_in_kpc,[log_rhob0,Rb,SigD,Rd,rho0_h,Rh])*np.sin(inclination)*np.cos(theta)
            rotated_inclined_map[i,j] = v + vsys

    return rotated_inclined_map
################################################################################


'''
################################################################################
# NFW model without bulge
#-------------------------------------------------------------------------------
def rot_incl_NFW_nb(shape,scale,params):

    SigD,r_d,rho_h,r_h,inclination,phi,center_x,center_y, vsys = params
    rotated_inclined_map = np.zeros(shape)
    
    for i in range(shape[0]):
        for j in range(shape[1]):
            x = (-(i - center_x) * np.sin(phi) - np.cos(phi) * (j - center_y)) / np.cos(inclination)
            y = ((i - center_x) * np.cos(phi) - np.sin(phi) * (j - center_y))
            r = np.sqrt(x ** 2 + y ** 2)
            theta = np.arctan2(x, y)
            r_in_kpc = r * scale
            v = vel_tot_NFW_nb(r_in_kpc,[SigD,r_d,rho_h,r_h])*np.sin(inclination)*np.cos(theta)
            rotated_inclined_map[i,j] = v + vsys

    return rotated_inclined_map
################################################################################
'''


################################################################################
# Burket model with bulge
#-------------------------------------------------------------------------------
def rot_incl_bur(shape,scale,params):

    log_rhob0, Rb, SigD, Rd, rho0_h, Rh, inclination, phi, center_x, center_y, vsys = params
    rotated_inclined_map = np.zeros(shape) 
    
    for i in range(shape[0]):
        for j in range(shape[1]):
            x =  ((j-center_x)*np.cos(phi) + np.sin(phi)*(i-center_y))/np.cos(inclination)
            y =  (-(j-center_x)*np.sin(phi) + np.cos(phi)*(i-center_y))
            r = np.sqrt(x**2 + y**2)
            theta = np.arctan2(-x,y)
            r_in_kpc = r*scale
            v = vel_tot_bur(r_in_kpc,[log_rhob0,Rb,SigD,Rd,rho0_h,Rh])*np.sin(inclination)*np.cos(theta)
            rotated_inclined_map[i,j] = v + vsys
             
    return rotated_inclined_map
################################################################################

'''
################################################################################
# Burket model without bulge
#-------------------------------------------------------------------------------
def rot_incl_bur_nb(shape,scale,params):
    SigD,r_d,rho_h,r_h,inclination,phi,center_x,center_y, vsys = params
    rotated_inclined_map = np.zeros(shape)
    
    for i in range(shape[0]):
        for j in range(shape[1]):
            x = (-(i - center_x) * np.sin(phi) - np.cos(phi) * (j - center_y)) / np.cos(inclination)
            y = ((i - center_x) * np.cos(phi) - np.sin(phi) * (j - center_y))
            r = np.sqrt(x ** 2 + y ** 2)
            theta = np.arctan2(x, y)
            r_in_kpc = r * scale
            v = vel_tot_bur_nb(r_in_kpc,[SigD,r_d,rho_h,r_h])*np.sin(inclination)*np.cos(theta)
            rotated_inclined_map[i,j] = v + vsys
    return rotated_inclined_map

##############################################################################
'''


################################################################################
# Loglikelihood Functions
#-------------------------------------------------------------------------------
# Isothermal model with bulge
#-------------------------------------------------------------------------------
"""
def loglikelihood_iso(params, scale, shape, vdata, inv_sigma2):

    # Construct the model
    model = rot_incl_iso(shape, scale, params)
    logL = -0.5 * ma.sum((vdata - model) ** 2 * inv_sigma2 - ma.log(inv_sigma2))
    #if params[3] >= params[5]:
        #logL += 1e7
    #elif params[1] >= params[5]:
        #logL += 1e7
    return logL

def nloglikelihood_iso(params, scale, shape, vdata, ivar):
    return -loglikelihood_iso(params, scale, shape, vdata, ivar)
"""
#-------------------------------------------------------------------------------
# Isothermal no bulge
#-------------------------------------------------------------------------------
"""
def loglikelihood_iso_nb(params, scale, shape, vdata, inv_sigma2):

    # Construct the model
    model = rot_incl_iso_nb(shape, scale, params)
    logL = -0.5 * ma.sum((vdata - model) ** 2 * inv_sigma2 - ma.log(inv_sigma2))
    return logL

def nloglikelihood_iso_nb(params, scale, shape, vdata, ivar):
    return -loglikelihood_iso_nb(params, scale, shape, vdata, ivar)
"""
#-------------------------------------------------------------------------------
# Isothermal model flat
#-------------------------------------------------------------------------------
def loglikelihood_iso_flat_constraints(params, 
                                       scale, 
                                       shape, 
                                       vdata_flat, 
                                       ivar_flat, 
                                       mask):

    #print('Best-fit values in loglikelihood_iso_flat:', params)
    params = np.ndarray.tolist(params)

    ############################################################################
    # Construct the model
    #---------------------------------------------------------------------------
    #model = rot_incl_iso(shape, scale, params)
    model = rot_incl_iso_cython(shape, scale, params)
    model_masked = ma.array(model, mask=mask)
    model_flat = model_masked.compressed()
    ############################################################################
    
    
    logL = -0.5 * np.sum((vdata_flat - model_flat)**2 * ivar_flat \
                         - np.log(ivar_flat))
    
    if params[3] >= params[5]:
        logL -= 1e10
    elif params[1] >= params[5]:
        logL -= 1e10
    elif params[1] >= params[3]:
        logL -= 1e10
    
    return logL



def nloglikelihood_iso_flat_constraints(params, 
                                        scale, 
                                        shape, 
                                        vdata_flat, 
                                        ivar_flat, 
                                        mask):
    return -loglikelihood_iso_flat_constraints(params, 
                                               scale, 
                                               shape, 
                                               vdata_flat, 
                                               ivar_flat, 
                                               mask)




def loglikelihood_iso_flat(params, scale, shape, vdata_flat, ivar_flat, mask):

    #print('Best-fit values in loglikelihood_iso_flat:', params)
    params = np.ndarray.tolist(params)

    ############################################################################
    # Construct the model
    #---------------------------------------------------------------------------
    #model = rot_incl_iso(shape, scale, params)
    model = rot_incl_iso_cython(shape, scale, params)
    model_masked = ma.array(model, mask=mask)
    model_flat = model_masked.compressed()
    ############################################################################
    
    
    logL = -0.5 * np.sum((vdata_flat - model_flat)**2 * ivar_flat \
                         - np.log(ivar_flat))
    
    return logL




def nloglikelihood_iso_flat(params, scale, shape, vdata_flat, ivar_flat, mask):
    return -loglikelihood_iso_flat(params, 
                                   scale, 
                                   shape, 
                                   vdata_flat, 
                                   ivar_flat, 
                                   mask)
#-------------------------------------------------------------------------------
# Isothermal no bulge flat
#-------------------------------------------------------------------------------
"""
def loglikelihood_iso_flat_nb(params, scale, shape, vdata_flat, ivar_flat, mask):

    # Construct the model
    model = rot_incl_iso_nb(shape, scale, params)
    model_masked = ma.array(model, mask=mask)
    model_flat = model_masked.compressed()

    logL = -0.5 * ma.sum((vdata_flat - model_flat) ** 2 * ivar_flat - np.log(ivar_flat))

    return logL

def nloglikelihood_iso_flat_nb(params, scale, shape, vdata_flat, ivar_flat, mask):
    return -loglikelihood_iso_flat_nb(params, scale, shape, vdata_flat, ivar_flat, mask)
"""
#-------------------------------------------------------------------------------

################################################################################
# NFW model with bulge
#-------------------------------------------------------------------------------
"""
def loglikelihood_NFW(params, scale, shape, vdata, inv_sigma2):

    # Construct the model
    model = rot_incl_NFW(shape, scale, params)
    logL = -0.5 * ma.sum((vdata - model) ** 2 * inv_sigma2 - ma.log(inv_sigma2))
    #if params[3] >= params[5]:
        #logL += 1e7
    #elif params[1] >= params[5]:
        #logL += 1e7
    return logL

def nloglikelihood_NFW(params, scale, shape, vdata, ivar):
    return -loglikelihood_NFW(params, scale, shape, vdata, ivar)
"""
#-------------------------------------------------------------------------------
# NFW no bulge
#-------------------------------------------------------------------------------
"""
def loglikelihood_NFW_nb(params, scale, shape, vdata, inv_sigma2):

    # Construct the model
    model = rot_incl_NFW_nb(shape, scale, params)
    logL = -0.5 * ma.sum((vdata - model) ** 2 * inv_sigma2 - ma.log(inv_sigma2))
    return logL

def nloglikelihood_NFW_nb(params, scale, shape, vdata, ivar):
    return -loglikelihood_NFW_nb(params, scale, shape, vdata, ivar)
"""
#-------------------------------------------------------------------------------
# NFW model flat
#-------------------------------------------------------------------------------
def loglikelihood_NFW_flat_constraints(params, 
                                       scale, 
                                       shape, 
                                       vdata_flat, 
                                       ivar_flat, 
                                       mask):
    #print('Best-fit values in loglikelihood_NFW_flat:', params)

    params = np.ndarray.tolist(params)
    ############################################################################
    # Construct the model
    #---------------------------------------------------------------------------
    #model = rot_incl_NFW(shape, scale, params)
    model = rot_incl_NFW_cython(shape, scale, params)
    model_masked = ma.array(model, mask=mask)
    model_flat = model_masked.compressed()
    ############################################################################
    
    
    logL = -0.5 * np.sum((vdata_flat - model_flat)**2 * ivar_flat \
                         - np.log(ivar_flat))
    
    if params[3] >= params[5]:
        logL -= 1e10
    elif params[1] >= params[5]:
        logL -= 1e10
    elif params[1] >= params[3]:
        logL -= 1e10

    return logL



def nloglikelihood_NFW_flat_constraints(params, scale, shape, vdata_flat, ivar_flat, mask):
    return -loglikelihood_NFW_flat_constraints(params, scale, shape, vdata_flat, ivar_flat, mask)



def loglikelihood_NFW_flat(params, scale, shape, vdata_flat, ivar_flat, mask):
    #print('Best-fit values in loglikelihood_NFW_flat:', params)

    params = np.ndarray.tolist(params)
    ############################################################################
    # Construct the model
    #---------------------------------------------------------------------------
    #model = rot_incl_NFW(shape, scale, params)
    model = rot_incl_NFW_cython(shape, scale, params)
    model_masked = ma.array(model, mask=mask)
    model_flat = model_masked.compressed()
    ############################################################################
    
    
    logL = -0.5 * np.sum((vdata_flat - model_flat)**2 * ivar_flat \
                         - np.log(ivar_flat))

    return logL



def nloglikelihood_NFW_flat(params, scale, shape, vdata_flat, ivar_flat, mask):
    return -loglikelihood_NFW_flat(params, scale, shape, vdata_flat, ivar_flat, mask)
#-------------------------------------------------------------------------------
# NFW no bulge flat
#-------------------------------------------------------------------------------
"""
def loglikelihood_NFW_flat_nb(params, scale, shape, vdata_flat, ivar_flat, mask):

    # Construct the model
    model = rot_incl_NFW_nb(shape, scale, params)
    model_masked = ma.array(model, mask=mask)
    model_flat = model_masked.compressed()
    logL = -0.5 * ma.sum((vdata_flat - model_flat) ** 2 * ivar_flat - np.log(ivar_flat))
    return logL

def nloglikelihood_NFW_flat_nb(params, scale, shape, vdata_flat, ivar_flat, mask):
    return -loglikelihood_NFW_flat_nb(params, scale, shape, vdata_flat, ivar_flat, mask)
"""
#-------------------------------------------------------------------------------
# Burket model with bulge
#-------------------------------------------------------------------------------
"""
def loglikelihood_bur(params, scale, shape, vdata, inv_sigma2):

    # Construct the model
    model = rot_incl_bur(shape, scale, params)
    logL = -0.5 * ma.sum((vdata - model) ** 2 * inv_sigma2 - ma.log(inv_sigma2))
    #if params[3] >= params[5]:
        #logL += 1e7
    #elif params[1] >= params[5]:
        #logL += 1e7
    return logL

def nloglikelihood_bur(params, scale, shape, vdata, ivar):
    return -loglikelihood_bur(params, scale, shape, vdata, ivar)
"""
#-------------------------------------------------------------------------------
# Burket no bulge
#-------------------------------------------------------------------------------
"""
def loglikelihood_bur_nb(params, scale, shape, vdata, inv_sigma2):

    # Construct the model
    model = rot_incl_bur_nb(shape, scale, params)
    logL = -0.5 * ma.sum((vdata - model) ** 2 * inv_sigma2 - ma.log(inv_sigma2))
    return logL

def nloglikelihood_bur_nb(params, scale, shape, vdata, ivar):
    return -loglikelihood_bur_nb(params, scale, shape, vdata, ivar)
"""
#-------------------------------------------------------------------------------
# Burket model flat
#-------------------------------------------------------------------------------
def loglikelihood_bur_flat_constraints(params, scale, shape, vdata_flat, ivar_flat, mask):

    #print('Best-fit values in loglikelihood_iso_flat:', params)

    params = np.ndarray.tolist(params)
    ############################################################################
    # Construct the model
    #---------------------------------------------------------------------------
    #model = rot_incl_bur(shape, scale, params)
    model = rot_incl_bur_cython(shape, scale, params)
    model_masked = ma.array(model, mask=mask)
    model_flat = model_masked.compressed()
    ############################################################################
    
    
    logL = -0.5 * np.sum((vdata_flat - model_flat)**2 * ivar_flat \
                         - np.log(ivar_flat))
    
    if params[3] >= params[5]:
        logL -= 1e10
    elif params[1] >= params[5]:
        logL -= 1e10
    elif params[1] >= params[3]:
        logL -= 1e10

    return logL



def nloglikelihood_bur_flat_constraints(params, scale, shape, vdata_flat, ivar_flat, mask):
    return -loglikelihood_bur_flat_constraints(params, scale, shape, vdata_flat, ivar_flat, mask)



def loglikelihood_bur_flat(params, scale, shape, vdata_flat, ivar_flat, mask):

    #print('Best-fit values in loglikelihood_iso_flat:', params)

    params = np.ndarray.tolist(params)
    ############################################################################
    # Construct the model
    #---------------------------------------------------------------------------
    #model = rot_incl_bur(shape, scale, params)
    model = rot_incl_bur_cython(shape, scale, params)
    model_masked = ma.array(model, mask=mask)
    model_flat = model_masked.compressed()
    ############################################################################
    
    
    logL = -0.5 * np.sum((vdata_flat - model_flat)**2 * ivar_flat \
                         - np.log(ivar_flat))

    return logL



def nloglikelihood_bur_flat(params, scale, shape, vdata_flat, ivar_flat, mask):
    return -loglikelihood_bur_flat(params, scale, shape, vdata_flat, ivar_flat, mask)
#-------------------------------------------------------------------------------
# Burket no bulge flat
#-------------------------------------------------------------------------------
"""
def loglikelihood_bur_flat_nb(params, scale, shape, vdata_flat, ivar_flat, mask):

    # Construct the model
    model = rot_incl_bur_nb(shape, scale, params)
    model_masked = ma.array(model, mask=mask)
    model_flat = model_masked.compressed()
    logL = -0.5 * ma.sum((vdata_flat - model_flat) ** 2 * ivar_flat - np.log(ivar_flat))
    return logL

def nloglikelihood_bur_flat_nb(params, scale, shape, vdata_flat, ivar_flat, mask):
    return -loglikelihood_bur_flat_nb(params, scale, shape, vdata_flat, ivar_flat, mask)
"""
##############################################################################
################################################################################
# Functions for constructing velocity maps
################################################################################



################################################################################
# Importing modules and functions
#-------------------------------------------------------------------------------
import numpy as np
import numpy.ma as ma

from rotation_curve_functions import vel_b, \
                                     disk_vel, \
                                     vel_h_iso, \
                                     vel_h_NFW, \
                                     vel_h_Burket, \
                                     v_tot_iso, \
                                     v_tot_iso_nb, \
                                     v_tot_NFW, \
                                     v_tot_NFW_nb, \
                                     v_tot_Burket, \
                                     v_tot_Burket_nb
################################################################################




################################################################################
# Isothermal model with bulge
#-------------------------------------------------------------------------------
def rot_incl_iso(shape, scale, params):

    A, Vin, SigD, Rd, Vinf, Rh, inclination, phi, center_x, center_y = params
    #print('A in rot_incl_iso:', A)

    rotated_inclined_map = np.zeros(shape)
    
    for i in range(shape[0]):
        for j in range(shape[1]):

            xb = ((i-center_x)*np.cos(np.pi/2) - np.sin(np.pi/2)*(j-center_y))
            yb = ((i-center_x)*np.sin(np.pi/2) + np.cos(np.pi/2)*(j-center_y))

            x = (xb*np.cos(phi) - yb*np.sin(phi))/np.cos(inclination)
            y = (xb*np.sin(phi) + yb*np.cos(phi))

            r = np.sqrt(x**2 + y**2)

            theta = np.arctan2(x,y)

            r_in_kpc = r*scale

            v = v_tot_iso(r_in_kpc,[A, Vin, SigD, Rd, Vinf, Rh])*np.sin(inclination)*np.cos(theta)

            rotated_inclined_map[i,j] = v

    return rotated_inclined_map
################################################################################




################################################################################
# Isothermal model without bulge
#-------------------------------------------------------------------------------
def rot_incl_iso_nb(shape,scale,params):

    SigD, Rd, Vinf, Rh,inclination,phi,center_x,center_y = params

    rotated_inclined_map = np.zeros(shape)
    
    for i in range(shape[0]):
        for j in range(shape[1]):

            xb = ((i-center_x)*np.cos(np.pi/2) - np.sin(np.pi/2)*(j-center_y))
            yb = ((i-center_x)*np.sin(np.pi/2) + np.cos(np.pi/2)*(j-center_y))

            x = (xb*np.cos(phi) - yb*np.sin(phi))/np.cos(inclination)
            y = (xb*np.sin(phi) + yb*np.cos(phi))

            r = np.sqrt(x**2 + y**2)

            theta = np.arctan2(x,y)

            r_in_kpc = r*scale

            v = v_tot_iso_nb(r_in_kpc,[SigD, Rd, Vinf, Rh])*np.sin(inclination)*np.cos(theta)

            rotated_inclined_map[i,j] = v

    return rotated_inclined_map
################################################################################





################################################################################
# NFW model with bulge
#-------------------------------------------------------------------------------
def rot_incl_NFW(shape,scale,params):
    A, Vin,SigD,r_d,rho_h,r_h,inclination,phi,center_x,center_y = params
    rotated_inclined_map = np.zeros(shape)
    
    for i in range(shape[0]):
        for j in range(shape[1]):
            xb = ((i-center_x)*np.cos(np.pi/2) - np.sin(np.pi/2)*(j-center_y))
            yb = ((i-center_x)*np.sin(np.pi/2) + np.cos(np.pi/2)*(j-center_y))
            x = (xb*np.cos(phi) - yb*np.sin(phi))/np.cos(inclination)
            y = (xb*np.sin(phi) + yb*np.cos(phi))
            r = np.sqrt(x**2+y**2)
            theta = np.arctan2(x,y)
            r_in_kpc = r*scale
            v = v_tot_NFW(r_in_kpc,[A,Vin,SigD,r_d,rho_h,r_h])*np.sin(inclination)*np.cos(theta)
            rotated_inclined_map[i,j] = v
    return rotated_inclined_map

# NFW model without bulge

def rot_incl_NFW_nb(shape,scale,params):
    SigD,r_d,rho_h,r_h,inclination,phi,center_x,center_y = params
    rotated_inclined_map = np.zeros(shape)
    
    for i in range(shape[0]):
        for j in range(shape[1]):
            xb = ((i-center_x)*np.cos(np.pi/2) - np.sin(np.pi/2)*(j-center_y))
            yb = ((i-center_x)*np.sin(np.pi/2) + np.cos(np.pi/2)*(j-center_y))
            x = (xb*np.cos(phi) - yb*np.sin(phi))/np.cos(inclination)
            y = (xb*np.sin(phi) + yb*np.cos(phi))
            r = np.sqrt(x**2+y**2)
            theta = np.arctan2(x,y)
            r_in_kpc = r*scale
            v = v_tot_NFW_nb(r_in_kpc,[SigD,r_d,rho_h,r_h])*np.sin(inclination)*np.cos(theta)
            rotated_inclined_map[i,j] = v
    return rotated_inclined_map

# Burket model with bulge

def rot_incl_bur(shape,scale,params):
    A, Vin,SigD,r_d,rho_h,r_h,inclination,phi,center_x,center_y = params
    rotated_inclined_map = np.zeros(shape)
    
    for i in range(shape[0]):
        for j in range(shape[1]):
            xb = ((i-center_x)*np.cos(np.pi/2) - np.sin(np.pi/2)*(j-center_y))
            yb = ((i-center_x)*np.sin(np.pi/2) + np.cos(np.pi/2)*(j-center_y))
            x = (xb*np.cos(phi) - yb*np.sin(phi))/np.cos(inclination)
            y = (xb*np.sin(phi) + yb*np.cos(phi))
            r = np.sqrt(x**2+y**2)
            theta = np.arctan2(x,y)
            r_in_kpc = r*scale
            v = v_tot_Burket(r_in_kpc,[A,Vin,SigD,r_d,rho_h,r_h])*np.sin(inclination)*np.cos(theta)
            rotated_inclined_map[i,j] = v
    return rotated_inclined_map

# Burket model without bulge

def rot_incl_bur_nb(shape,scale,params):
    SigD,r_d,rho_h,r_h,inclination,phi,center_x,center_y = params
    rotated_inclined_map = np.zeros(shape)
    
    for i in range(shape[0]):
        for j in range(shape[1]):
            xb = ((i-center_x)*np.cos(np.pi/2) - np.sin(np.pi/2)*(j-center_y))
            yb = ((i-center_x)*np.sin(np.pi/2) + np.cos(np.pi/2)*(j-center_y))
            x = (xb*np.cos(phi) - yb*np.sin(phi))/np.cos(inclination)
            y = (xb*np.sin(phi) + yb*np.cos(phi))
            r = np.sqrt(x**2+y**2)
            theta = np.arctan2(x,y)
            r_in_kpc = r*scale
            v = v_tot_Burket_nb(r_in_kpc,[SigD,r_d,rho_h,r_h])*np.sin(inclination)*np.cos(theta)
            rotated_inclined_map[i,j] = v
    return rotated_inclined_map

##############################################################################
# Loglikelihood Functions
#-----------------------------------------------------------------------------

# Isothermal model with bulge

def loglikelihood_iso(params, scale, shape, vdata, inv_sigma2):

    # Construct the model
    model = rot_incl_iso(shape, scale, params)
    
    logL = -0.5 * ma.sum((vdata - model) ** 2 * inv_sigma2 - ma.log(inv_sigma2))

    return logL

def nloglikelihood_iso(params, scale, shape, vdata, ivar):
    return -loglikelihood_iso(params, scale, shape, vdata, ivar)

# Isothermal model flat

def loglikelihood_iso_flat(params, scale, shape, vdata, inv_sigma2, mask):

    #print('Best-fit values in loglikelihood_iso_flat:', params)

    # Construct the model
    model = rot_incl_iso(shape, scale, params)
    model_masked = ma.array(model, mask=mask)
    model_flat = model_masked.compressed()

    logL = -0.5 * ma.sum((vdata - model_flat) ** 2 * inv_sigma2 - np.log(inv_sigma2))

    return logL

# NFW model with bulge

def loglikelihood_NFW(params, scale, shape, vdata, inv_sigma2):

    # Construct the model
    model = rot_incl_NFW(shape, scale, params)

    logL = -0.5 * ma.sum((vdata - model) ** 2 * inv_sigma2 - np.log(inv_sigma2))

    return logL

def nloglikelihood_NFW(params, scale, shape, vdata, ivar):
    return -loglikelihood_NFW(params, scale, shape, vdata, ivar)

# NFW model flat

def loglikelihood_NFW_flat(params, scale, shape, vdata, inv_sigma2, mask):

    # Construct the model
    model = rot_incl_NFW(shape, scale, params)
    model_masked = ma.array(model, mask=mask)
    model_flat = model_masked.compressed()

    logL = -0.5 * ma.sum((vdata - model_flat) ** 2 * inv_sigma2 - np.log(inv_sigma2))

    return logL

# Burket model with bulge

def loglikelihood_bur(params, scale, shape, vdata, inv_sigma2):

    # Construct the model
    model = rot_incl_bur(shape, scale, params)

    logL = -0.5 * ma.sum((vdata - model) ** 2 * inv_sigma2 - np.log(inv_sigma2))

    return logL

def nloglikelihood_bur(params, scale, shape, vdata, ivar):
    return -loglikelihood_bur(params, scale, shape, vdata, ivar)

# Burket model flat

def loglikelihood_bur_flat(params, scale, shape, vdata, inv_sigma2, mask):

    # Construct the model
    model = rot_incl_bur(shape, scale, params)
    model_masked = ma.array(model, mask=mask)
    model_flat = model_masked.compressed()

    logL = -0.5 * ma.sum((vdata - model_flat) ** 2 * inv_sigma2 - ma.log(inv_sigma2))
    
    return logL

##############################################################################
##############################################################################
# Functions for constructing velocity maps
#-----------------------------------------------------------------------------

# Importing modules and functions
import numpy as np
import numpy.ma as ma
from rotation_curve_functions import vel_b, disk_vel, vel_h_iso, vel_h_NFW, vel_h_Burket, vel_co_iso, vel_co_iso_nb, vel_co_NFW, vel_co_NFW_nb, vel_co_Burket, vel_co_Burket_nb

# Isothermal model with bulge

def rot_incl_iso(shape,scale,params):
    A, Vin, SigD, Rd, Vinf, Rh,inclination,phi,center_x,center_y = params
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
            v = vel_co_iso(r_in_kpc,[A, Vin, SigD, Rd, Vinf, Rh])*np.sin(inclination)*np.cos(theta)
            rotated_inclined_map[i,j] = v
    return rotated_inclined_map

# Isothermal model without bulge

def rot_incl_iso_nb(shape,scale,params):
    SigD, Rd, Vinf, Rh,inclination,phi,center_x,center_y = params
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
            v = vel_co_iso_nb(r_in_kpc,[SigD, Rd, Vinf, Rh])*np.sin(inclination)*np.cos(theta)
            rotated_inclined_map[i,j] = v
    return rotated_inclined_map

# NFW model with bulge

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
            v = vel_co_NFW(r_in_kpc,[A,Vin,SigD,r_d,rho_h,r_h])*np.sin(inclination)*np.cos(theta)
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
            v = vel_co_NFW_nb(r_in_kpc,[SigD,r_d,rho_h,r_h])*np.sin(inclination)*np.cos(theta)
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
            v = vel_co_Burket(r_in_kpc,[A,Vin,SigD,r_d,rho_h,r_h])*np.sin(inclination)*np.cos(theta)
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
            v = vel_co_Burket_nb(r_in_kpc,[SigD,r_d,rho_h,r_h])*np.sin(inclination)*np.cos(theta)
            rotated_inclined_map[i,j] = v
    return rotated_inclined_map

##############################################################################
# Loglikelihood Functions
#-----------------------------------------------------------------------------

# Isothermal model with bulge

def loglikelihood_iso(params, scale, shape, vdata, ivar):
    # Construct the model
    model = rot_incl_iso(shape, scale, params)
    inv_sigma2 = ivar
    logL = -0.5 * ma.sum((vdata - model) ** 2 * inv_sigma2 - np.log(inv_sigma2))
    return logL

def nloglikelihood_iso(params, scale, shape, vdata, ivar):
    return -loglikelihood_iso(params, scale, shape, vdata, ivar)

# Isothermal model without bulge

def loglikelihood_iso_nb(params, scale, shape, vdata, ivar):
    # Construct the model
    model = rot_incl_iso_nb(shape, scale, params)
    inv_sigma2 = ivar
    logL = -0.5 * ma.sum((vdata - model) ** 2 * inv_sigma2 - np.log(inv_sigma2))
    return logL

def nloglikelihood_iso_nb(params, scale, shape, vdata, ivar):
    return -loglikelihood_iso_nb(params, scale, shape, vdata, ivar)

# NFW model with bulge

def loglikelihood_NFW(params, scale, shape, vdata, ivar):
    # Construct the model
    model = rot_incl_NFW(shape, scale, params)
    inv_sigma2 = ivar
    logL = -0.5 * ma.sum((vdata - model) ** 2 * inv_sigma2 - np.log(inv_sigma2))
    return logL

def nloglikelihood_NFW(params, scale, shape, vdata, ivar):
    return -loglikelihood_NFW(params, scale, shape, vdata, ivar)

# NFW model without bulge

def loglikelihood_NFW_nb(params, scale, shape, vdata, ivar):
    # Construct the model
    model = rot_incl_NFW_nb(shape, scale, params)
    inv_sigma2 = ivar
    logL = -0.5 * ma.sum((vdata - model) ** 2 * inv_sigma2 - np.log(inv_sigma2))
    return logL

def nloglikelihood_NFW_nb(params, scale, shape, vdata, ivar):
    return -loglikelihood_NFW_nb(params, scale, shape, vdata, ivar)

# Burket model with bulge

def loglikelihood_bur(params, scale, shape, vdata, ivar):
    # Construct the model
    model = rot_incl_bur(shape, scale, params)
    inv_sigma2 = ivar
    logL = -0.5 * ma.sum((vdata - model) ** 2 * inv_sigma2 - np.log(inv_sigma2))
    return logL

def nloglikelihood_bur(params, scale, shape, vdata, ivar):
    return -loglikelihood_bur(params, scale, shape, vdata, ivar)

# Burket model without bulge

def loglikelihood_bur_nb(params, scale, shape, vdata, ivar):
    # Construct the model
    model = rot_incl_bur_nb(shape, scale, params)
    inv_sigma2 = ivar
    logL = -0.5 * ma.sum((vdata - model) ** 2 * inv_sigma2 - ma.log(inv_sigma2))
    return logL

def nloglikelihood_bur_nb(params, scale, shape, vdata, ivar):
    return -loglikelihood_bur_nb(params, scale, shape, vdata, ivar)

##############################################################################
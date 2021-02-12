##############################################################################
# Functions for constructing velocity maps
#-----------------------------------------------------------------------------

# Importing modules and functions
import numpy as np
import numpy.ma as ma
from Rotation_Curve_Functions import vel_b, v_d, vel_h_iso, vel_h_NFW, vel_h_Burket, v_co_iso, v_co_iso_nb, v_co_NFW, v_co_NFW_nb, v_co_Burket, v_co_Burket_nb

# Isothermal model with bulge

def rot_incl_iso(shape,scale,params):
    A, Vin, logMdisk, Rd, Vinf, Rh,inclination,phi,center_x,center_y = params
    rotated_inclined_map = np.zeros(shape)
    
    for i in range(shape[0]):
        for j in range(shape[1]):
            x = ((i-center_x)*np.cos(phi) - np.sin(phi)*(j-center_y))/np.cos(inclination)
            y = ((i-center_x)*np.sin(phi) + np.cos(phi)*(j-center_y))
            r = np.sqrt(x**2+y**2)
            theta = np.arcsin(x/r)
            r_in_kpc = r*scale
            v = v_co_iso(r_in_kpc,[A, Vin, logMdisk, Rd, Vinf, Rh])*np.sin(inclination)*np.sin(theta)
            rotated_inclined_map[i,j] = v
    return rotated_inclined_map

# Isothermal model without bulge

def rot_incl_iso_nb(shape,scale,params):
    logMdisk, Rd, Vinf, Rh,inclination,phi,center_x,center_y = params
    rotated_inclined_map = np.zeros(shape)
    
    for i in range(shape[0]):
        for j in range(shape[1]):
            x = ((i-center_x)*np.cos(phi) - np.sin(phi)*(j-center_y))/np.cos(inclination)
            y = ((i-center_x)*np.sin(phi) + np.cos(phi)*(j-center_y))
            r = np.sqrt(x**2+y**2)
            theta = np.arcsin(x/r)
            r_in_kpc = r*scale
            v = v_co_iso_nb(r_in_kpc,[logMdisk, Rd, Vinf, Rh])*np.sin(inclination)*np.sin(theta)
            rotated_inclined_map[i,j] = v
    return rotated_inclined_map

# NFW model with bulge

def rot_incl_NFW(shape,scale,params):
    A, Vin,logM,r_d,rho_h,r_h,inclination,phi,center_x,center_y = params
    rotated_inclined_map = np.zeros(shape)
    
    for i in range(shape[0]):
        for j in range(shape[1]):
            x = ((i-center_x)*np.cos(phi) - np.sin(phi)*(j-center_y))/np.cos(inclination)
            y = ((i-center_x)*np.sin(phi) + np.cos(phi)*(j-center_y))
            r = np.sqrt(x**2+y**2)
            theta = np.arcsin(x/r)
            r_in_kpc = r*scale
            v = v_co_NFW(r_in_kpc,[A,Vin,logM,r_d,rho_h,r_h])*np.sin(inclination)*np.sin(theta)
            rotated_inclined_map[i,j] = v
    return rotated_inclined_map

# NFW model without bulge

def rot_incl_NFW_nb(shape,scale,params):
    logM,r_d,rho_h,r_h,inclination,phi,center_x,center_y = params
    rotated_inclined_map = np.zeros(shape)
    
    for i in range(shape[0]):
        for j in range(shape[1]):
            x = ((i-center_x)*np.cos(phi) - np.sin(phi)*(j-center_y))/np.cos(inclination)
            y = ((i-center_x)*np.sin(phi) + np.cos(phi)*(j-center_y))
            r = np.sqrt(x**2+y**2)
            theta = np.arcsin(x/r)
            r_in_kpc = r*scale
            v = v_co_NFW_nb(r_in_kpc,[logM,r_d,rho_h,r_h])*np.sin(inclination)*np.sin(theta)
            rotated_inclined_map[i,j] = v
    return rotated_inclined_map

# Burket model with bulge

def rot_incl_bur(shape,scale,params):
    A, Vin,logM,r_d,rho_h,r_h,inclination,phi,center_x,center_y = params
    rotated_inclined_map = np.zeros(shape)
    
    for i in range(shape[0]):
        for j in range(shape[1]):
            x = ((i-center_x)*np.cos(phi) - np.sin(phi)*(j-center_y))/np.cos(inclination)
            y = ((i-center_x)*np.sin(phi) + np.cos(phi)*(j-center_y))
            r = np.sqrt(x**2+y**2)
            theta = np.arcsin(x/r)
            r_in_kpc = r*scale
            v = v_co_Burket(r_in_kpc,[A,Vin,logM,r_d,rho_h,r_h])*np.sin(inclination)*np.sin(theta)
            rotated_inclined_map[i,j] = v
    return rotated_inclined_map

# Burket model without bulge

def rot_incl_bur_nb(shape,scale,params):
    logM,r_d,rho_h,r_h,inclination,phi,center_x,center_y = params
    rotated_inclined_map = np.zeros(shape)
    
    for i in range(shape[0]):
        for j in range(shape[1]):
            x = ((i-center_x)*np.cos(phi) - np.sin(phi)*(j-center_y))/np.cos(inclination)
            y = ((i-center_x)*np.sin(phi) + np.cos(phi)*(j-center_y))
            r = np.sqrt(x**2+y**2)
            theta = np.arcsin(x/r)
            r_in_kpc = r*0.46886408261217366
            v = v_co_Burket_nb(r_in_kpc,[logM,r_d,rho_h,r_h])*np.sin(inclination)*np.sin(theta)
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
    logL = -0.5 * ma.sum((vdata - model) ** 2 * inv_sigma2 - np.log(inv_sigma2))
    return logL

def nloglikelihood_bur_nb(params, scale, shape, vdata, ivar):
    return -loglikelihood_bur_nb(params, scale, shape, vdata, ivar)

##############################################################################
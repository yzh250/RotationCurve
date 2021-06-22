import matplotlib.pyplot as plt
import numpy as np
import os.path
from astropy.io import fits
from astropy.table import QTable, Column
import astropy.units as u
from scipy.optimize import minimize
import numpy.ma as ma
#import sys
import numdifftools as ndt
import csv

# Physics Constants
c = 3E5 # km * s ^1
h = 1 # reduced hubble constant
H_0 =  100 * h # km * s^-1 * Mpc^-1

# Import functions from other .py files
from rotation_curve_functions import vel_b, \
                                     disk_vel,\
                                     vel_h_iso, \
                                     vel_h_NFW, \
                                     vel_h_Burket, \
                                     v_tot_iso,\
                                     v_tot_NFW,\
                                     v_tot_Burket_nb

from Velocity_Map_Functions import rot_incl_iso,\
                                   rot_incl_NFW, \
                                   rot_incl_bur, \
                                   loglikelihood_iso,\
                                   loglikelihood_NFW, \
                                   loglikelihood_bur,\
                                   loglikelihood_iso_flat,\
                                   loglikelihood_NFW_flat, \
                                   loglikelihood_bur_flat,\
                                   nloglikelihood_iso,\
                                   nloglikelihood_NFW, \
                                   nloglikelihood_bur

from RC_2D_Fit_Functions import Galaxy_Data, \
                                Galaxy_Fitting,\
                                Hessian_Calculation_Isothermal,\
                                Hessian_Calculation_NFW,\
                                Hessian_Calculation_Burket,\
                                Plotting_Isothermal,\
                                Plotting_NFW,\
                                Plotting_Burket

# Used files
DTable1 = QTable.read('Master_Table.txt',format='ascii.commented_header')
DTable2 = QTable.read('DRPall-master_file.txt',format='ascii.ecsv')
# Get the Mass of stars & redshifts & angular resolution of r50
m = DTable1['NSA_Mstar'].data
rat = DTable1['NSA_ba'].data
phi = DTable1['NSA_phi'].data
z = DTable2['redshift'].data
r50_ang = DTable2['nsa_elpetro_th50_r'].data

# Fitting all 7443 galaxies
# all_galaxies = ['7443-12701', '7443-12702', '7443-12703', '7443-12704', '7443-12705']

#----------------------------------------------------------------------------------------------------------------------------------
# Testing on 7443-12705
# Obtaining information for the galaxy 7443-12705
galaxy_ID = '7443-12705'
scale, incl, ph, rband, Ha_vel, Ha_vel_ivar, Ha_vel_mask, vmasked, gshape, x_center_guess, y_center_guess = Galaxy_Data(galaxy_ID)

parameters = [incl, ph, x_center_guess, y_center_guess]

Isothermal_fit, NFW_fit, Burket_Fit = Galaxy_Fitting(parameters, scale, gshape, vmasked, Ha_vel_ivar)

# Plotting
Plotting_Isothermal(galaxy_ID,gshape, scale, Isothermal_fit,Ha_vel_mask)
Plotting_NFW(galaxy_ID,gshape, scale, NFW_fit,Ha_vel_mask)
Plotting_Burket(galaxy_ID,gshape, scale, Burket_Fit,Ha_vel_mask)

# Calculating Chi2
# Isothermal
full_vmap_iso = rot_incl_iso(gshape, scale, Isothermal_fit)
# Masked array
vmap_iso = ma.array(full_vmap_iso, mask = Ha_vel_mask)
# need to calculate number of data points in the fitted map
nd_iso = vmap_iso.shape[0]*vmap_iso.shape[1] - np.sum(vmap_iso.mask)
chi2_iso = np.nansum((vmasked - vmap_iso) ** 2 * Ha_vel_ivar)
chi2_iso_norm = chi2_iso/(nd_iso-8)

# NFW
full_vmap_NFW = rot_incl_NFW(gshape, scale, NFW_fit)
# Masked array
vmap_NFW = ma.array(full_vmap_NFW, mask = Ha_vel_mask)
# need to calculate number of data points in the fitted map
nd_NFW = vmap_NFW.shape[0]*vmap_NFW.shape[1] - np.sum(vmap_NFW.mask)
chi2_NFW = np.nansum((vmasked - vmap_NFW) ** 2 * Ha_vel_ivar)
chi2_NFW_norm = chi2_NFW/(nd_NFW-8)

# Burket
full_vmap_bur = rot_incl_bur(gshape, scale, Burket_Fit)
# Masked array
vmap_bur = ma.array(full_vmap_bur, mask = Ha_vel_mask)
# need to calculate number of data points in the fitted map
nd_bur = vmap_bur.shape[0]*vmap_bur.shape[1] - np.sum(vmap_bur.mask)
chi2_bur = np.nansum((vmasked - vmap_bur) ** 2 * Ha_vel_ivar)
chi2_bur_norm = chi2_bur/(nd_bur-8)

print(chi2_iso_norm)
print(chi2_NFW_norm)
print(chi2_bur_norm)

# Calculating the Hessian Matrix
Hessian_iso = Hessian_Calculation_Isothermal(Isothermal_fit,scale, gshape, vmasked, Ha_vel_ivar)
Hessian_NFW = Hessian_Calculation_NFW(NFW_fit,scale, gshape, vmasked, Ha_vel_ivar)
Hessian_bur = Hessian_Calculation_Burket(Burket_Fit,scale, gshape, vmasked, Ha_vel_ivar)
#----------------------------------------------------------------------------------------------------------------------------------

'''
#----------------------------------------------------------------------------------------------------------------------------------
# All Galaxies
plate = DTable1['MaNGA_plate'].data
IFU = DTable1['MaNGA_IFU'].data
galaxy_ID_all = []
for i in range(len(DTable1)):
    galaxy_ID = str(plate[i]) + '-' + str(IFU[i])
    galaxy_ID_all.append(galaxy_ID)
    print(galaxy_ID_all)
    
# Lists of needed properties of each galaxy
scale_n = []
incl_n = [] 
ph_n = [] 
rband_n = []
Ha_vel_n = []
Ha_vel_ivar_n = []
Ha_vel_mask_n = []
vmasked_n = []
gshape_n = [] 
x_center_guess_n = [] 
y_center_guess_n = []

# Lists of fitted solutions
Isothermal_Fits = []
NFW_Fits = []
Burket_Fits = []

# Lists of fitted model map
vmap_fit_n_iso = []
vmap_fit_n_NFW = []
vmap_fit_nbur = []

# Lists of chi2 values 
chi2_n_iso = []
chi2_n_NFW = []
chi2_n_bur = []

# Fitting Pipeline
for i in range(len(galaxy_ID_all)):
    # Obtaining needed properties
    scale, incl, ph, rband, Ha_vel, Ha_vel_ivar, Ha_vel_mask, vmasked, gshape, x_center_guess, y_center_guess = Galaxy_Data(galaxy_ID_all[i])
    
    # Saving data in to a list
    scale_n.append(scale)
    incl_n.append(incl)
    ph_n.append(phi)
    rband_n.append(rband)
    Ha_vel_n.append(Ha_vel)
    Ha_vel_ivar_n.append(Ha_vel_ivar)
    Ha_vel_mask_n.append(Ha_vel_mask)
    vmasked_n.append(vmasked)
    gshape_n.append(gshape)
    x_center_guess_n.append(x_center_guess) 
    y_center_guess_n.append(y_center_guess)
    
    # Fitting
    params = [incl, ph, x_center_guess, y_center_guess]
    Isothermal_fit, NFW_fit, Burket_fit = Galaxy_Fitting(parameters, scale, gshape, vmasked, Ha_vel_ivar)
    
    # Calculating Chi2
    full_vmap = rot_incl_iso(gshape, scale, Isothermal_fit)
    # need to calculate number of data points in the fitted map
    vmap_1.shape[0]*vmap_1.shape[1] - np.sum(vmap_1_masked.mask)
    # Isothermal
    masked_vmap_iso = ma.array(rot_incl_iso(gshape, scale, Isothermal_fit))
    chi2_iso = np.nansum((vmasked - masked_vmap_iso) ** 2 * Ha_vel_ivar])
    chi2/(-8)
    
    # Plotting
    #Plotting_Isothermal(galaxy_ID,gshape, scale, Isothermal_fit,Ha_vel_mask)
    #Plotting_NFW(galaxy_ID,gshape, scale, NFW_fit,Ha_vel_mask)
    #Plotting_Burket(galaxy_ID,gshape, scale, Burket_Fit,Ha_vel_mask)
    
    # Saving Fitted solution to lists
    Isothermal_Fits.append(Isothermal_fit)
    NFW_Fits.append(NFW_fit)
    Burket_Fits.append(Burket_fit)

    # Numerically calculating the Hessian Matrix
    
# Saving the fit solutions into a data table
with open('Fit_Solutions_DRP_all_Isothermal.csv',mode='w') as M_csv:
    DRP_iso = csv.writer(M_csv, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    DRP_iso.writerow(['MaNGA_plate','MaNGA_IFU','Alpha','V_int','Sigma_d','R_d','V_inf','R_h','incl','phi','center_x','center_y','chi2_iso','chi2_NFW','chi2_bur'])
    for i in range(len(galaxy_ID_all)):
        DRP_iso.writerow([plate,
                          IFU,
                          Isothermal_Fits[i][0],
                          Isothermal_Fits[i][1],
                          Isothermal_Fits[i][2],
                          Isothermal_Fits[i][3],
                          Isothermal_Fits[i][4],
                          Isothermal_Fits[i][5],
                          Isothermal_Fits[i][6],
                          Isothermal_Fits[i][7],])
'''












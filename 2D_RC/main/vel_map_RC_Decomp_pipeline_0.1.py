################################################################################
# Import modules
#-------------------------------------------------------------------------------
import numpy as np
import numpy.ma as ma

from astropy.table import Table, QTable

import csv

import time


# Import functions from other .py files
from Velocity_Map_Functions import rot_incl_iso,\
                                   rot_incl_NFW, \
                                   rot_incl_bur

from RC_2D_Fit_Functions import Galaxy_Data, \
                                Galaxy_Fitting_iso,\
                                Galaxy_Fitting_NFW, \
                                Galaxy_Fitting_bur, \
                                Hessian_Calculation_Isothermal,\
                                Hessian_Calculation_NFW,\
                                Hessian_Calculation_Burket,\
                                Plotting_Isothermal,\
                                Plotting_NFW,\
                                Plotting_Burket

from mapSmoothness_functions import how_smooth
import os.path
from os import path
################################################################################

################################################################################
# Physics Constants
#-------------------------------------------------------------------------------
c = 3E5 # km * s ^1
h = 1 # reduced hubble constant
H_0 =  100 * h # km * s^-1 * Mpc^-1
################################################################################

################################################################################
# Used files
#-------------------------------------------------------------------------------

MANGA_FOLDER = '/Users/richardzhang/Documents/UR_Stuff/Research_UR/SDSS/dr16/manga/spectro/'
DRP_FILENAME = MANGA_FOLDER + 'redux/v2_4_3/drpall-v2_4_3.fits'
VEL_MAP_FOLDER = MANGA_FOLDER + 'analysis/v2_4_3/2.2.1/HYB10-GAU-MILESHC/'

DTable =  Table.read(DRP_FILENAME, format='fits')

DRP_index = {}

for i in range(len(DTable)):
    gal_ID = DTable['plateifu'][i]

    DRP_index[gal_ID] = i
################################################################################

################################################################################
# Get the Mass of stars & redshifts & angular resolution of r50
#-------------------------------------------------------------------------------
m = DTable['nsa_elpetro_mass']
rat = DTable['nsa_elpetro_ba']
phi = DTable['nsa_elpetro_phi']
z = DTable['nsa_z']
r50_ang = DTable['nsa_elpetro_th50_r']
################################################################################

#-------------------------------------------------------------------------------
# Obtaining information for all 7443 galaxies
#-------------------------------------------------------------------------------
galaxy_ID = ['7443-1901','7443-1902','7443-3701',\
             '7443-3702','7443-3703','7443-3704',\
             '7443-6101','7443-6102','7443-6103',\
             '7443-6104','7443-9101','7443-9102',\
             '7443-12701','7443-12702','7443-12703',\
             '7443-12704','7443-12705']
             #,'7495-1901',\
             #'7495-1902','7495-3701','7495-3702',\
             #'7495-3703','7495-3704','7495-6101',\
             #'7495-6102','7495-6103','7495-6104',\
             #'7495-9101','7495-9102','7495-12701',\
             #'7495-12702','7495-12703','7495-12704',\
             #'7495-12705']

for i in range(len(galaxy_ID)):

    plate, IFU = galaxy_ID[i].split('-')

    data_file = VEL_MAP_FOLDER+plate+'/'+IFU+'/manga-'+galaxy_ID[i]+'-MAPS-HYB10-GAU-MILESHC.fits.gz'

    j = DRP_index[galaxy_ID[i]]

    redshift = z[j]
    velocity =  redshift* c
    distance = (velocity / H_0) * 1E3 #kpc
    scale = 0.5 * (distance) / 206265

    incl = np.arccos(rat[j])

    ph = phi[j] * np.pi / 180

    if path.exists(data_file):
            # Get data
            # scale, incl, ph, rband, Ha_vel, Ha_vel_ivar, Ha_vel_mask, vmasked, gshape, x_center_guess, y_center_guess = Galaxy_Data(galaxy_ID)
            rband, Ha_vel, Ha_vel_ivar, Ha_vel_mask, vmasked, ivar_masked, gshape, x_center_guess, y_center_guess = Galaxy_Data(galaxy_ID[i])
            # -------------------------------------------------------------------------------

            ################################################################################
            # Smoothness Check
            # -------------------------------------------------------------------------------
            max_map_smoothness = 1.85

            map_smoothness = how_smooth(Ha_vel, Ha_vel_mask)

            if map_smoothness <= max_map_smoothness:
                # -------------------------------------------------------------------------------
                # Fit the galaxy (normal likelihood)
                # -------------------------------------------------------------------------------
                print('Fitting galaxy ', galaxy_ID[i])
                start_time = time.time()

                parameters = [incl, ph, x_center_guess, y_center_guess]

                Isothermal_fit = Galaxy_Fitting_iso(parameters,
                                                    scale,
                                                    gshape,
                                                    vmasked,
                                                    Ha_vel_ivar,
                                                    galaxy_ID[i])

                NFW_fit = Galaxy_Fitting_NFW(parameters,
                                             scale,
                                             gshape,
                                             vmasked,
                                             Ha_vel_ivar,
                                             galaxy_ID[i])

                Burket_fit = Galaxy_Fitting_bur(parameters,
                                                scale,
                                                gshape,
                                                vmasked,
                                                Ha_vel_ivar,
                                                galaxy_ID[i])

                print('Fit galaxy', time.time() - start_time)
                # -------------------------------------------------------------------------------

                # -------------------------------------------------------------------------------
                # Plotting
                # -------------------------------------------------------------------------------
                Plotting_Isothermal(galaxy_ID[i], gshape, scale, Isothermal_fit, Ha_vel_mask)
                Plotting_NFW(galaxy_ID[i], gshape, scale, NFW_fit, Ha_vel_mask)
                Plotting_Burket(galaxy_ID[i], gshape, scale, Burket_fit, Ha_vel_mask)
                # -------------------------------------------------------------------------------

                # -------------------------------------------------------------------------------
                # Calculating Chi2
                # -------------------------------------------------------------------------------
                print('Calculating chi2')
                start_time = time.time()

                # -------------------------------------------------------------------------------
                # Isothermal

                full_vmap_iso = rot_incl_iso(gshape, scale, Isothermal_fit)

                # Masked array
                vmap_iso = ma.array(full_vmap_iso, mask=Ha_vel_mask)

                # need to calculate number of data points in the fitted map
                # nd_iso = vmap_iso.shape[0]*vmap_iso.shape[1] - np.sum(vmap_iso.mask)
                nd_iso = np.sum(~vmap_iso.mask)

                # chi2_iso = np.nansum((vmasked - vmap_iso) ** 2 * Ha_vel_ivar)
                chi2_iso = ma.sum(Ha_vel_ivar * (vmasked - vmap_iso) ** 2)

                # chi2_iso_norm = chi2_iso/(nd_iso - 8)
                chi2_iso_norm = chi2_iso / (nd_iso - len(Isothermal_fit))
                # -------------------------------------------------------------------------------

                # -------------------------------------------------------------------------------
                # NFW

                full_vmap_NFW = rot_incl_NFW(gshape, scale, NFW_fit)

                # Masked array
                vmap_NFW = ma.array(full_vmap_NFW, mask=Ha_vel_mask)

                # need to calculate number of data points in the fitted map
                # nd_NFW = vmap_NFW.shape[0]*vmap_NFW.shape[1] - np.sum(vmap_NFW.mask)
                nd_NFW = np.sum(~vmap_NFW.mask)

                # chi2_NFW = np.nansum((vmasked - vmap_NFW) ** 2 * Ha_vel_ivar)
                chi2_NFW = ma.sum(Ha_vel_ivar * (vmasked - vmap_NFW) ** 2)

                # chi2_NFW_norm = chi2_NFW/(nd_NFW - 8)
                chi2_NFW_norm = chi2_NFW / (nd_NFW - len(NFW_fit))
                # -------------------------------------------------------------------------------

                # -------------------------------------------------------------------------------
                # Burket

                full_vmap_bur = rot_incl_bur(gshape, scale, Burket_fit)

                # Masked array
                vmap_bur = ma.array(full_vmap_bur, mask=Ha_vel_mask)

                # need to calculate number of data points in the fitted map
                # nd_bur = vmap_bur.shape[0]*vmap_bur.shape[1] - np.sum(vmap_bur.mask)
                nd_bur = np.sum(~vmap_bur.mask)

                # chi2_bur = np.nansum((vmasked - vmap_bur) ** 2 * Ha_vel_ivar)
                chi2_bur = ma.sum(Ha_vel_ivar * (vmasked - vmap_bur) ** 2)

                # chi2_bur_norm = chi2_bur/(nd_bur-8)
                chi2_bur_norm = chi2_bur / (nd_bur - len(Burket_fit))
                # -------------------------------------------------------------------------------
                print('Isothermal chi2:', chi2_iso_norm, time.time() - start_time)
                print('NFW chi2:', chi2_NFW_norm)
                print('Burket chi2:', chi2_bur_norm)
                # -------------------------------------------------------------------------------

                '''
                # -------------------------------------------------------------------------------
                # Calculating the Hessian Matrix
                # -------------------------------------------------------------------------------
                print('Calculating Hessian')
                start_time = time.time()

                Hessian_iso = Hessian_Calculation_Isothermal(Isothermal_fit,
                                                             scale,
                                                             gshape,
                                                             vmasked,
                                                             Ha_vel_ivar)

                Hessian_NFW = Hessian_Calculation_NFW(NFW_fit,
                                                      scale,
                                                      gshape,
                                                      vmasked,
                                                      Ha_vel_ivar)

                Hessian_bur = Hessian_Calculation_Burket(Burket_fit,
                                                         scale,
                                                         gshape,
                                                         vmasked,
                                                         Ha_vel_ivar)

                print('Calculated Hessian', time.time() - start_time)
                # -------------------------------------------------------------------------------
                '''
                # Three files to save data for each halo model

            else:
                print(galaxy_ID[i] + ' does not have rotation curve')

    else:
        print('No data for the galaxy.')






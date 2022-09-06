################################################################################
# Import modules
#-------------------------------------------------------------------------------
import numpy as np
import numpy.ma as ma
from astropy.table import Table, QTable
import csv


import time


# Import functions from other .py files
from Velocity_Map_Functions import find_phi

from RC_2D_Fit_Functions import Galaxy_Data, \
                                Galaxy_Fitting_iso,\
                                Galaxy_Fitting_NFW, \
                                Galaxy_Fitting_bur, \
                                Hessian_Calculation_Isothermal,\
                                Hessian_Calculation_NFW,\
                                Hessian_Calculation_Burket,\
                                Plotting_Isothermal,\
                                Plotting_NFW,\
                                Plotting_Burkert,\
                                getTidal,\
                                deproject_spaxel,\
                                plot_rot_curve,\
                                plot_diagnostic_panel,\
                                run_MCMC

from Velocity_Map_Functions_cython import rot_incl_iso,\
                                          rot_incl_NFW, \
                                          rot_incl_bur

from mapSmoothness_functions import how_smooth

from os import path

import matplotlib.pyplot as plt
################################################################################




################################################################################
# Physics Constants
#-------------------------------------------------------------------------------
c = 3E5 # km * s ^1
h = 1 # reduced hubble constant
H_0 =  100 * h # km * s^-1 * Mpc^-1
q0 = 0.2 # minimum inclination value
################################################################################

################################################################################
# Used files (local)
#-------------------------------------------------------------------------------
'''
MANGA_FOLDER = '/Users/richardzhang/Documents/UR_Stuff/Research_UR/SDSS/dr16/manga/spectro/'

DRP_FILENAME = MANGA_FOLDER + 'redux/v3_1_1/drpall-v3_1_1.fits'

# Can't really use this anymore
VEL_MAP_FOLDER = MANGA_FOLDER + 'analysis//v3_1_1/2.1.1/HYB10-GAU-MILESHC/'

MORPH_FOLDER = '/Users/richardzhang/Documents/UR_Stuff/Research_UR/SDSS/dr16/manga/morph/'
'''
################################################################################

################################################################################
# Used files (bluehive)
#-------------------------------------------------------------------------------
MANGA_FOLDER_yifan = '/home/yzh250/Documents/UR_Stuff/Research_UR/SDSS/dr17/manga/spectro/'

DRP_FILENAME = MANGA_FOLDER_yifan + 'redux/v3_1_1/drpall-v3_1_1.fits'

VEL_MAP_FOLDER = '/scratch/kdougla7/data/SDSS/dr17/manga/spectro/analysis/v3_1_1/3.1.0/HYB10-MILESHC-MASTARSSP/'

MORPH_FOLDER = '/home/yzh250/Documents/UR_Stuff/Research_UR/SDSS/dr17/manga/morph/'

SMOOTHNESS_MORPH_FOLDER = '/home/yzh250/Documents/UR_Stuff/Research_UR/RotationCurve/2D_RC/main/'

smoothness_morph_file = SMOOTHNESS_MORPH_FOLDER + 'cross_table.csv'
################################################################################


DTable =  Table.read(DRP_FILENAME, format='fits')

#MTable =  Table.read(MORPH_file, format='fits')

DRP_index = {}

for i in range(len(DTable)):
    gal_ID = DTable['plateifu'][i]

    DRP_index[gal_ID] = i

# DL morph catalog
cross_match_table = Table.read(smoothness_morph_file,format='ascii.commented_header')
gal_ID_cross = cross_match_table['galaxy_ID'].data
ttype = cross_match_table['DL_ttype'].data
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





################################################################################
# Obtaining information for MaNGA galaxies
#-------------------------------------------------------------------------------
#galaxy_ID = ['8554-12701']
galaxy_ID = []
plateifu = DTable['plateifu'].data

for i in range(len(plateifu)):
    galaxy_ID.append(str(plateifu[i],'utf-8'))

no_data_count = 0

smoothness_leq_2 = 0
ttype_g_0 = 0

smoothness_leq_2_ttype_g_0 = 0

smoothness_leq_2_not_ttype_g_0 = 0
ttype_g_0_not_smoothness_leq_2 = 0
not_ttype_g_0_not_smoothness_leq_2 = 0

incl_less_0 = 0
tidal_presence = 0
masked_data = 0

for i in range(len(galaxy_ID)):

    plate, IFU = galaxy_ID[i].split('-')

    # bluehive
    data_file = VEL_MAP_FOLDER + plate + '/' + IFU + '/manga-' + galaxy_ID[i] + '-MAPS-HYB10-MILESHC-MASTARSSP.fits.gz'
    # local
    #data_file = VEL_MAP_FOLDER + plate + '/' + IFU + '/manga-' + galaxy_ID[i] + '-MAPS-HYB10-GAU-MILESHC.fits.gz'

    j = DRP_index[galaxy_ID[i]]

    redshift = z[j]
    velocity =  redshift* c
    distance = (velocity / H_0) * 1000 #kpc
    scale = 0.5 * distance / 206265

    #c_scale['scale'][i] = scale

    #incl = np.arccos(rat[j])
    cosi2 = (rat[j]**2 - q0**2)/(1 - q0**2)
    if cosi2 < 0:
        cosi2 = 0

    incl = np.arccos(np.sqrt(cosi2))

    #ph = phi[j] * np.pi / 180

    if path.exists(data_file):
        print(galaxy_ID[i] + ' data exists.')
        ########################################################################
        # Get data
        #-----------------------------------------------------------------------
        # scale, incl, ph, rband, Ha_vel, Ha_vel_ivar, Ha_vel_mask, vmasked, gshape, x_center_guess, y_center_guess = Galaxy_Data(galaxy_ID)
        data_maps, gshape = Galaxy_Data(galaxy_ID[i],VEL_MAP_FOLDER)
        #-----------------------------------------------------------------------


        ########################################################################
        # Selection
        #-----------------------------------------------------------------------
        # Morphological cut
        #tidal = getTidal(galaxy_ID[i], MORPH_FOLDER)
        tidal = getTidal(galaxy_ID[i], MORPH_FOLDER)

        Ttype = 0
        if galaxy_ID[i] == gal_ID_cross[i]:
            Ttype = ttype[i]

        # Smoothness cut
        max_map_smoothness = 2

        map_smoothness = how_smooth(data_maps['Ha_vel'], data_maps['Ha_vel_mask'])

        SN_map = data_maps['Ha_flux'] * np.sqrt(data_maps['Ha_flux_ivar'])
        Ha_vel_mask = data_maps['Ha_vel_mask'] + (SN_map < 5)

        vmasked = ma.array(data_maps['Ha_vel'], mask = Ha_vel_mask)
        ivar_masked = ma.array(data_maps['Ha_vel_ivar'], mask = Ha_vel_mask)

        r_band_masked = ma.array(data_maps['r_band'],mask=Ha_vel_mask)

        center_guess = np.unravel_index(ma.argmax(r_band_masked), gshape)
        x_center_guess = center_guess[0]
        y_center_guess = center_guess[1]

        '''
        global_max = ma.max(vmasked)

        unmasked_data = True

        if np.isnan(global_max) or (global_max is ma.masked):
            unmasked_data = False
        '''
        if map_smoothness <= max_map_smoothness:
            print(galaxy_ID[i] + ' has a smoothness score less than or equal to 2')
            smoothness_leq_2 += 1
            if Ttype > 0:
            print(galaxy_ID[i] + ' has a ttype greater than 0 and a smoothness score less than or equal to 2')
            smoothness_leq_2_ttype_g_0 += 1
            else：
            print(galaxy_ID[i] + ' has a smoothness score less than or equal to 2 but a ttype less or equal to 0')
            smoothness_leq_2_not_ttype_g_0 += 1

        elif Ttype > 0:
            print(galaxy_ID[i] + ' has a ttype greater than 0')
            ttype_g_0 += 1
            if (map_smoothness > max_map_smoothness):
            print(galaxy_ID[i] + ' has a ttype greater than 0 but a smoothness score greater than 2')
            ttype_g_0_not_smoothness_leq_2 += 1

        else:
            print(galaxy_ID[i] + ' has ttype less than or equal to 0 and a smoothness score greater than 2')
            not_ttype_g_0_not_smoothness_leq_2 += 1
    else:
        print(galaxy_ID[i] + ' no data.')
        no_data_count += 1


print(str(no_data_count) + ' galaxies have no data')
print(str(smoothness_leq_2) + ' galaxies have smoothness score less than or equal to 2')
print(str(ttype_g_0) + ' galaxies have a ttype greater than 0')
print(str(ttype_g_0_not_smoothness_leq_2) + ' galaxies have ttype greater than 0 but a smoothness score greater than 2')
print(str(smoothness_leq_2_not_ttype_g_0) + ' galaxies have a smoothness score less than or equal to 2 but a ttype less or equal to 0')
print(str(smoothness_leq_2_ttype_g_0) + ' galaxies have ttype greater than 0 and a smoothness score less than or equal to 2')
print(str(not_ttype_g_0_not_smoothness_leq_2) + ' galaxies have ttype less than or equal to 0 and a smoothness score greater than 2')



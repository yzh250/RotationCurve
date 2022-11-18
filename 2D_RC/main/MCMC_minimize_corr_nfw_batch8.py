################################################################################
# Import modules
#-------------------------------------------------------------------------------
import numpy as np
import numpy.ma as ma
from astropy.table import Table, QTable
from astropy.io import ascii
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
MANGA_FOLDER_yifan = '/home/yzh250/Documents/UR_Stuff/Research_UR/SDSS/dr17/manga/spectro/'

DRP_FILENAME = MANGA_FOLDER_yifan + 'redux/v3_1_1/drpall-v3_1_1.fits'

VEL_MAP_FOLDER = '/scratch/kdougla7/data/SDSS/dr17/manga/spectro/analysis/v3_1_1/3.1.0/HYB10-MILESHC-MASTARSSP/'

#MORPH_FOLDER = '/home/yzh250/Documents/UR_Stuff/Research_UR/SDSS/dr17/manga/morph/'

SMOOTHNESS_MORPH_FOLDER = '/home/yzh250/Documents/UR_Stuff/Research_UR/RotationCurve/2D_RC/main/'

smoothness_morph_file = SMOOTHNESS_MORPH_FOLDER + 'cross_table.csv'
################################################################################

################################################################################
# DRP all table
DTable =  Table.read(DRP_FILENAME, format='fits')

#MTable =  Table.read(MORPH_file, format='fits')

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

################################################################################
# Importing fitted values & chi2 for each galaxy
fit_mini_nfw_name = SMOOTHNESS_MORPH_FOLDER + 'nfw_mini_clean2.csv'
fit_mini_nfw_table = ascii.read(fit_mini_nfw_name,'r')
fit_mini_nfw = fit_mini_nfw_table[840:960]
################################################################################

################################################################################
#galaxy = []
#plateifu = DTable['plateifu'].data

#for i in range(len(plateifu)):
    #gal_ID.append(str(plateifu[i],'utf-8'))
################################################################################

c_nfw_MCMC = Table()
c_nfw_MCMC['gal_ID'] = fit_mini_nfw['galaxy_ID']
c_nfw_MCMC['rho0_b'] = np.nan
c_nfw_MCMC['Rb'] = np.nan
c_nfw_MCMC['SigD'] = np.nan
c_nfw_MCMC['Rd'] = np.nan
c_nfw_MCMC['rho0_h'] = np.nan
c_nfw_MCMC['Rh'] = np.nan
c_nfw_MCMC['incl'] = np.nan
c_nfw_MCMC['phi'] = np.nan
c_nfw_MCMC['x_cen'] = np.nan
c_nfw_MCMC['y_cen'] = np.nan
c_nfw_MCMC['Vsys'] = np.nan
c_nfw_MCMC['chi2'] = np.nan

################################################################################
for i in range(len(fit_mini_nfw)):
    # obtain galaxy data & initial guess parameters

    gal_fit = list(fit_mini_nfw[i])

    gal_ID = gal_fit[0]

    plate, IFU = gal_ID.split('-')

    data_file = VEL_MAP_FOLDER + plate + '/' + IFU + '/manga-' + gal_ID + '-MAPS-HYB10-MILESHC-MASTARSSP.fits.gz'

    j = DRP_index[gal_ID]

    redshift = z[j]
    velocity =  redshift* c
    distance = (velocity / H_0) * 1000 #kpc
    scale = 0.5 * distance / 206265

    #incl = np.arccos(rat[j])
    cosi2 = (rat[j]**2 - q0**2)/(1 - q0**2)
    if cosi2 < 0:
        cosi2 = 0

    incl = np.arccos(np.sqrt(cosi2))

    print('Incl calculated: ' + str(incl), flush=True)

    data_maps, gshape = Galaxy_Data(gal_ID,VEL_MAP_FOLDER)

    #-----------------------------------------------------------------------

    ########################################################################
    # Selection
    #-----------------------------------------------------------------------
    # Morphological cut
    #tidal = getTidal(gal_ID, MORPH_FOLDER)
    #tidal = getTidal(gal_ID, SMOOTHNESS_MORPH_FOLDER)

    # Smoothness cut
    '''
    max_map_smoothness = 1.85

    map_smoothness = how_smooth(data_maps['Ha_vel'], data_maps['Ha_vel_mask'])

    print('smoothness calculated: ' + str(map_smoothness), flush=True)
    '''

    SN_map = data_maps['Ha_flux'] * np.sqrt(data_maps['Ha_flux_ivar'])
    Ha_vel_mask = data_maps['Ha_vel_mask'] + (SN_map < 5)

    vmasked = ma.array(data_maps['Ha_vel'], mask = Ha_vel_mask)
    ivar_masked = ma.array(data_maps['Ha_vel_ivar'], mask = Ha_vel_mask)

    r_band_masked = ma.array(data_maps['r_band'],mask=Ha_vel_mask)

    center_guess = np.unravel_index(ma.argmax(r_band_masked), gshape)
    x_center_guess = center_guess[0]
    y_center_guess = center_guess[1]

    print('center found',flush=True)

    global_max = ma.max(vmasked)

    unmasked_data = True

    if np.isnan(global_max) or (global_max is ma.masked):
        unmasked_data = False

    # center coordinates
    center_coord = (x_center_guess, y_center_guess)

    if gal_ID in ['8466-12705']:
        center_coord = (37,42)

        ####################################################################
        # Find initial guess for phi
        #-------------------------------------------------------------------

    print('Start finding phi', flush=True)

    phi_guess = find_phi(center_coord, phi[j], vmasked)

    if gal_ID in ['8134-6102']:
        phi_guess += 0.25 * np.pi

    elif gal_ID in ['8932-12704', '8252-6103']:
        phi_guess -= 0.25 * np.pi

    elif gal_ID in ['8613-12703', '8726-1901', '8615-1901', '8325-9102',
                          '8274-6101', '9027-12705', '9868-12702', '8135-1901',
                          '7815-1901', '8568-1901', '8989-1902', '8458-3701',
                          '9000-1901', '9037-3701', '8456-6101']:
        phi_guess += 0.5 * np.pi

    elif gal_ID in ['9864-3702', '8601-1902']:
        phi_guess -= 0.5 * np.pi

    elif gal_ID in ['9502-12702']:
        phi_guess += 0.75 * np.pi

    elif gal_ID in ['7495-6104']:
        phi_guess -= 0.8 * np.pi

    elif gal_ID in ['7495-12704','7815-6103','9029-12705', '8137-3701', '8618-3704', '8323-12701',
                          '8942-3703', '8333-12701', '8615-6103', '9486-3704',
                          '8937-1902', '9095-3704', '8466-1902', '9508-3702',
                          '8727-3703', '8341-12704', '8655-6103']:
        phi_guess += np.pi

    elif gal_ID in ['7815-9102']:
        phi_guess -= np.pi

    elif gal_ID in ['7443-9101', '7443-3704']:
        phi_guess -= 1.06 * np.pi

    elif gal_ID in ['8082-1901', '8078-3703', '8551-1902', '9039-3703',
                          '8624-1902', '8948-12702', '8443-6102', '8259-1901']:
        phi_guess += 1.5 * np.pi

    elif gal_ID in ['8241-12705', '8326-6102']:
        phi_guess += 1.75 * np.pi

    elif gal_ID in ['7443-6103']:
        phi_guess += 2.3 * np.pi

    # phi value
    phi_guess = phi_guess % (2 * np.pi)

    parameters = [incl, phi_guess, x_center_guess, y_center_guess]

    NFW_fit_mini = gal_fit
    Rb = NFW_fit_mini[4]
    Rd = NFW_fit_mini[6]
    Rh = NFW_fit_mini[8]

    print(Rb,Rd,Rh,flush=True)
    
    if not (Rb < Rd and Rd < Rh):
        print('fitting MCMC ' + gal_ID,flush=True)
        NFW_fit_MCMC, chi2_nfw_norm_MCMC = run_MCMC(gal_ID,VEL_MAP_FOLDER,parameters,scale,'nfw')
        #c_nfw_MCMC['rho0_b'][i] = gal_ID
        c_nfw_MCMC['rho0_b'][i] = NFW_fit_MCMC[0]
        c_nfw_MCMC['Rb'][i] = NFW_fit_MCMC[1]
        c_nfw_MCMC['SigD'][i] = NFW_fit_MCMC[2]
        c_nfw_MCMC['Rd'][i] = NFW_fit_MCMC[3]
        c_nfw_MCMC['rho0_h'][i] = NFW_fit_MCMC[4]
        c_nfw_MCMC['Rh'][i] = NFW_fit_MCMC[5]
        c_nfw_MCMC['incl'][i] = NFW_fit_MCMC[6]
        c_nfw_MCMC['phi'][i] = NFW_fit_MCMC[7]
        c_nfw_MCMC['x_cen'][i] = NFW_fit_MCMC[8]
        c_nfw_MCMC['y_cen'][i] = NFW_fit_MCMC[9]
        c_nfw_MCMC['Vsys'][i] = NFW_fit_MCMC[10]
        c_nfw_MCMC['chi2'][i] = chi2_nfw_norm_MCMC
    else:
        print(gal_ID + ' good fits from minimize with physical values',flush=True)
        
c_nfw_MCMC.write('nfw_mcmc_corr_mini_b8.csv', format='ascii.csv', overwrite=True)


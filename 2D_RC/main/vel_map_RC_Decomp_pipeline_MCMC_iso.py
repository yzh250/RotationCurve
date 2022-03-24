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
                                Plotting_Burket,\
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

fits_file = '/Users/richardzhang/Documents/UR_Stuff/Research_UR/RotationCurve/2D_RC/main/Notebooks/'
'''
################################################################################

################################################################################
# Used files (bluehive)
#-------------------------------------------------------------------------------
MANGA_FOLDER_yifan = '/home/yzh250/Documents/UR_Stuff/Research_UR/SDSS/dr17/manga/spectro/'

DRP_FILENAME = MANGA_FOLDER_yifan + 'redux/v3_1_1/drpall-v3_1_1.fits'

VEL_MAP_FOLDER = '/scratch/kdougla7/data/SDSS/dr17/manga/spectro/analysis/v3_1_1/3.1.0/HYB10-MILESHC-MASTARSSP/'

MORPH_FOLDER = '/home/yzh250/Documents/UR_Stuff/Research_UR/SDSS/dr17/manga/morph/'

fits_file = '/home/yzh250/Documents/UR_Stuff/Research_UR/RotationCurve/2D_RC/main/'
################################################################################


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
# Obtaining information for MaNGA galaxies
#-------------------------------------------------------------------------------
#galaxy_ID = ['8554-12701']
galaxy_ID = []
plateifu = DTable['plateifu'].data

for i in range(len(plateifu)):
    galaxy_ID.append(str(plateifu[i],'utf-8'))

#-------------------------------------------------------------------------------
#plate = ['7443','7495','7815','7957','7958','7960','7962','7964','7968','7972','7975','7977','7990','7991','7992']
#IFU = ['1901','1902','3701','3702','3703','3704','6101','6102','6103','6104','9101','9102','12701','12702','12703','12704','12705']

'''
for i in range(len(plate)):
   for j in range(len(IFU)):
       galaxy_ID.append(plate[i] + '-' + IFU[j])
'''
#-------------------------------------------------------------------------------

'''
c_scale = Table()
c_scale['galaxy_ID'] = galaxy_ID
c_scale['scale'] = np.nan
'''

# Creating MCMC file

# Isothermal
#c_iso = open('iso_exp.csv','w')
#writer_iso = csv.writer(c_iso)
#writer_iso.writerow(['galaxy_ID', 'A', 'Vin', 'SigD', 'Rd', 'rho0_h', 'Rh', 'incl', 'phi', 'x_cen', 'y_cen','Vsys','chi2'])
c_iso_MCMC = Table()
c_iso_MCMC['galaxy_ID'] = galaxy_ID
c_iso_MCMC['A'] = np.nan
c_iso_MCMC['Vin'] = np.nan
c_iso_MCMC['SigD'] = np.nan
c_iso_MCMC['Rd'] = np.nan
c_iso_MCMC['rho0_h'] = np.nan
c_iso_MCMC['Rh'] = np.nan
c_iso_MCMC['incl'] = np.nan
c_iso_MCMC['phi'] = np.nan
c_iso_MCMC['x_cen'] = np.nan
c_iso_MCMC['y_cen'] = np.nan
c_iso_MCMC['Vsys'] = np.nan
c_iso_MCMC['chi2'] = np.nan

# Fitting the galaxy
fit_mini_iso_name = fits_file + 'iso_mini.csv'
fit_mini_iso = ascii.read(fit_mini_iso_name,'r')


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

    if path.exists(data_file) and (incl > 0):
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

        # Smoothness cut
        max_map_smoothness = 1.85

        map_smoothness = how_smooth(data_maps['Ha_vel'], data_maps['Ha_vel_mask'])

        SN_map = data_maps['Ha_flux'] * np.sqrt(data_maps['Ha_flux_ivar'])
        Ha_vel_mask = data_maps['Ha_vel_mask'] + (SN_map < 5)

        vmasked = ma.array(data_maps['Ha_vel'], mask = Ha_vel_mask)
        ivar_masked = ma.array(data_maps['Ha_vel_ivar'], mask = Ha_vel_mask)

        r_band_masked = ma.array(data_maps['r_band'],mask=Ha_vel_mask)

        center_guess = np.unravel_index(ma.argmax(r_band_masked), gshape)
        x_center_guess = center_guess[0]
        y_center_guess = center_guess[1]

        global_max = ma.max(vmasked)

        unmasked_data = True

        if np.isnan(global_max) or (global_max is ma.masked):
            unmasked_data = False

        if map_smoothness <= max_map_smoothness and tidal == 0 and (unmasked_data == True):
            
            print('Fitting galaxy ', galaxy_ID[i], flush=True)
            start_time = time.time()

            center_coord = (x_center_guess, y_center_guess)

            if galaxy_ID[i] in ['8466-12705']:
                center_coord = (37,42)

            ####################################################################
            # Find initial guess for phi
            #-------------------------------------------------------------------
            phi_guess = find_phi(center_coord, phi[j], vmasked)

            if galaxy_ID[i] in ['8134-6102']:
                phi_guess += 0.25 * np.pi

            elif galaxy_ID[i] in ['8932-12704', '8252-6103']:
                phi_guess -= 0.25 * np.pi

            elif galaxy_ID[i] in ['8613-12703', '8726-1901', '8615-1901', '8325-9102',
                                 '8274-6101', '9027-12705', '9868-12702', '8135-1901',
                                 '7815-1901', '8568-1901', '8989-1902', '8458-3701',
                                 '9000-1901', '9037-3701', '8456-6101']:
                phi_guess += 0.5 * np.pi

            elif galaxy_ID[i] in ['9864-3702', '8601-1902']:
                phi_guess -= 0.5 * np.pi

            elif galaxy_ID[i] in ['9502-12702']:
                phi_guess += 0.75 * np.pi

            elif galaxy_ID[i] in ['7495-6104']:
                phi_guess -= 0.8 * np.pi

            elif galaxy_ID[i] in ['7495-12704','7815-6103','9029-12705', '8137-3701', '8618-3704', '8323-12701',
                                 '8942-3703', '8333-12701', '8615-6103', '9486-3704',
                                 '8937-1902', '9095-3704', '8466-1902', '9508-3702',
                                 '8727-3703', '8341-12704', '8655-6103']:
                phi_guess += np.pi

            elif galaxy_ID[i] in ['7815-9102']:
                phi_guess -= np.pi

            elif galaxy_ID[i] in ['7443-9101', '7443-3704']:
                phi_guess -= 1.06 * np.pi

            elif galaxy_ID[i] in ['8082-1901', '8078-3703', '8551-1902', '9039-3703',
                                 '8624-1902', '8948-12702', '8443-6102', '8259-1901']:
                phi_guess += 1.5 * np.pi

            elif galaxy_ID[i] in ['8241-12705', '8326-6102']:
                phi_guess += 1.75 * np.pi

            elif galaxy_ID[i] in ['7443-6103']:
                phi_guess += 2.3 * np.pi

            # elif gal_ID in ['8655-1902', '7960-3701', '9864-9101', '8588-3703']:
            #     phi_guess = phi_EofN_deg * np.pi / 180.

            #print(phi_guess * 180 / (np.pi), flush=True)

            phi_guess = phi_guess % (2 * np.pi)

            parameters = [incl, phi_guess, x_center_guess, y_center_guess]
            ####################################################################

            Isothermal_fit_mini = list(fit_mini_iso[i])
            chi2_iso_norm = Isothermal_fit_mini[-1]

            ####################################################################
            # MCMC
            # If chi2 value of any model for this galaxy > 200 from minize
            # Run MCMC
            #-------------------------------------------------------------------
            if not np.isnan(chi2_iso_norm) and (chi2_iso_norm >= 150 or chi2_iso_norm < 200):
                Isothermal_fit_MCMC, chi2_iso_norm_MCMC = run_MCMC(galaxy_ID[i],MANGA_FOLDER,parameters,scale,'iso')
                c_iso_MCMC['A'][i] = Isothermal_fit_MCMC[0]
                c_iso_MCMC['Vin'][i] = Isothermal_fit_MCMC[1]
                c_iso_MCMC['SigD'][i] = Isothermal_fit_MCMC[2]
                c_iso_MCMC['Rd'][i] = Isothermal_fit_MCMC[3]
                c_iso_MCMC['rho0_h'][i] = Isothermal_fit_MCMC[4]
                c_iso_MCMC['Rh'][i] = Isothermal_fit_MCMC[5]
                c_iso_MCMC['incl'][i] = Isothermal_fit_MCMC[6]
                c_iso_MCMC['phi'][i] = Isothermal_fit_MCMC[7]
                c_iso_MCMC['x_cen'][i] = Isothermal_fit_MCMC[8]
                c_iso_MCMC['y_cen'][i] = Isothermal_fit_MCMC[9]
                c_iso_MCMC['Vsys'][i] = Isothermal_fit_MCMC[10]
                c_iso_MCMC['chi2'][i] = chi2_iso_norm_MCMC
            else:
                print(galaxy_ID[i] + ' have good fits from minimize')

            ####################################################################
            print('MCMC Isothermal chi2:', chi2_iso_norm_MCMC, time.time() - start_time)
            ####################################################################
            ####################################################################
            


            ####################################################################
            # Write results to file
            #-------------------------------------------------------------------
            '''
            writer_iso.writerow([galaxy_ID[i], 
                                 Isothermal_fit[0], 
                                 Isothermal_fit[1], 
                                 Isothermal_fit[2], 
                                 Isothermal_fit[3], 
                                 Isothermal_fit[4], 
                                 Isothermal_fit[5], 
                                 Isothermal_fit[6], 
                                 Isothermal_fit[7], 
                                 Isothermal_fit[8], 
                                 Isothermal_fit[9], 
                                 Isothermal_fit[10], 
                                 chi2_iso_norm])
            '''
            '''
            c_iso['A'][i] = Isothermal_fit[0]
            c_iso['Vin'][i] = Isothermal_fit[1]
            c_iso['SigD'][i] = Isothermal_fit[2]
            c_iso['Rd'][i] = Isothermal_fit[3]
            c_iso['rho0_h'][i] = Isothermal_fit[4]
            c_iso['Rh'][i] = Isothermal_fit[5]
            c_iso['incl'][i] = Isothermal_fit[6]
            c_iso['phi'][i] = Isothermal_fit[7]
            c_iso['x_cen'][i] = Isothermal_fit[8]
            c_iso['y_cen'][i] = Isothermal_fit[9]
            c_iso['Vsys'][i] = Isothermal_fit[10]
            c_iso['chi2'][i] = chi2_iso_norm
            '''
            '''
            writer_nfw.writerow([galaxy_ID[i], 
                                 NFW_fit[0], 
                                 NFW_fit[1], 
                                 NFW_fit[2], 
                                 NFW_fit[3], 
                                 NFW_fit[4], 
                                 NFW_fit[5], 
                                 NFW_fit[6], 
                                 NFW_fit[7], 
                                 NFW_fit[8], 
                                 NFW_fit[9], 
                                 NFW_fit[10], 
                                 chi2_NFW_norm])
            '''
            '''
            c_nfw['A'][i] = NFW_fit[0]
            c_nfw['Vin'][i] = NFW_fit[1]
            c_nfw['SigD'][i] = NFW_fit[2]
            c_nfw['Rd'][i] = NFW_fit[3]
            c_nfw['rho0_h'][i] = NFW_fit[4]
            c_nfw['Rh'][i] = NFW_fit[5]
            c_nfw['incl'][i] = NFW_fit[6]
            c_nfw['phi'][i] = NFW_fit[7]
            c_nfw['x_cen'][i] = NFW_fit[8]
            c_nfw['y_cen'][i] = NFW_fit[9]
            c_nfw['Vsys'][i] = NFW_fit[10]
            c_nfw['chi2'][i] = chi2_NFW_norm
            '''
            '''
            writer_bur.writerow([galaxy_ID[i], 
                                 Burket_fit[0],
                                 Burket_fit[1],
                                 Burket_fit[2],
                                 Burket_fit[3],
                                 Burket_fit[4],
                                 Burket_fit[5],
                                 Burket_fit[6],
                                 Burket_fit[7],
                                 Burket_fit[8],
                                 Burket_fit[9],
                                 Burket_fit[10],
                                 chi2_bur_norm])
            '''
            '''
            c_bur['A'][i] = Burket_fit[0]
            c_bur['Vin'][i] = Burket_fit[1]
            c_bur['SigD'][i] = Burket_fit[2]
            c_bur['Rd'][i] = Burket_fit[3]
            c_bur['rho0_h'][i] = Burket_fit[4]
            c_bur['Rh'][i] = Burket_fit[5]
            c_bur['incl'][i] = Burket_fit[6]
            c_bur['phi'][i] = Burket_fit[7]
            c_bur['x_cen'][i] = Burket_fit[8]
            c_bur['y_cen'][i] = Burket_fit[9]
            c_bur['Vsys'][i] = Burket_fit[10]
            c_bur['chi2'][i] = chi2_bur_norm
            '''
            ####################################################################

            '''
            ####################################################################
            # Calculating the Hessian Matrix
            #-------------------------------------------------------------------
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
            ####################################################################
            '''
        
        else:
            print(galaxy_ID[i] + ' does not have rotation curve', flush=True)
            '''
            writer_iso.writerow([galaxy_ID[i], 
                                 'N/A', 
                                 'N/A', 
                                 'N/A', 
                                 'N/A', 
                                 'N/A', 
                                 'N/A', 
                                 'N/A', 
                                 'N/A', 
                                 'N/A', 
                                 'N/A',
                                 'N/A',
                                 'N/A'])
            writer_nfw.writerow([galaxy_ID[i], 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A','N/A','N/A'])
            writer_bur.writerow([galaxy_ID[i], 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A','N/A','N/A'])
            '''
    else:
        print('No data for the galaxy.', flush=True)
        '''
        writer_iso.writerow([galaxy_ID[i], 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A','N/A'])
        writer_nfw.writerow([galaxy_ID[i], 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A','N/A'])
        writer_bur.writerow([galaxy_ID[i], 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A','N/A'])
        '''
'''
c_iso.close()
c_nfw.close()
c_bur.close()
'''

#c_iso.write('iso_mini.csv', format='ascii.csv', overwrite=True)
#c_nfw.write('nfw_mini.csv', format='ascii.csv', overwrite=True)
#c_bur.write('bur_mini.csv', format='ascii.csv', overwrite=True)
#c_scale.write('gal_scale.csv', format='ascii.csv',overwrite=True)
c_iso_MCMC.write('iso_mcmc.csv', format='ascii.csv', overwrite=True)


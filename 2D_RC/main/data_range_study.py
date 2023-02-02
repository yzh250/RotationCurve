from astropy.io import ascii
from astropy.table import Table,QTable
import numpy as np
import numpy.ma as ma
#import pandas as pd
#import corner
import matplotlib.pyplot as plt
#%matplotlib notebook
#import emcee
from astropy.io import fits
from astropy.table import Table, QTable

import sys
sys.path.insert(1, '/home/yzh250/Documents/UR_Stuff/Research_UR/RotationCurve/2D_RC/main/')
#sys.path.insert(1, '/Users/kellydouglass/Documents/Research/Rotation_curves/Yifan_Zhang/RotationCurve/2D_RC/main/')
from RC_2D_Fit_Functions import Galaxy_Data, getTidal
from Velocity_Map_Functions_cython import rot_incl_iso, rot_incl_NFW, rot_incl_bur

#MANGA_FOLDER = '/Users/richardzhang/Documents/UR_Stuff/Research_UR/SDSS/dr17/manga/spectro/'
MANGA_FOLDER = '/home/yzh250/Documents/UR_Stuff/Research_UR/SDSS/dr17/manga/spectro/'
#MANGA_FOLDER = '/Users/kellydouglass/Documents/Research/data/SDSS/dr16/manga/spectro/'

DRP_FILENAME = MANGA_FOLDER + 'redux/v3_1_1/drpall-v3_1_1.fits'

VEL_MAP_FOLDER = '/scratch/kdougla7/data/SDSS/dr17/manga/spectro/analysis/v3_1_1/3.1.0/HYB10-MILESHC-MASTARSSP/'

from Velocity_Map_Functions_cython import rot_incl_iso,\
                                          rot_incl_NFW, \
                                          rot_incl_bur           

from galaxy_component_functions_cython import vel_tot_iso,\
                                              vel_tot_NFW,\
                                              vel_tot_bur,\
                                              bulge_vel,\
                                              disk_vel,\
                                              halo_vel_iso,\
                                              halo_vel_NFW,\
                                              halo_vel_bur
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
################################################################################
# Physics Constants
#-------------------------------------------------------------------------------
c = 3E5 # km * s ^1
h = 1 # reduced hubble constant
H_0 =  100 * h # km * s^-1 * Mpc^-1
q0 = 0.2 # minimum inclination value
################################################################################

# DRP all table
DTable =  Table.read(DRP_FILENAME, format='fits')

#MTable =  Table.read(MORPH_file, format='fits')

DRP_index = {}

for i in range(len(DTable)):
    gal_ID = DTable['plateifu'][i]

    DRP_index[gal_ID] = i
    
m = DTable['nsa_elpetro_mass']
rat = DTable['nsa_elpetro_ba']
phi = DTable['nsa_elpetro_phi']
z = DTable['nsa_z']
r50_ang = DTable['nsa_elpetro_th50_r']

r90_file = '/home/yzh250/Documents/UR_Stuff/Research_UR/RotationCurve/2D_RC/main/r90.csv'
################################################################################

                                     

# Important fitting results
iso_fits = Table.read('iso_fits_final.csv',format='ascii.csv')
nfw_fits = Table.read('nfw_fits_final.csv',format='ascii.csv')
bur_fits = Table.read('bur_fits_final.csv',format='ascii.csv')

# Modified velocity map deprojection function
def vel_map_depro(mHa_vel, best_fit_values, scale):
    i_angle = best_fit_values[6]#np.arccos(best_fit_values['ba'])
    ############################################################################


    ############################################################################
    # Convert rotation angle from degrees to radians
    #---------------------------------------------------------------------------
    phi = best_fit_values[7]
    ############################################################################


    ############################################################################
    # Deproject all data values in the given velocity map
    #---------------------------------------------------------------------------
    vel_array_shape = mHa_vel.shape

    r_deproj = np.zeros(vel_array_shape)
    v_deproj = np.zeros(vel_array_shape)

    theta = np.zeros(vel_array_shape)

    for i in range(vel_array_shape[0]):
        for j in range(vel_array_shape[1]):

            r_deproj[i,j], theta[i,j] = deproject_spaxel((i,j), 
                                                         (best_fit_values[8], best_fit_values[9]), 
                                                         phi, 
                                                         i_angle)

            ####################################################################
            # Find the sign of r_deproj
            #-------------------------------------------------------------------
            if np.cos(theta[i,j]) < 0:
                r_deproj[i,j] *= -1
            ####################################################################

    # Scale radii to convert from spaxels to kpc
    r_deproj *= scale

    # Deproject velocity values
    v_deproj = (mHa_vel - best_fit_values[10])/np.abs(np.cos(theta))
    v_deproj /= np.sin(i_angle)

    # Apply mask to arrays
    rm_deproj = ma.array(r_deproj, mask=mHa_vel.mask)
    vm_deproj = ma.array(v_deproj, mask=mHa_vel.mask)
    
    return rm_deproj, vm_deproj

# deprojected rotation curve
def rot_curve(best_fit_values,halo_model):
    r = np.linspace(0,1000,5000)
    
    v_b = np.zeros(len(r))
    v_d = np.zeros(len(r))
    v_h = np.zeros(len(r))
    v = np.zeros(len(r))
    
    for oo in range(len(r)):
        if r[oo] > 0:
            v_b[oo] = bulge_vel(r[oo]*1000,best_fit_values[0],best_fit_values[1]*1000)
            v_d[oo] = disk_vel(r[oo]*1000,best_fit_values[2],best_fit_values[3]*1000)
            if halo_model == 'Isothermal':
                v_h[oo] = halo_vel_iso(r[oo]*1000,best_fit_values[4],best_fit_values[5]*1000)
                v[oo] = vel_tot_iso(r[oo],best_fit_values[0],best_fit_values[1],best_fit_values[2],best_fit_values[3],best_fit_values[4],best_fit_values[5])
            elif halo_model == 'NFW':
                v_h[oo] = halo_vel_NFW(r[oo]*1000,best_fit_values[4],best_fit_values[5]*1000)
                v[oo] = vel_tot_NFW(r[oo],best_fit_values[0],best_fit_values[1],best_fit_values[2],best_fit_values[3],best_fit_values[4],best_fit_values[5])
            elif halo_model == 'Burkert':
                v_h[oo] = halo_vel_bur(r[oo]*1000,best_fit_values[4],best_fit_values[5]*1000)
                v[oo] = vel_tot_bur(r[oo],best_fit_values[0],best_fit_values[1],best_fit_values[2],best_fit_values[3],best_fit_values[4],best_fit_values[5])
            else:
                print('Fit function not known.  Please update plot_rot_curve function.')
        else:
            v_b[oo] = -bulge_vel(np.abs(r[oo]*1000),best_fit_values[0],best_fit_values[1]*1000)
            v_d[oo] = -disk_vel(np.abs(r[oo]*1000),best_fit_values[2],best_fit_values[3]*1000)
            if halo_model == 'Isothermal':
                v_h[oo] = -halo_vel_iso(np.abs(r[oo]*1000),best_fit_values[4],best_fit_values[5]*1000)
                v[oo] = -vel_tot_iso(np.abs(r[oo]),best_fit_values[0],best_fit_values[1],best_fit_values[2],best_fit_values[3],best_fit_values[4],best_fit_values[5])
            elif halo_model == 'NFW':
                v_h[oo] = -halo_vel_NFW(np.abs(r[oo]*1000),best_fit_values[4],best_fit_values[5]*1000)
                v[oo] = -vel_tot_NFW(np.abs(r[oo]),best_fit_values[0],best_fit_values[1],best_fit_values[2],best_fit_values[3],best_fit_values[4],best_fit_values[5])
            elif halo_model == 'Burkert':
                v_h[oo] = -halo_vel_bur(np.abs(r[oo]*1000),best_fit_values[4],best_fit_values[5]*1000)
                v[oo] = -vel_tot_bur(np.abs(r[oo]),best_fit_values[0],best_fit_values[1],best_fit_values[2],best_fit_values[3],best_fit_values[4],best_fit_values[5])
            else:
                print('Fit function not known.  Please update plot_rot_curve function.')
    return r, v_b, v_d, v_h, v

# Modified function to obtain information for a given galaxy
def get_info(galaxy_ID,fit, r90_file, flag):
    
    j = DRP_index[galaxy_ID]

    redshift = z[j]
    velocity =  redshift* c
    distance = (velocity / H_0) * 1000 #kpc
    scale = 0.5 * distance / 206265
    
    #incl = np.arccos(rat[j])
    cosi2 = (rat[j]**2 - q0**2)/(1 - q0**2)
    if cosi2 < 0:
        cosi2 = 0

    incl = np.arccos(np.sqrt(cosi2))
    
    print(galaxy_ID)
    
    plate, IFU = galaxy_ID.split('-')
    
    #map_file_name = 'manga-' + galaxy_ID + '-MAPS-HYB10-MILESHC-MASTARSSP.fits.gz'
    
    #print(map_file_name)
    cube = fits.open(VEL_MAP_FOLDER + plate + '/' + IFU + '/manga-' + galaxy_ID + '-MAPS-HYB10-MILESHC-MASTARSSP.fits.gz')
    maps = {}

    # bluehive
    maps['r_band'] = cube['SPX_MFLUX'].data
    maps['Ha_vel'] = cube['EMLINE_GVEL'].data[23]
    maps['Ha_vel_ivar'] = cube['EMLINE_GVEL_IVAR'].data[23]
    maps['Ha_vel_mask'] = cube['EMLINE_GVEL_MASK'].data[23]


    maps['vmasked'] = ma.array(maps['Ha_vel'], mask=maps['Ha_vel_mask'])
    #maps['r_band_masked'] = ma.array(maps['r_band'],mask=maps['Ha_vel_mask'])
    maps['ivar_masked'] = ma.array(maps['Ha_vel_ivar'], mask=maps['Ha_vel_mask'])

    gshape = maps['vmasked'].shape
    ############################################################################

    # Ha flux
    maps['Ha_flux'] = cube['EMLINE_GFLUX'].data[23]
    maps['Ha_flux_ivar'] = cube['EMLINE_GFLUX_IVAR'].data[23]
    maps['Ha_flux_mask'] = cube['EMLINE_GFLUX_MASK'].data[23]
    maps['Ha_flux_masked'] = ma.array(maps['Ha_flux'], mask=maps['Ha_flux_mask'])
    
    SN_map = maps['Ha_flux'] * np.sqrt(maps['Ha_flux_ivar'])
    Ha_vel_mask = maps['Ha_vel_mask'] + (SN_map < 5)

    vmasked = ma.array(maps['Ha_vel'], mask = Ha_vel_mask)
    ivar_masked = ma.array(maps['Ha_vel_ivar'], mask = Ha_vel_mask)
    
    '''
    for ii in range(len(fit_cat_mini)):
        if fit_cat_mini['galaxy_ID'][ii] == galaxy_ID:
            mini_fit = list(fit_cat_mini[ii])
            mcmc_fit = list(fit_cat_mcmc[ii])
            chi2_mini = mini_fit[-1]
            chi2_mcmc = mcmc_fit[-1]
            mini_fit = mini_fit[3:-1]
            mcmc_fit = mcmc_fit[3:-1]
    '''
    
    #print(str(galaxy_ID) + ' minimize fit: ')
    #print(mini_fit)
    #print(str(galaxy_ID) + ' mcmc fit: ')
    #print(mcmc_fit)
    
    f_r90 = Table.read(r90_file,format='ascii.csv')
    
    r90 = 0
    for jj in range(len(f_r90)):
        if f_r90['galaxy_ID'][jj] == galaxy_ID:
            r90 = f_r90['r90'][jj]
            
    if flag == 'iso':
            
        fitted_map = rot_incl_iso(gshape, scale, fit)

        mfitted_map = ma.array(fitted_map, mask=Ha_vel_mask)
        
        r, v_b, v_d, v_h, v = rot_curve(fit,'Isothermal')
        
    elif flag == 'nfw':
            
        fitted_map = rot_incl_NFW(gshape, scale, fit)

        mfitted_map = ma.array(fitted_map, mask=Ha_vel_mask)
        
        r, v_b, v_d, v_h, v = rot_curve(fit,'NFW')
        
    elif flag == 'bur':
            
        fitted_map = rot_incl_bur(gshape, scale, fit)

        mfitted_map = ma.array(fitted_map, mask=Ha_vel_mask)
        
        r, v_b, v_d, v_h, v = rot_curve(fit,'Burkert')

    return vmasked, ivar_masked, incl, scale, r90, mfitted_map, r, v_d, v_h, v

# Extracting r90 and r_max

galaxy_ID_iso = iso_fits['galaxy_ID']
galaxy_ID_nfw = nfw_fits['galaxy_ID']
galaxy_ID_bur = bur_fits['galaxy_ID']

r90_iso = []
rmax_iso = []
r90_nfw = []
rmax_nfw = []
r90_bur = []
rmax_bur = []

for a in range(len(galaxy_ID_iso)):

    gal_ID = galaxy_ID_iso[a]

    fit = list(iso_fits[a])[3:-1]

    vmasked, ivar_masked, incl, scale, r90, mfitted_map, r, v_d, v_h, v = get_info(gal_ID,fit, r90_file, 'iso')

    rm_deproj, vm_deproj = vel_map_depro(vmasked, fit, scale)

    r90_iso.append(r90)
    rmax_iso.append(np.max(rm_deproj))

for b in range(len(galaxy_ID_nfw)):

    gal_ID = galaxy_ID_nfw[b]

    fit = list(nfw_fits[b])[3:-1]

    vmasked, ivar_masked, incl, scale, r90, mfitted_map, r, v_d, v_h, v = get_info(gal_ID,fit, r90_file, 'nfw')

    rm_deproj, vm_deproj = vel_map_depro(vmasked, fit, scale)

    r90_nfw.append(r90)
    rmax_nfw.append(np.max(rm_deproj))

for c in range(len(galaxy_ID_iso)):

    gal_ID = galaxy_ID_bur[c]

    fit = list(bur_fits[c])[3:-1]

    vmasked, ivar_masked, incl, scale, r90, mfitted_map, r, v_d, v_h, v = get_info(gal_ID,fit, r90_file, 'bur')

    rm_deproj, vm_deproj = vel_map_depro(vmasked, fit, scale)

    r90_bur.append(r90)
    rmax_bur.append(np.max(rm_deproj))

c_iso = Table()
c_iso['galaxy_ID'] = galaxy_ID_iso
c_iso['r90'] = r90_iso
c_iso['r_max'] = rmax_iso
c_nfw = Table()
c_nfw['galaxy_ID'] = galaxy_ID_nfw
c_nfw['r90'] = r90_nfw
c_nfw['r_max'] = rmax_nfw
c_bur = Table()
c_bur['galaxy_ID'] = galaxy_ID_bur
c_bur['r90'] = r90_bur
c_bur['r_max'] = rmax_bur


c_iso = Table.read('iso_data_range.csv',format='ascii.csv')
c_nfw = Table.read('nfw_data_range.csv',format='ascii.csv')
c_bur = Table.read('bur_data_range.csv',format='ascii.csv')
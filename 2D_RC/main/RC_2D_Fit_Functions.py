################################################################################
# Import modules
#-------------------------------------------------------------------------------
import matplotlib.pyplot as plt

import numpy as np
import numpy.ma as ma

from astropy.io import fits
from astropy.table import Table, QTable

from scipy.optimize import minimize, Bounds
#from scipy_minimize_custom import minimize_powell

import numdifftools as ndt

# Import functions from other .py files
from Velocity_Map_Functions import loglikelihood_iso_flat,\
                                   loglikelihood_NFW_flat, \
                                   loglikelihood_bur_flat,\
                                   nloglikelihood_iso_flat,\
                                   nloglikelihood_NFW_flat,\
                                   nloglikelihood_bur_flat,\
                                   loglikelihood_iso_flat_constraints,\
                                   loglikelihood_NFW_flat_constraints,\
                                   loglikelihood_bur_flat_constraints

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

import csv
import emcee
################################################################################




################################################################################
# Physics Constants
#-------------------------------------------------------------------------------
c = 3E5 # km/s
h = 1 # reduced hubble constant
H_0 =  100 * h # km * s^-1 * Mpc^-1
################################################################################

'''
################################################################################
# Used files
#-------------------------------------------------------------------------------
DTable1 = QTable.read('Master_Table.txt',format='ascii.commented_header')
DTable2 = QTable.read('DRPall-master_file.txt',format='ascii.ecsv')
################################################################################




################################################################################
# Get the Mass of stars & redshifts & angular resolution of r50
#-------------------------------------------------------------------------------
m = DTable1['NSA_Mstar'].data
rat = DTable1['NSA_ba'].data
phi = DTable1['NSA_phi'].data
z = DTable2['redshift'].data
r50_ang = DTable2['nsa_elpetro_th50_r'].data
################################################################################
'''



def Galaxy_Data(galaxy_ID, VEL_MAP_FOLDER):
    '''
    PARAMETERS
    ==========

    galaxy_ID : string
        'Plate-IFU'
        
    MANGA_FOLDER : string
        Location of the MaNGA data on the current computer system


    RETURNS
    =======

    physical properties & data of the galaxy

    '''

    plate, IFU = galaxy_ID.split('-')
    '''
    ############################################################################
    # Obtain redshift
    #---------------------------------------------------------------------------
    for i in range(len(DTable2)):
        if DTable2['MaNGA_plate'][i] == int(plate) and DTable2['MaNGA_IFU'][i] == int(IFU):
            redshift = z[i]
            velocity = redshift * c
            distance = (velocity / H_0) * 1E3 #kpc
            scale = 0.5 * (distance) / 206265
    ############################################################################
    '''
    '''
    ############################################################################
    # Obtain inclination
    #---------------------------------------------------------------------------
    incl = 0
    for i in range(len(DTable1)):
        if DTable1['MaNGA_plate'][i] == int(plate) and DTable1['MaNGA_IFU'][i] == int(IFU):
            incl = np.arccos(rat[i])
    ############################################################################
    '''
    '''
    ############################################################################
    # Obtain phi
    #---------------------------------------------------------------------------
    ph = 0
    for i in range(len(DTable1)):
        if DTable1['MaNGA_plate'][i] == int(plate) and DTable1['MaNGA_IFU'][i] == int(IFU):
            ph = phi[i] * np.pi / 180
    ############################################################################
    '''

    ############################################################################
    # Obtaining Data Cubes, Inverse Variances, and Masks
    #---------------------------------------------------------------------------
    
    #bluehive
    cube = fits.open(VEL_MAP_FOLDER + plate + '/' + IFU + '/manga-' + galaxy_ID + '-MAPS-HYB10-MILESHC-MASTARSSP.fits.gz')
    #local
    #cube = fits.open(VEL_MAP_FOLDER + plate + '/' + IFU + '/manga-' + galaxy_ID + '-MAPS-HYB10-GAU-MILESHC.fits.gz')

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

    #local
    '''
    maps['r_band'] = cube['SPX_MFLUX'].data
    maps['Ha_vel'] = cube['EMLINE_GVEL'].data[18]
    maps['Ha_vel_ivar'] = cube['EMLINE_GVEL_IVAR'].data[18]
    maps['Ha_vel_mask'] = cube['EMLINE_GVEL_MASK'].data[18]


    maps['vmasked'] = ma.array(maps['Ha_vel'], mask=maps['Ha_vel_mask'])
    #maps['r_band_masked'] = ma.array(maps['r_band'],mask=maps['Ha_vel_mask'])
    maps['ivar_masked'] = ma.array(maps['Ha_vel_ivar'], mask=maps['Ha_vel_mask'])

    gshape = maps['vmasked'].shape
    ############################################################################

    # Ha flux
    maps['Ha_flux'] = cube['EMLINE_GFLUX'].data[18]
    maps['Ha_flux_ivar'] = cube['EMLINE_GFLUX_IVAR'].data[18]
    maps['Ha_flux_mask'] = cube['EMLINE_GFLUX_MASK'].data[18]
    maps['Ha_flux_masked'] = ma.array(maps['Ha_flux'], mask=maps['Ha_flux_mask'])
    '''

    ############################################################################
    # Finding the center
    #---------------------------------------------------------------------------
    '''
    center_guess = np.unravel_index(ma.argmax(maps['r_band_masked']), gshape)
    x_center_guess = center_guess[0]
    y_center_guess = center_guess[1]
    '''
    ############################################################################

    #return r_band, Ha_vel, Ha_vel_ivar, Ha_vel_mask, Ha_flux, Ha_flux_ivar, Ha_flux_mask, vmasked, Ha_flux_masked, ivar_masked, gshape, x_center_guess, y_center_guess
    return maps, gshape#, x_center_guess, y_center_guess





def getTidal(gal_ID, MORPH_file_path):

    MORPH_file = MORPH_file_path + 'manga_visual_morpho-2.0.1.fits'

    Mfile = fits.open(MORPH_file)
    Mdata = Mfile[1].data
    Mfile.close()

    Mdata_index = Mdata['NAME'] == 'manga-' + gal_ID

    return Mdata['TIDAL'][Mdata_index]




def Galaxy_Fitting_iso(params, scale, shape, vmap, ivar, mask):
    '''

    :param params:
    :param scale:
    :param shape:
    :param vmap:
    :param ivar:
    :return:
    '''
    '''
    plt.figure()
    plt.imshow(ma.array(vmap, mask=mask), cmap='RdBu_r', origin='lower')
    plt.colorbar()
    plt.show()
    '''
    incl, ph, x_guess, y_guess = params

    # Isothermal Fitting
    bounds_iso = [[-7, 1], # Bulge density [log(Msun/pc^3)]
                  #[0, 10], # Bulge density [Msun/pc^3]
                  [0, 10],  # Bulge radius [kpc/h]
                  [0.1, 3000],  # Surface Density [Msol/pc^2]
                  [0.1, 30],  # Disk radius [kpc/h]
                  [-7, 2],  # Halo density [log(Msun/pc^3)]
                  #[0, 100],  # Halo density [Msun/pc^3]
                  [0.1, 1000],  # Halo radius [kpc]
                  [0.1, 0.436*np.pi],  # Inclination angle
                  [0, 2.2 * np.pi],  # Phase angle
                  [x_guess-10, x_guess+10],  # center_x
                  [y_guess-10, y_guess+10], # center_y
                  [-100,100]] # systemic velocity

    vsys = 0

    ig_iso = [-1, 1, 1000, 4, -3, 25, incl, ph, x_guess, y_guess, vsys]
    #ig_iso = [-1, 1, 1000, 4, 0.001, 25, incl, ph, x_guess, y_guess, vsys]
    #ig_iso = [0.0001, 4, 2000, 25, 5, 250, incl, ph, x_guess, y_guess, vsys]
    print(ig_iso)

    bestfit_iso = minimize(nloglikelihood_iso_flat,
                           ig_iso,
                           args=(scale, 
                                 shape, 
                                 vmap.compressed(), 
                                 ivar.compressed(),mask),
                           method='Powell',
                           bounds=bounds_iso)
    print('---------------------------------------------------')
    print(bestfit_iso)

    return bestfit_iso.x

'''
def Galaxy_Fitting_iso_nb(params, scale, shape, vmap, ivar):
    

    :param params:
    :param scale:
    :param shape:
    :param vmap:
    :param ivar:
    :return:
    

    incl, ph, x_guess, y_guess = params

    # Isothermal Fitting
    bounds_iso = [[0, 10000],  # Surface Density [Msol/pc^2]
                  [0.1, 30],  # Disk radius [kpc]
                  [0.0001, 0.1],  # Halo density [Msun/pc^2]
                  [0.1, 1000],  # Halo radius [kpc]
                  [0.1, 0.5*np.pi],  # Inclination angle
                  [0, 2 * np.pi],  # Phase angle
                  [x_guess-10, x_guess+10],  # center_x
                  [y_guess-10, y_guess+10], # center_y
                  [-100,100]] # systemic velocity

    vsys = 0

    ig_iso = [1000, 4, 0.006, 25, incl, ph, x_guess, y_guess, vsys]

    bestfit_iso = minimize(nloglikelihood_iso_nb,
                           ig_iso,
                           args=(scale, shape, vmap, ivar),
                           method='Powell',
                           bounds=bounds_iso)
    print('---------------------------------------------------')
    print(bestfit_iso)

    return bestfit_iso.x
'''

def Galaxy_Fitting_NFW(params, scale, shape, vmap, ivar, mask):
    '''

    :param params:
    :param scale:
    :param shape:
    :param vmap:
    :param ivar:
    :return:
    '''

    incl, ph, x_guess, y_guess = params

    # NFW Fitting
    bounds_NFW = [[-7, 1], # Bulge density [log(Msun/pc^3)]
                  #[0, 10], # Bulge density [Msun/pc^3]
                  [0, 5],  # Bulge radius [kpc/h]
                  [0.1, 3000], # Surface Density [Msol/pc^2]
                  [0.1, 30],   # Disk radius [kpc/h]
                  [-7, 2],     # Halo density [log(Msun/pc^3)]
                  #[0, 100],    # Halo density [Msun/pc^3]
                  [0.1, 1000], # Halo radius [kpc]
                  [0.1, 0.436*np.pi], # Inclination angle
                  [0, 2.2 * np.pi], # Phase angle
                  [x_guess-10, x_guess+10], # center_x
                  [y_guess-10, y_guess+10], # center_y
                  [-100,100]] # systemic velocity

    vsys = 0

    ig_NFW = [-1, 1, 1000, 4, -3, 25, incl, ph, x_guess, y_guess, vsys]
    #ig_NFW = [0.1, 1, 1000, 4, 0.001, 25, incl, ph, x_guess, y_guess, vsys]
    print(ig_NFW)

    bestfit_NFW = minimize(nloglikelihood_NFW_flat,
                           ig_NFW, 
                           args=(scale, shape, vmap.compressed(), ivar.compressed(), mask),
                           method='Powell', 
                           bounds=bounds_NFW)
    print('---------------------------------------------------')
    print(bestfit_NFW)

    return bestfit_NFW.x

'''
def Galaxy_Fitting_NFW_nb(params, scale, shape, vmap, ivar):
    

    :param params:
    :param scale:
    :param shape:
    :param vmap:
    :param ivar:
    :return:
    

    incl, ph, x_guess, y_guess = params

    # NFW Fitting
    bounds_NFW = [[0, 10000],  # Surface Density [Msol/pc^2]
                  [0.1, 30],  # Disk radius [kpc]
                  [0.0001, 0.1],  # Halo density [Msun/pc^2]
                  [0.1, 1000],  # Halo radius [kpc]
                  [0.1, 0.5*np.pi],  # Inclination angle
                  [0, 2 * np.pi],  # Phase angle
                  [x_guess-10, x_guess+10],  # center_x
                  [y_guess-10, y_guess+10], # center_y
                  [-100,100]] # systemic velocity

    vsys = 0

    ig_NFW = [1000, 4, 0.006, 25, incl, ph, x_guess, y_guess, vsys]

    bestfit_NFW = minimize(nloglikelihood_NFW_nb,
                           ig_NFW,
                           args=(scale, shape, vmap, ivar),
                           method='Powell',
                           bounds=bounds_NFW)
    print('---------------------------------------------------')
    print(bestfit_NFW)

    return bestfit_NFW.x
'''

def Galaxy_Fitting_bur(params, scale, shape, vmap, ivar, mask):
    '''

    :param params:
    :param scale:
    :param shape:
    :param vmap:
    :param ivar:
    :return:
    '''

    incl, ph, x_guess, y_guess = params

    # Burket Fitting
    bounds_bur = [[-7, 1], # Bulge density [log(Msun/pc^3)]
                  #[0, 10], # Bulge density [Msun/pc^3]
                  [0, 5],  # Bulge radius [kpc/h]
                  [0.1, 3000], # Surface Density [Msol/pc^2]
                  [0.1, 30],   # Disk radius [kpc/h]
                  [-7, 2],    # Halo density [log(Msun/pc^3)]
                  #[0, 100],    # Halo density [Msun/pc^3]
                  [0.1, 1000], # Halo radius [kpc]
                  [0.1, 0.436*np.pi],  # Inclination angle
                  [0, 2.2 * np.pi],  # Phase angle
                  [x_guess-10, x_guess+10], # center_x
                  [y_guess-10, y_guess+10], # center_y
                  [-100,100]] # systemic velocity



    vsys = 0

    ig_bur = [-1, 1, 1000, 4, -3, 25, incl, ph, x_guess, y_guess, vsys]
    #ig_bur = [0.1, 1, 1000, 4, 0.001, 25, incl, ph, x_guess, y_guess, vsys]
    #print(ig_bur)

    bestfit_bur = minimize(nloglikelihood_bur_flat,
                           ig_bur, 
                           args=(scale, shape, vmap.compressed(), ivar.compressed(), mask),
                           method='Powell', 
                           bounds=bounds_bur)
    print('---------------------------------------------------')
    print(bestfit_bur)

    return bestfit_bur.x

'''
def Galaxy_Fitting_bur_nb(params, scale, shape, vmap, ivar):
    

    :param params:
    :param scale:
    :param shape:
    :param vmap:
    :param ivar:
    :return:

    incl, ph, x_guess, y_guess = params

    # Burket Fitting
    bounds_bur = [[0, 10000],  # Surface Density [Msol/pc^2]
                  [0.1, 30],  # Disk radius [kpc]
                  [0.0001, 0.1],  # Halo central density[km/s]
                  [0.1, 1000],  # Halo radius [kpc]
                  [0.1, 0.5*np.pi],  # Inclination angle
                  [0, 2 * np.pi],  # Phase angle
                  [x_guess-10, x_guess+10],  # center_x
                  [y_guess-10, y_guess+10], # center_y
                  [-100,100]] # systemic velocity



    vsys = 0

    ig_bur = [1000, 4, 0.006, 25, incl, ph, x_guess, y_guess, vsys]

    bestfit_bur = minimize(nloglikelihood_bur_nb,
                           ig_bur,
                           args=(scale, shape, vmap, ivar),
                           method='Powell',
                           bounds=bounds_bur)
    print('---------------------------------------------------')
    print(bestfit_bur)

    return bestfit_bur.x
'''

# Showing and Saving Images
# Isothermal
def Plotting_Isothermal(ID, shape, scale, fit_solution, mask, ax=None):
    '''

    :param ID:
    :param shape:
    :param scale:
    :param fit_solution:
    :param mask:
    :return:
    '''
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10,10))
        
    iso_map = ax.imshow(ma.array(rot_incl_iso(shape, scale, fit_solution), mask=mask), 
               origin='lower',
               cmap='RdBu_r')

    ax.set_title(ID + ' Isothermal Fit')

    ax.set_xlabel('spaxel')
    ax.set_ylabel('spaxel')

    cbar = plt.colorbar(iso_map, ax=ax)
    cbar.set_label('km/s')

    #plt.close()

# NFW
def Plotting_NFW(ID, shape, scale, fit_solution, mask, ax=None):
    '''

    :param ID:
    :param shape:
    :param scale:
    :param fit_solution:
    :param mask:
    :return:
    '''

    if ax is None:
        fig, ax = plt.subplots()
        
    NFW_map = ax.imshow(ma.array(rot_incl_NFW(shape, scale, fit_solution), mask=mask), 
               origin='lower',
               cmap='RdBu_r')

    ax.set_title(ID + ' NFW Fit')

    ax.set_xlabel('spaxel')
    ax.set_ylabel('spaxel')

    cbar = plt.colorbar(NFW_map, ax=ax)
    cbar.set_label('km/s')

    #plt.close()
    
# Burket
def Plotting_Burkert(ID, shape, scale, fit_solution, mask, ax=None):
    '''

    :param ID:
    :param shape:
    :param scale:
    :param fit_solution:
    :param mask:
    :return:
    '''

    if ax is None:
        fig, ax = plt.subplots()
        
    bur_map = ax.imshow(ma.array(rot_incl_bur(shape, scale, fit_solution), mask=mask), 
               origin='lower',
               cmap='RdBu_r')

    ax.set_title(ID + ' Burkert Fit')

    ax.set_xlabel('spaxel')
    ax.set_ylabel('spaxel')

    cbar = plt.colorbar(bur_map, ax=ax)
    cbar.set_label('km/s')

    #plt.savefig(ID + ' Burket.png', format='png')
    #plt.close()

################################################################################
def deproject_spaxel(coords, center, phi, i_angle):
    '''
    Calculate the deprojected radius for the given coordinates in the map.


    PARAMETERS
    ==========

    coords : length-2 tuple
        (i,j) coordinates of the current spaxel

    center : length-2 tuple
        (i,j) coordinates of the galaxy's center

    phi : float
        Rotation angle (in radians) east of north of the semi-major axis.

    i_angle : float
        Inclination angle (in radians) of the galaxy.


    RETURNS
    =======

    r : float
        De-projected radius from the center of the galaxy for the given spaxel 
        coordinates.
    '''


    # Distance components between center and current location
    delta = np.subtract(coords, center)

    # x-direction distance relative to the semi-major axis
    dx_prime = (delta[1]*np.cos(phi) + delta[0]*np.sin(phi))/np.cos(i_angle)

    # y-direction distance relative to the semi-major axis
    dy_prime = (-delta[1]*np.sin(phi) + delta[0]*np.cos(phi))

    # De-projected radius for the current point
    r = np.sqrt(dx_prime**2 + dy_prime**2)

    # Angle (counterclockwise) between North and current position
    theta = np.arctan2(-dx_prime, dy_prime)

    return r, theta
################################################################################





################################################################################
def plot_rot_curve(mHa_vel, 
                   mHa_vel_ivar,
                   best_fit_values,
                   scale,
                   gal_ID, 
                   halo_model,
                   IMAGE_DIR=None, 
                   IMAGE_FORMAT='jpg',
                   ax=None):
    '''
    Plot the galaxy rotation curve.


    PARAMETERS
    ==========

    mHa_vel : numpy ndarray of shape (n,n)
        Masked H-alpha velocity array

    mHa_vel_ivar : numpy ndarray of shape (n,n)
        Masked array of the inverse variance of the H-alpha velocity 
        measurements

    best_fit_values : dictionary
        Best-fit values for the velocity map

    scale : float
        Pixel scale (to convert from pixels to kpc)

    gal_ID : string
        MaNGA <plate>-<IFU> for the current galaxy

    fit_function : string
        Determines which function to use for the velocity.  Options are 'BB' and 
        'tanh'.

    IMAGE_DIR : str
        Path of directory in which to store plot.
        Default is None (image will not be saved)

    IMAGE_FORMAT : str
        Format of saved plot
        Default is 'eps'

    ax : matplotlib.pyplot figure axis object
        Axis handle on which to create plot
    '''


    if ax is None:
        fig, ax = plt.subplots(figsize=(3,3))


    ############################################################################
    # Convert axis ratio to angle of inclination
    #---------------------------------------------------------------------------
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
    ############################################################################


    ############################################################################
    # Calculate functional form of rotation curve
    #---------------------------------------------------------------------------
    r = np.linspace(ma.min(rm_deproj), ma.max(rm_deproj), 100)

    #if fit_function == 'BB':
        #v = rot_fit_BB(r, [best_fit_values['v_max'], 
                           #best_fit_values['r_turn'], 
                           #best_fit_values['alpha']])
    #elif fit_function == 'tanh':
        #v = rot_fit_tanh(r, [best_fit_values['v_max'], 
                             #best_fit_values['r_turn']])
    v_b = np.zeros(len(r))
    v_d = np.zeros(len(r))
    v_h = np.zeros(len(r))
    v = np.zeros(len(r))

    for i in range(len(r)):
        if r[i] > 0:
            v_b[i] = bulge_vel(r[i]*1000,best_fit_values[0],best_fit_values[1]*1000)
            v_d[i] = disk_vel(r[i]*1000,best_fit_values[2],best_fit_values[3]*1000)
            if halo_model == 'Isothermal':
                v_h[i] = halo_vel_iso(r[i]*1000,best_fit_values[4],best_fit_values[5]*1000)
                v[i] = vel_tot_iso(r[i],best_fit_values[0],best_fit_values[1],best_fit_values[2],best_fit_values[3],best_fit_values[4],best_fit_values[5])
            elif halo_model == 'NFW':
                v_h[i] = halo_vel_NFW(r[i]*1000,best_fit_values[4],best_fit_values[5]*1000)
                v[i] = vel_tot_NFW(r[i],best_fit_values[0],best_fit_values[1],best_fit_values[2],best_fit_values[3],best_fit_values[4],best_fit_values[5])
            elif halo_model == 'Burkert':
                v_h[i] = halo_vel_bur(r[i]*1000,best_fit_values[4],best_fit_values[5]*1000)
                v[i] = vel_tot_bur(r[i],best_fit_values[0],best_fit_values[1],best_fit_values[2],best_fit_values[3],best_fit_values[4],best_fit_values[5])
            else:
                print('Fit function not known.  Please update plot_rot_curve function.')
        else:
            v_b[i] = -bulge_vel(np.abs(r[i]*1000),best_fit_values[0],best_fit_values[1]*1000)
            v_d[i] = -disk_vel(np.abs(r[i]*1000),best_fit_values[2],best_fit_values[3]*1000)
            if halo_model == 'Isothermal':
                v_h[i] = -halo_vel_iso(np.abs(r[i]*1000),best_fit_values[4],best_fit_values[5]*1000)
                v[i] = -vel_tot_iso(np.abs(r[i]),best_fit_values[0],best_fit_values[1],best_fit_values[2],best_fit_values[3],best_fit_values[4],best_fit_values[5])
            elif halo_model == 'NFW':
                v_h[i] = -halo_vel_NFW(np.abs(r[i]*1000),best_fit_values[4],best_fit_values[5]*1000)
                v[i] = -vel_tot_NFW(np.abs(r[i]),best_fit_values[0],best_fit_values[1],best_fit_values[2],best_fit_values[3],best_fit_values[4],best_fit_values[5])
            elif halo_model == 'Burkert':
                v_h[i] = -halo_vel_bur(np.abs(r[i]*1000),best_fit_values[4],best_fit_values[5]*1000)
                v[i] = -vel_tot_bur(np.abs(r[i]),best_fit_values[0],best_fit_values[1],best_fit_values[2],best_fit_values[3],best_fit_values[4],best_fit_values[5])
            else:
                print('Fit function not known.  Please update plot_rot_curve function.')
    ############################################################################


    ############################################################################
    # Plot rotation curve
    #---------------------------------------------------------------------------
    ax.set_title(gal_ID + ' ' +  halo_model)

    ax.plot(rm_deproj, vm_deproj, 'k.', markersize=1)
    ax.plot(r, v, 'c',label='v tot')
    ax.plot(r, v_b, '--',label='bulge')
    ax.plot(r, v_d,'.-',label='disk')
    ax.plot(r, v_h,':',label='halo')

    vmax = 0

    if np.isfinite(np.max(np.abs(v))):
        vmax = np.max(np.abs(v))
        ax.set_ylim([-1.25*vmax,1.25*vmax])
        ax.tick_params(axis='both', direction='in')
        ax.yaxis.set_ticks_position('both')
        ax.xaxis.set_ticks_position('both')
        ax.set_xlabel('Deprojected radius [kpc/h]')
        ax.set_ylabel('Rotational velocity [km/s]')
        plt.legend()
    else:
        vmax = 1000
        ax.set_ylim([-vmax,vmax])
        ax.tick_params(axis='both', direction='in')
        ax.yaxis.set_ticks_position('both')
        ax.xaxis.set_ticks_position('both')
        ax.set_xlabel('Deprojected radius [kpc/h]')
        ax.set_ylabel('Rotational velocity [km/s]')
        plt.legend()
    #plt.savefig(gal_ID + ' rotation curve ' + halo_model + '.png',format='png')
    ############################################################################


"""
################################################################################
def plot_fits(ID, vmasked, ivar_masked, scale, shape, Isothermal_Fit, NFW_Fit, Burket_Fit, mask):
    plt.figure(figsize=(12,12))
    plt.subplot(2,2,1)
    plt.imshow(vmasked,origin='lower',cmap='RdBu_r')
    plt.title(ID + ' Data Map')
    
    cbar = plt.colorbar()
    cbar.set_label('km/s')
    
    plt.subplot(2,2,2)
    plt.imshow(ma.array(rot_incl_iso(shape, scale, Isothermal_Fit), mask=mask), 
               origin='lower', 
               cmap='RdBu_r')

    plt.xlabel('spaxel')
    plt.ylabel('spaxel')
    plt.title(ID + ' Isothermal Fit')

    plt.xlabel('spaxel')
    plt.ylabel('spaxel')

    cbar = plt.colorbar()
    cbar.set_label('km/s')
    
    
    plt.subplot(2,2,3)
    plt.imshow(ma.array(rot_incl_NFW(shape, scale, NFW_Fit), mask=mask),
               origin='lower',
               cmap='RdBu_r')

    plt.xlabel('spaxel')
    plt.ylabel('spaxel')
    plt.title(ID + ' NFW Fit')

    cbar = plt.colorbar()
    cbar.set_label('km/s')
    
    plt.subplot(2,2,4)
    plt.imshow(ma.array(rot_incl_bur(shape, scale, Burket_Fit), mask=mask), 
               origin='lower', 
               cmap='RdBu_r')

    plt.xlabel('spaxel')
    plt.ylabel('spaxel')
    plt.title(ID + ' Burket Fit')
    
    cbar = plt.colorbar()
    cbar.set_label('km/s')
    
    
    plt.suptitle('Fit Analysis ' + ID)
    plt.savefig('Fit Analysis ' + ID + '.png', format='png')
    plt.close()

################################################################################
"""

def plot_diagnostic_panel( ID, shape, scale, Isothermal_Fit, NFW_Fit, Burket_Fit, mask, vmasked, ivar_masked):
    '''
    Plot a two by two paneled image containging the entire r-band array, the 
    masked H-alpha array, the masked H-alpha array containing ovals of the 
    spaxels processed in the algorithm, and the averaged max and min rotation 
    curves along with the stellar mass rotation curve.


    Parameters
    ==========

    gal_ID : string
        MaNGA plate number - MaNGA fiberID number

    r_band : numpy array of shape (n,n)
        r_band flux map

    masked_Ha_vel : numpy array of shape (n,n)
        Masked H-alpha velocity map

    masked_vel_contour_plot : numpy array of shape (n,n)
        Masked H-alpha velocity map showing only those spaxels within annuli

    data_table : Astropy QTable
        Table containing measured rotational velocities at given deprojected 
        radii

    IMAGE_DIR : string
        Path of directory to store images.  Default is None (does not save 
        figure)

    IMAGE_FORMAT : string
        Format of saved image.  Default is 'eps'
    '''


#    panel_fig, (( Ha_vel_panel, mHa_vel_panel),
#                ( contour_panel, rot_curve_panel)) = plt.subplots( 2, 2)
    panel_fig, ((Isothermal_Plot_panel, NFW_Plot_panel, Burket_Plot_panel),
               (RC_Isothermal, RC_NFW, RC_Burket)) = plt.subplots( 2, 3)
    panel_fig.set_figheight( 10)
    panel_fig.set_figwidth( 15)
    plt.suptitle(ID + " Diagnostic Panel", y=1.05, fontsize=16)
    
    Plotting_Isothermal(ID, shape, scale, Isothermal_Fit, mask, ax = Isothermal_Plot_panel)
    
    Plotting_NFW(ID, shape, scale, NFW_Fit, mask, ax = NFW_Plot_panel)
    
    Plotting_Burkert(ID, shape, scale, Burket_Fit, mask, ax = Burket_Plot_panel)
    
    plot_rot_curve(vmasked,ivar_masked,Isothermal_Fit,scale,ID,'Isothermal', ax = RC_Isothermal)
    
    plot_rot_curve(vmasked,ivar_masked,NFW_Fit,scale,ID,'NFW', ax = RC_NFW)
    
    plot_rot_curve(vmasked,ivar_masked,Burket_Fit,scale,ID,'Burkert', ax = RC_Burket)
    
    panel_fig.tight_layout()
    
    plt.savefig(ID + '_Diagnostic_Panels_new')

# MCMC running 
def log_prior(params):

    log_rhob0,Rb,SigD,Rd,log_rhoh0,Rh,inclination,phi,center_x,center_y,vsys = params

    logP = 0

    rhob_check = -7 < log_rhob0 < 1
    #rhob_check = 0 < log_rhob0 < 10
    Rb_check = 0 < Rb < 5

    SigD_check = 0.1 < SigD < 3000
    Rd_check = 0.1 < Rd < 30

    rhoh_check = -7 < log_rhoh0 < 2
    #rhoh_check = 0 < log_rhoh0 < 100
    Rh_check = 0.01 < Rh < 500

    i_check = 0 < inclination < np.pi*0.436
    phi_check = 0 < phi < 2*np.pi

    x_check = 10 < center_x < 50
    y_check = 10 < center_y < 50

    v_check = -100 < vsys < 100

    if rhob_check and Rb_check and SigD_check and Rd_check and rhoh_check and Rh_check and i_check and phi_check and x_check and y_check and v_check:
        logP = 0

    # setting constraints on the radii
    elif (Rh < Rb) or (Rh < Rd) or (Rd < Rb):
        logP = -np.inf

    else:
        logP = -np.inf

    return logP

def log_prob_iso(params, scale, shape, vdata, ivar, mask):

    lp = log_prior(params)

    logL = loglikelihood_iso_flat_constraints(params, 
                                              scale, 
                                              shape, 
                                              vdata.compressed(), 
                                              ivar.compressed(), 
                                              mask)

    if not np.isfinite(lp) or not np.isfinite(logL):
        return -np.inf 
    else:
        return lp + logL

def log_prob_NFW(params, scale, shape, vdata, ivar, mask):

    lp = log_prior(params)

    logL = loglikelihood_NFW_flat_constraints(params, 
                                              scale, 
                                              shape, 
                                              vdata.compressed(), 
                                              ivar.compressed(), 
                                              mask)

    if not np.isfinite(lp) or not np.isfinite(logL):
        return -np.inf 
    else:
        return lp + logL

def log_prob_bur(params, scale, shape, vdata, ivar, mask):

    lp = log_prior(params)

    logL = loglikelihood_bur_flat_constraints(params, 
                                              scale, 
                                              shape, 
                                              vdata.compressed(), 
                                              ivar.compressed(), 
                                              mask)

    if not np.isfinite(lp) or not np.isfinite(logL):
        return -np.inf 
    else:
        return lp + logL

def run_MCMC(gal_ID,VEL_MAP_FOLDER,init_param_geo,scale,model):

    data_maps, gshape = Galaxy_Data(gal_ID,VEL_MAP_FOLDER)

    #data_map = data_maps['vmasked']

    SN_map = data_maps['Ha_flux'] * np.sqrt(data_maps['Ha_flux_ivar'])
    Ha_vel_mask = data_maps['Ha_vel_mask'] + (SN_map < 5)

    vmasked = ma.array(data_maps['Ha_vel'], mask = Ha_vel_mask)
    ivar_masked = ma.array(data_maps['Ha_vel_ivar'], mask = Ha_vel_mask)

    model_guesses = [-1, 1, 1000, 4, -3, 25]

    geo_guesses = init_param_geo + [0]

    pos_model = np.random.uniform(low=[-7,0.00001,0.00001,0.0001,-7,0.1], 
                        high=[1,10,3000,30,2,1000], 
                        size=(64,6))

    pos_geo = np.array(geo_guesses) + np.random.uniform(np.random.uniform(low=-1e-3*np.ones(len(geo_guesses)), 
                                              high=1e-3*np.ones(len(geo_guesses)), 
                                              size=(64,len(geo_guesses))))

    pos_combined = np.column_stack((pos_model,pos_geo))

    print('initial guesses generated: ' ,flush=True)
    print(pos_combined ,flush=True)

    nwalkers, ndim = pos_combined.shape
    #-------------------------------------------------------------------------------
    if model == 'iso':

        bad_sampler_iso = emcee.EnsembleSampler(nwalkers, 
                                                ndim, 
                                                log_prob_iso, 
                                                args=(scale, 
                                                      gshape, 
                                                      vmasked, 
                                                      ivar_masked, 
                                                      Ha_vel_mask))

        print('bad sampler iso',flush=True)

        bad_sampler_iso.run_mcmc(pos_combined, 10000, progress=True)
        bad_samples_iso = bad_sampler_iso.get_chain()
        #bad_samples_iso = bad_sampler_iso.get_chain(discard=500)

        print('bad samples_iso',flush=True)

        ns_iso, nw_iso, nd_iso = bad_samples_iso.shape

        #np.save('bad_samples_iso_' + gal_ID + '_comb.npy', bad_samples_iso)

        good_walkers_iso = bad_sampler_iso.acceptance_fraction > 0
        #np.save('good_walkers_iso_' + gal_ID + '_comb.npy', good_walkers_iso)

        good_samples_iso = bad_samples_iso[:,good_walkers_iso,:]

        print('ok',flush=True)

        # Check if there are walkers being left out
        # Make the general cut at 4000 steps
        if good_samples_iso.shape != bad_samples_iso.shape:
            print("Inconsistent shapes")
            ns_iso_good, nw_iso_good, nd_iso_good = good_samples_iso[4000:,:,:].shape
            trimmed_flat_good_samples_iso = good_samples_iso[4000:,:,:].reshape(6000*nw_iso_good, nd_iso_good)
            samples_mean_iso = np.mean(trimmed_flat_good_samples_iso, axis=0)
            print(samples_mean_iso)
        else:
            trimmed_flat_good_samples_iso = good_samples_iso[4000:,:,:].reshape(6000*nw_iso, nd_iso)
            samples_mean_iso = np.mean(trimmed_flat_good_samples_iso, axis=0)
            print(samples_mean_iso)

        # In case the trimming is too much
        if not (np.isfinite(samples_mean_iso[-1])):
            if good_samples_iso.shape != bad_samples_iso.shape:
                ns_iso_good, nw_iso_good, nd_iso_good = good_samples_iso.shape
                flat_good_samples_iso = good_samples_iso.reshape(ns_iso_good*nw_iso_good, nd_iso_good)
                samples_mean_iso = np.mean(flat_good_samples_iso, axis=0)
                print(samples_mean_iso)
            else:
                flat_good_samples_iso = good_samples_iso.reshape(ns_iso*nw_iso, nd_iso)
                samples_mean_iso = np.mean(flat_good_samples_iso, axis=0)
                print(samples_mean_iso)

        print('MCMC fits generated',flush=True)

        fitted_map_mcmc_iso = rot_incl_iso(gshape, scale, np.ndarray.tolist(samples_mean_iso))

        print('fitted map generated',flush=True)

        mfitted_map_mcmc_iso = ma.array(fitted_map_mcmc_iso, mask=Ha_vel_mask)

        print('masked map generated',flush=True)

        nd_iso_mcmc = np.sum(~mfitted_map_mcmc_iso.mask)

        print('number of data calculated',flush=True)

        # chi2_iso = np.nansum((vmasked - vmap_iso) ** 2 * Ha_vel_ivar)
        chi2_iso_mcmc = ma.sum(ivar_masked * (vmasked - mfitted_map_mcmc_iso) ** 2)

        print('chi2 calculated',flush=True)

        # chi2_iso_norm = chi2_iso/(nd_iso - 8)
        chi2_iso_norm_mcmc = chi2_iso_mcmc / (nd_iso_mcmc - len(samples_mean_iso))

        print('normalized chi2 calculated',flush=True)

        return samples_mean_iso, chi2_iso_norm_mcmc
    #-------------------------------------------------------------------------------
    elif model == 'nfw':
        bad_sampler_NFW = emcee.EnsembleSampler(nwalkers, 
                                                ndim, 
                                                log_prob_NFW, 
                                                args=(scale, 
                                                      gshape, 
                                                      vmasked, 
                                                      ivar_masked, 
                                                      Ha_vel_mask))

        print('bad sampler NFW',flush=True)

        bad_sampler_NFW.run_mcmc(pos_combined, 10000, progress=True)
        bad_samples_NFW = bad_sampler_NFW.get_chain()
        #bad_samples_NFW = bad_sampler_NFW.get_chain(discard=500)

        print('bad samples_NFW',flush=True)

        ns_NFW, nw_NFW, nd_NFW = bad_samples_NFW.shape

        #np.save('bad_samples_NFW_' + gal_ID + '_comb.npy', bad_samples_NFW)

        good_walkers_NFW = bad_sampler_NFW.acceptance_fraction > 0
        #np.save('good_walkers_NFW_' + gal_ID + '_comb.npy', good_walkers_NFW)

        good_samples_NFW = bad_samples_NFW[:,good_walkers_NFW,:]

        print('ok',flush=True)

        # Check if there are walkers being left out
        # Make the general cut at 4000 steps
        if good_samples_NFW.shape != bad_samples_NFW.shape:
            print("Inconsistent shapes")
            ns_NFW_good, nw_NFW_good, nd_NFW_good = good_samples_NFW[4000:,:,:].shape
            trimmed_flat_good_samples_NFW = good_samples_NFW[4000:,:,:].reshape(6000*nw_NFW_good, nd_NFW_good)
            samples_mean_NFW = np.mean(trimmed_flat_good_samples_NFW, axis=0)
            print(samples_mean_NFW)
        else:
            trimmed_flat_good_samples_NFW = good_samples_NFW[4000:,:,:].reshape(6000*nw_NFW, nd_NFW)
            samples_mean_NFW = np.mean(trimmed_flat_good_samples_NFW, axis=0)
            print(samples_mean_NFW)

        # In case the trimming is too much
        if not (np.isfinite(samples_mean_NFW[-1])):
            if good_samples_NFW.shape != bad_samples_NFW.shape:
                ns_NFW_good, nw_NFW_good, nd_NFW_good = good_samples_NFW.shape
                flat_good_samples_NFW = good_samples_NFW.reshape(ns_NFW_good*nw_NFW_good, nd_NFW_good)
                samples_mean_NFW = np.mean(flat_good_samples_NFW, axis=0)
                print(samples_mean_NFW)
            else:
                flat_good_samples_NFW = good_samples_NFW.reshape(ns_NFW*nw_NFW, nd_NFW)
                samples_mean_NFW = np.mean(flat_good_samples_NFW, axis=0)
                print(samples_mean_NFW)


        print('MCMC fits generated',flush=True)

        fitted_map_mcmc_NFW = rot_incl_NFW(gshape, scale, np.ndarray.tolist(samples_mean_NFW))

        print('fitted map generated',flush=True)

        mfitted_map_mcmc_NFW = ma.array(fitted_map_mcmc_NFW, mask=Ha_vel_mask)

        print('masked map generated',flush=True)

        nd_NFW_mcmc = np.sum(~mfitted_map_mcmc_NFW.mask)

        print('number of data calculated',flush=True)

        chi2_NFW_mcmc = ma.sum(ivar_masked * (vmasked - mfitted_map_mcmc_NFW)**2)

        print('chi2 calculated',flush=True)

        chi2_NFW_norm_mcmc = chi2_NFW_mcmc / (nd_NFW_mcmc - len(samples_mean_NFW))

        print('normalized chi2 calculated',flush=True)

        return samples_mean_NFW, chi2_NFW_norm_mcmc
    #-------------------------------------------------------------------------------
    elif model == 'bur':
        bad_sampler_bur = emcee.EnsembleSampler(nwalkers, 
                                                ndim, 
                                                log_prob_bur, 
                                                args=(scale, 
                                                      gshape, 
                                                      vmasked, 
                                                      ivar_masked, 
                                                      Ha_vel_mask))

        print('bad sampler bur',flush=True)

        bad_sampler_bur.run_mcmc(pos_combined, 10000, progress=True)
        bad_samples_bur = bad_sampler_bur.get_chain()
        #bad_samples_bur = bad_sampler_bur.get_chain(discard=500)

        print('bad samples_bur',flush=True)

        ns_bur, nw_bur, nd_bur = bad_samples_bur.shape

        #np.save('bad_samples_bur_' + gal_ID + '_comb.npy', bad_samples_bur)

        good_walkers_bur = bad_sampler_bur.acceptance_fraction > 0
        #np.save('good_walkers_bur_' + gal_ID + '_comb.npy', good_walkers_bur)

        good_samples_bur = bad_samples_bur[:,good_walkers_bur,:]

        print('ok',flush=True)

        # Check if there are walkers being left out
        # Make the general cut at 4000 steps
        if good_samples_bur.shape != bad_samples_bur.shape:
            print("Inconsistent shapes")
            ns_bur_good, nw_bur_good, nd_bur_good = good_samples_bur[4000:,:,:].shape
            trimmed_flat_good_samples_bur = good_samples_bur[4000:,:,:].reshape(6000*nw_bur_good, nd_bur_good)
            samples_mean_bur = np.mean(trimmed_flat_good_samples_bur, axis=0)
            print(samples_mean_bur)
        else:
            trimmed_flat_good_samples_bur = good_samples_bur[4000:,:,:].reshape(6000*nw_bur, nd_bur)
            samples_mean_bur = np.mean(trimmed_flat_good_samples_bur, axis=0)
            print(samples_mean_bur)

        # In case the trimming is too much
        if not (np.isfinite(samples_mean_bur[-1])):
            if good_samples_bur.shape != bad_samples_bur.shape:
                ns_bur_good, nw_bur_good, nd_bur_good = good_samples_bur.shape
                flat_good_samples_bur = good_samples_bur.reshape(ns_bur_good*nw_bur_good, nd_bur_good)
                samples_mean_bur = np.mean(flat_good_samples_bur, axis=0)
                print(samples_mean_bur)
            else:
                flat_good_samples_bur = good_samples_bur.reshape(ns_bur*nw_bur, nd_bur)
                samples_mean_bur = np.mean(flat_good_samples_bur, axis=0)
                print(samples_mean_bur)

        print('MCMC fits generated',flush=True)

        fitted_map_mcmc_bur = rot_incl_bur(gshape, scale, np.ndarray.tolist(samples_mean_bur))

        print('fitted map generated',flush=True)

        mfitted_map_mcmc_bur = ma.array(fitted_map_mcmc_bur, mask=Ha_vel_mask)

        print('masked map generated',flush=True)

        nd_bur_mcmc = np.sum(~mfitted_map_mcmc_bur.mask)

        print('number of data calculated',flush=True)

        chi2_bur_mcmc = ma.sum(ivar_masked * (vmasked - mfitted_map_mcmc_bur)**2)

        print('chi2 calculated',flush=True)

        chi2_bur_norm_mcmc = chi2_bur_mcmc / (nd_bur_mcmc - len(samples_mean_bur))

        print('normalized chi2 calculated',flush=True)

        return samples_mean_bur, chi2_bur_norm_mcmc
    ################################################################################


# Functions for calculating the Hessain Matrix numerically
# Isothermal Model
def Hessian_Calculation_Isothermal(fit_solution, scale, shape, vmap, ivar):
    '''

    :param fit_solution:
    :param scale:
    :param shape:
    :param vmap:
    :param ivar:
    :return:
    '''

    print('Best-fit values in Hessian_Calculation_Isothermal:', fit_solution)

    mask = vmap.mask

    vmap_flat = vmap.compressed()

    ivar_masked = ma.array(ivar,mask=mask)
    ivar_flat = ivar_masked.compressed()

    #print('Calculating Hessian')
    hessian_iso = ndt.Hessian(loglikelihood_iso_flat, step=0.01*fit_solution)#, method='forward', order=1)

    #print('Evaluating Hessian at solution')
    hess_ll_iso = hessian_iso(fit_solution, 
                              scale, 
                              shape, 
                              vmap_flat, 
                              ivar_flat,mask)
    hess_inv_iso = np.linalg.inv(hess_ll_iso)
    fit_err_iso = np.sqrt(np.diag(-hess_inv_iso))

    print('-------------------------------------------')
    print('Hessian matrix for Isothermal')
    print(fit_err_iso)

    return fit_err_iso

# NFW Model
def Hessian_Calculation_NFW(fit_solution, scale, shape, vmap, ivar):
    '''

    :param fit_solution:
    :param scale:
    :param shape:
    :param vmap:
    :param ivar:
    :return:
    '''

    print('Best-fit values in Hessian_Calculation_NFW:', fit_solution)

    mask = vmap.mask
    vmap_flat = vmap.compressed()
    ivar_masked = ma.array(ivar, mask=mask)
    ivar_flat = ivar_masked.compressed()
    hessian_NFW = ndt.Hessian(loglikelihood_NFW_flat,step=0.01*fit_solution)
    hess_ll_NFW = hessian_NFW(fit_solution, scale, shape, vmap_flat, ivar_flat, mask)
    hess_inv_NFW = np.linalg.inv(hess_ll_NFW)
    fit_err_NFW = np.sqrt(np.diag(-hess_inv_NFW))
    print('-------------------------------------------')
    print('Hessian matrix for NFW')
    print(fit_err_NFW)

    return fit_err_NFW

# Burket Model
def Hessian_Calculation_Burket(fit_solution, scale, shape, vmap, ivar):
    '''

    :param fit_solution:
    :param scale:
    :param shape:
    :param vmap:
    :param ivar:
    :return:
    '''

    print('Best-fit values in Hessian_Calculation_Burket:', fit_solution)

    mask = vmap.mask
    vmap_flat = vmap.compressed()
    ivar_masked = ma.array(ivar, mask=mask)
    ivar_flat = ivar_masked.compressed()
    hessian_bur = ndt.Hessian(loglikelihood_bur_flat,step=0.01*fit_solution)
    hess_ll_bur = hessian_bur(fit_solution, scale, shape, vmap_flat, ivar_flat, mask)
    hess_inv_bur = np.linalg.inv(hess_ll_bur)
    fit_err_bur = np.sqrt(np.diag(-hess_inv_bur))
    print('-------------------------------------------')
    print('Hessian matrix for Burket')
    print(fit_err_bur)

    return fit_err_bur
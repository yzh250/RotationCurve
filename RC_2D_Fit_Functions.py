################################################################################
# Import modules
#-------------------------------------------------------------------------------
import matplotlib.pyplot as plt

import numpy as np
import numpy.ma as ma

from astropy.io import fits
from astropy.table import QTable

from scipy.optimize import minimize

import numdifftools as ndt


# Import functions from other .py files
from Velocity_Map_Functions import rot_incl_iso,\
                                   rot_incl_NFW, \
                                   rot_incl_bur, \
                                   nloglikelihood_iso,\
                                   nloglikelihood_NFW, \
                                   nloglikelihood_bur,\
                                   loglikelihood_iso_flat,\
                                   loglikelihood_NFW_flat, \
                                   loglikelihood_bur_flat,\
                                   nloglikelihood_iso_flat,\
                                   nloglikelihood_NFW_flat,\
                                   nloglikelihood_bur_flat
################################################################################




################################################################################
# Physics Constants
#-------------------------------------------------------------------------------
c = 3E5 # km * s ^1
h = 1 # reduced hubble constant
H_0 =  100 * h # km * s^-1 * Mpc^-1
################################################################################


MANGA_FOLDER = '/Users/richardzhang/Documents/UR_Stuff/Research_UR/SDSS/dr16/manga/spectro/'
VEL_MAP_FOLDER = MANGA_FOLDER + 'analysis/v2_4_3/2.2.1/HYB10-GAU-MILESHC/'


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




def Galaxy_Data(galaxy_ID):
    '''
    PARAMETERS
    ==========

    galaxy_ID : string
        'Plate-IFU'


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
    #cube = fits.open('manga-' + galaxy_ID + '-MAPS-HYB10-GAU-MILESHC.fits.gz')
    cube = fits.open(VEL_MAP_FOLDER + plate + '/' + IFU + '/manga-' + galaxy_ID + '-MAPS-HYB10-GAU-MILESHC.fits.gz')

    r_band = cube['SPX_MFLUX'].data
    Ha_vel = cube['EMLINE_GVEL'].data[18]
    Ha_vel_ivar = cube['EMLINE_GVEL_IVAR'].data[18]
    Ha_vel_mask = cube['EMLINE_GVEL_MASK'].data[18]

    vmasked = ma.array(Ha_vel, mask=Ha_vel_mask)
    ivar_masked = ma.array(Ha_vel_ivar, mask=Ha_vel_mask)

    gshape = vmasked.shape
    ############################################################################


    ############################################################################
    # Finding the center
    #---------------------------------------------------------------------------
    center_guess = np.unravel_index(ma.argmin(np.abs(vmasked), axis=None), 
                                    vmasked.shape)
    x_center_guess = center_guess[0]
    y_center_guess = center_guess[1]
    ############################################################################

    #return scale, incl, ph, r_band, Ha_vel, Ha_vel_ivar, Ha_vel_mask, vmasked, gshape, x_center_guess, y_center_guess
    return r_band, Ha_vel, Ha_vel_ivar, Ha_vel_mask, vmasked, ivar_masked, gshape, x_center_guess, y_center_guess



def Galaxy_Fitting_iso(params, scale, shape, vmap, ivar, mask):
    '''

    :param params:
    :param scale:
    :param shape:
    :param vmap:
    :param ivar:
    :return:
    '''

    incl, ph, x_guess, y_guess = params

    vmap_flat = vmap.compressed()
    ivar_masked = ma.array(ivar,mask=mask)
    ivar_flat = ivar_masked.compressed()

    # Isothermal Fitting
    bounds_iso = [[1e-9, 1],  # Scale Factor [unitless]
                  [0.001, 1000],  # Bulge Scale Velocity [km/s]
                  [0, 10000],  # Surface Density [Msol/pc^2]
                  [0.1, 20],  # Disk radius [kpc]
                  [50, 5000],  # Velocity at infinity [km/s]
                  [0.1, 100],  # Halo radius [kpc]
                  [0.1, 0.5*np.pi],  # Inclination angle
                  [0, 2 * np.pi],  # Phase angle
                  [x_guess-10, x_guess+10],  # center_x
                  [y_guess-10, y_guess+10]]  # center_y

    ig_iso = [0.5, 127, 1000, 4, 150, 25, incl, ph, x_guess, y_guess]

    print(ig_iso)

    bestfit_iso = minimize(nloglikelihood_iso_flat,
                           ig_iso, 
                           args=(scale, shape, vmap_flat, ivar_flat, mask),
                           method='Powell', 
                           bounds=bounds_iso)
    print('---------------------------------------------------')
    print(bestfit_iso)

    return bestfit_iso.x


def Galaxy_Fitting_NFW(params, scale, shape, vmap, ivar,mask):
    '''

    :param params:
    :param scale:
    :param shape:
    :param vmap:
    :param ivar:
    :return:
    '''

    incl, ph, x_guess, y_guess = params

    vmap_flat = vmap.compressed()
    ivar_masked = ma.array(ivar, mask=mask)
    ivar_flat = ivar_masked.compressed()

    # NFW Fitting
    bounds_NFW = [[1e-9, 1],  # Scale Factor [unitless]
                  [0.001, 1000],  # Bulge Scale Velocity [km/s]
                  [0, 10000],  # Surface Density [Msol/pc^2]
                  [0.1, 20],  # Disk radius [kpc]
                  [0.0001, 0.1],  # Halo density [Msun/pc^2]
                  [0.1, 100],  # Halo radius [kpc]
                  [0.1, 0.5*np.pi],  # Inclination angle
                  [0, 2 * np.pi],  # Phase angle
                  [27, 47],  # center_x
                  [27, 47]]  # center_y

    ig_NFW = [0.4, 127, 1000, 4, 0.006, 25, incl, ph, x_guess, y_guess]

    bestfit_NFW = minimize(nloglikelihood_NFW_flat,
                           ig_NFW, 
                           args=(scale, shape, vmap_flat, ivar_flat, mask),
                           method='Powell', 
                           bounds=bounds_NFW)
    print('---------------------------------------------------')
    print(bestfit_NFW)

    return bestfit_NFW.x, 



def Galaxy_Fitting_bur(params, scale, shape, vmap, ivar,mask):
    '''

    :param params:
    :param scale:
    :param shape:
    :param vmap:
    :param ivar:
    :return:
    '''

    incl, ph, x_guess, y_guess = params

    vmap_flat = vmap.compressed()
    ivar_masked = ma.array(ivar, mask=mask)
    ivar_flat = ivar_masked.compressed()

    # Burket Fitting
    bounds_bur = [[1e-9, 1],  # Scale Factor [unitless]
                  [0.001, 1000],  # Bulge Scale Velocity [km/s]
                  [0, 10000],  # Surface Density [Msol/pc^2]
                  [0.1, 20],  # Disk radius [kpc]
                  [0.0001, 0.1],  # Halo central density[km/s]
                  [0.1, 100],  # Halo radius [kpc]
                  [0.1, 0.5*np.pi],  # Inclination angle
                  [0, 2 * np.pi],  # Phase angle
                  [27, 47],  # center_x
                  [27, 47]]  # center_y

    ig_bur = [0.4, 127, 1000, 4, 0.006, 25, incl, ph, x_guess, y_guess]

    bestfit_bur = minimize(nloglikelihood_bur_flat,
                           ig_bur, 
                           args=(scale, shape, vmap_flat, ivar_flat, mask),
                           method='Powell', 
                           bounds=bounds_bur)
    print('---------------------------------------------------')
    print(bestfit_bur)

    return bestfit_bur.x






# Showing and Saving Images
# Isothermal
def Plotting_Isothermal(ID, shape, scale, fit_solution, mask):
    '''

    :param ID:
    :param shape:
    :param scale:
    :param fit_solution:
    :param mask:
    :return:
    '''
    plt.figure()

    plt.imshow(ma.array(rot_incl_iso(shape, scale, fit_solution), mask=mask), 
               origin='lower', 
               cmap='RdBu_r')

    plt.title(ID + ' Isothermal Fit')

    plt.xlabel('spaxel')
    plt.ylabel('spaxel')

    cbar = plt.colorbar()
    cbar.set_label('km/s')

    plt.show()
    #plt.savefig(ID + ' Isothermal.png', format='png')
    #plt.close()
    




# NFW
def Plotting_NFW(ID, shape, scale, fit_solution, mask):
    '''

    :param ID:
    :param shape:
    :param scale:
    :param fit_solution:
    :param mask:
    :return:
    '''

    plt.figure()

    plt.imshow(ma.array(rot_incl_NFW(shape, scale, fit_solution), mask=mask), 
               origin='lower', 
               cmap='RdBu_r')

    plt.title(ID + ' NFW Fit')

    plt.xlabel('spaxel')
    plt.ylabel('spaxel')

    cbar = plt.colorbar()
    cbar.set_label('km/s')

    plt.show()
    #plt.savefig(ID + ' NFW.png', format='png')
    #plt.close()




# Burket
def Plotting_Burket(ID, shape, scale, fit_solution, mask):
    '''

    :param ID:
    :param shape:
    :param scale:
    :param fit_solution:
    :param mask:
    :return:
    '''

    plt.figure()

    plt.imshow(ma.array(rot_incl_bur(shape, scale, fit_solution), mask=mask), 
               origin='lower', 
               cmap='RdBu_r')

    plt.title(ID + ' Burket Fit')

    plt.xlabel('spaxel')
    plt.ylabel('spaxel')

    cbar = plt.colorbar()
    cbar.set_label('km/s')

    plt.show()
    #plt.savefig(ID + ' Burket.png', format='png')
    #plt.close()



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
    hessian_iso = ndt.Hessian(loglikelihood_iso_flat)#, method='forward', order=1)
    #print('Evaluating Hessian at solution')
    hess_ll_iso = hessian_iso(fit_solution,scale,shape,vmap_flat,ivar_flat,mask)
    hess_inv_iso = np.linalg.inv(hess_ll_iso)
    fit_err_iso = np.sqrt(np.diag(np.abs(hess_inv_iso)))
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
    mask = vmap.mask
    vmap_flat = vmap.compressed
    ivar_masked = ma.array(ivar, mask=mask)
    ivar_flat = ivar_masked.compressed()
    #print('Inverse variance flat array')
    #print(ivar_flat)
    hessian_NFW = ndt.Hessian(loglikelihood_NFW_flat)
    hess_ll_NFW = hessian_NFW(fit_solution, scale, shape, vmap_flat, ivar_flat, mask)
    hess_inv_NFW = np.linalg.inv(hess_ll_NFW)
    fit_err_NFW = np.sqrt(np.diag(np.abs(hess_inv_NFW)))
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
    mask = vmap.mask
    vmap_flat = vmap.compressed()
    ivar_masked = ma.array(ivar, mask=mask)
    ivar_flat = ivar_masked.compressed()
    hessian_bur = ndt.Hessian(loglikelihood_bur_flat)
    hess_ll_bur = hessian_bur(fit_solution, scale, shape, vmap_flat, ivar_flat, mask)
    hess_inv_bur = np.linalg.inv(hess_ll_bur)
    fit_err_bur = np.sqrt(np.diag(np.abs(hess_inv_bur)))
    print('-------------------------------------------')
    print('Hessian matrix for Burket')
    print(fit_err_bur)

    return fit_err_bur
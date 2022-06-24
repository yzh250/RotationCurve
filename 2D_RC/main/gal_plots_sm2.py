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
# Obtaining information for MaNGA galaxies with smoothness smaller than 2
#-------------------------------------------------------------------------------
galaxy_ID = ['10001-6104','7957-12703','7962-3704','7964-1902','7991-1902','8078-12702','8082-3704','8083-3704','8083-6103','8086-6103',\
 '8133-12704','8133-6103','8137-12701','8140-12702','8143-1902','8144-6103','8145-12703','8146-12703','8146-3701','8149-12705',\
 '8155-3704','8241-9102','8249-1902','8252-12701','8253-12702','8253-12705','8254-12702','8254-12705','8255-12703','8255-6101',\
 '8255-6104','8258-6104','8262-12702','8263-6103','8274-12705','8311-9101','8315-9101','8317-12703','8318-6103','8323-3704',\
 '8325-1901','8332-3704','8333-12701','8439-1901','8439-3702','8440-3704','8444-9101','8449-6104','8455-12702','8458-6102',\
 '8461-9102','8481-6102','8485-3701','8486-6104','8547-12703','8552-9102','8566-9101','8567-3704','8567-6103','8591-3702',\
 '8597-1902','8612-6102','8612-9102','8618-6104','8623-12701','8623-12705','8623-3702','8624-3702','8624-6102','8626-12701',\
 '8626-6103','8655-6101','8713-1901','8718-3704','8726-1901','8726-3701','8726-6103','8728-3702','8728-9101','8931-9101',\
 '8933-12701','8935-3703','8936-6102','8936-9102','8939-12705','8940-3702','8941-3704','8944-1902','8948-3704','8950-9101',\
 '8978-3704','8980-1902','8987-9102','8992-12701','8993-6104','8996-3701','8997-12703','8997-12705','8997-3702','9002-3702',\
 '9024-12703','9027-6101','9028-3702','9029-6104','9031-12702','9031-3703','9034-1902','9034-6103','9035-12702','9035-1901',\
 '9037-9101','9039-6104','9044-12704','9044-6103','9047-9101','9048-9101','9183-12703','9183-3702','9185-6103','9193-12703',\
 '9194-6103','9194-9101','9490-1901','9491-12702','9492-12702','9492-3702','9493-3701','9494-6102','9501-3704','9505-12702',\
 '9505-12703','9508-3702','9512-6102','9864-9102','9865-12704','9865-3704','9870-1902','9870-3701','9870-6101','9876-3701','9891-3704']
#-------------------------------------------------------------------------------

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

        plt.imshow(vmasked,cmap='RdBu_r',origin='lower')
        plt.title(galaxy_ID[i]+'_sm2')
        plt.savefig(galaxy_ID[i]+'_sm2',format='png')


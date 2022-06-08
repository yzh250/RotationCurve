import numpy as np
import numpy.ma as ma
from astropy.table import Table, QTable
import csv

MORPH_FOLDER = '/home/yzh250/Documents/UR_Stuff/Research_UR/SDSS/dr17/manga/morph/'

def galaxies_dict(galaxy_ID_list):
    '''
    Built dictionary of (plate, IFU) tuples that refer to the galaxy's row 
    index in the ref_table.
    PARAMETERS
    ==========
    ref_table : astropy table
        Data table with columns
          - MaNGA_plate (int) : MaNGA plate number
          - MaNGA_IFU (int)   : MaNGA IFU number
    RETURNS
    =======
    ref_dict : dictionary
        Dictionary with keys (plate, IFU) and values are the row index in 
        ref_table
    '''


    # Initialize dictionary to store (plate, IFU) and row index
    ref_dict = {}


    for i in range(len(galaxy_ID_list)):

        galaxy_ID = galaxy_ID_list[i]

        ref_dict[galaxy_ID] = i

    return ref_dict

def match_morph_dl(galaxy_ID_list,smoothness_Table):
    '''
    Locate the galaxy morphology from the MaNGA Deep Learning morphology catalog 
    and add it to the given data table.
    PARAMETERS
    ==========
    data : astropy table (may or may not have quantities)
        Table of galaxies
    RETURNS
    =======
    data : astropy table (may or may not have quantities)
        Table of galaxies, with the added morphology data:
          - GZ_edge_on : Likelihood that the galaxy is edge-on
          - GZ_bar : Likelihood of a bar
          - GZ_spiral : Likelihood that the galaxy is a spiral
    '''

    ############################################################################
    # Initialize morphology columns in the data table
    #---------------------------------------------------------------------------
    smoothness_Table['DL_ttype'] = np.nan
    smoothness_Table['DL_PLTG'] = np.nan
    smoothness_Table['DL_Visual_Class'] = np.nan
    #data['DL_s0'] = np.nan
    #data['DL_edge_on'] = np.nan
    #data['DL_bar_GZ2'] = np.nan
    #data['DL_bar_N10'] = np.nan
    #data['DL_merge'] = np.nan
    ############################################################################


    ############################################################################
    # Load in morphology data
    #---------------------------------------------------------------------------
    data_directory = '/Users/richardzhang/Documents/UR_Stuff/Research_UR/SDSS/dr16/manga/morph/'

    morph_filename = data_directory + 'manga-morphology-dl-DR17.fits'

    morph_data = Table.read(morph_filename, format='fits')

    #print(morph_data.colnames)
    ############################################################################


    ############################################################################
    # Build galaxy reference dictionary
    #---------------------------------------------------------------------------
    data_dict = galaxies_dict(galaxy_ID_list)
    ############################################################################

    ############################################################################
    # Insert morphology data into table
    #---------------------------------------------------------------------------
    for i in range(len(morph_data)):

        ########################################################################
        # Deconstruct galaxy ID
        #-----------------------------------------------------------------------
        gal_ID = morph_data['PLATEIFU'][i].strip()
        ########################################################################

        if gal_ID in data_dict:
            ####################################################################
            # Find galaxy's row number in the data table
            #-------------------------------------------------------------------
            gal_i = data_dict[gal_ID]
            ####################################################################


            ####################################################################
            # Insert morphology data into data table
            #-------------------------------------------------------------------
            smoothness_Table['DL_ttype'][gal_i] = morph_data['T-Type'][i]
            smoothness_Table['DL_PLTG'][gal_i] = morph_data['P_LTG'][i]
            smoothness_Table['DL_Visual_Class'][gal_i] = morph_data['Visual_Class'][i]
            #data['DL_ttype'][gal_i] = morph_data['TTYPE'][i]
            #data['DL_s0'][gal_i] = morph_data['P_S0'][i]
            #data['DL_edge_on'][gal_i] = morph_data['P_EDGE_ON'][i]
            #data['DL_bar_GZ2'][gal_i] = morph_data['P_BAR_GZ2'][i]
            #data['DL_bar_N10'][gal_i] = morph_data['P_BAR_N10'][i]
            #data['DL_merge'][gal_i] = morph_data['P_MERG'][i]
            ####################################################################
    ############################################################################
    return smoothness_Table
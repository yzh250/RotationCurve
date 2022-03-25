import numpy as np
import numpy.ma as ma
from astropy.table import Table, QTable
import csv
from morph_smoothness_study_functions import galaxies_dict, match_morph_dl

################################################################################
# Used files (local)
#-------------------------------------------------------------------------------
MANGA_FOLDER = '/Users/richardzhang/Documents/UR_Stuff/Research_UR/SDSS/dr16/manga/spectro/'

DRP_FILENAME = MANGA_FOLDER + 'redux/v3_1_1/drpall-v3_1_1.fits'

SMOOTHNESS_FOLDER = '/Users/richardzhang/Documents/UR_Stuff/Research_UR/RotationCurve/2D_RC/main/'

smoothness_file = SMOOTHNESS_FOLDER + 'gal_smoothness.csv'
################################################################################

DTable =  Table.read(DRP_FILENAME, format='fits')

galaxy_ID_list = []
plateifu = DTable['plateifu'].data

for i in range(len(plateifu)):
    galaxy_ID_list.append(str(plateifu[i],'utf-8'))

smoothness_Table = Table.read(smoothness_file,format='ascii.csv')

match_morph_dl(galaxy_ID_list,smoothness_Table)

smoothness_Table.write('cross_table.csv',format='ascii.commented_header',overwrite=True)
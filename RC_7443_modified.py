################################################################################
# All the libraries used & constant values
# -------------------------------------------------------------------------------
import numpy as np

import matplotlib.pyplot as plt

from astropy.table import QTable

from scipy.optimize import minimize

from rotation_curve_functions import nloglike_Bur, v_co_Burket, v_co_Burket_nb,v_d, vel_h_Burket, vel_b, nloglike_Bur_nb, RC_fitting, RC_plotting

import dynesty
from dynesty import plotting as dyplot
################################################################################

################################################################################
# Reading in the master table
master_table = QTable.read('Master_Table.txt', format='ascii.commented_header')

################################################################################

################################################################################
# Reading in all the objects with Plate No. 7443
# -------------------------------------------------------------------------------
plate_ID = '7443'

IFU = ['1901','1902','3701','3702','3703','3704','6101','6102','6103','6104','9101','9102','12701','12702','12703','12704','12705']

galaxy_ID_list = []

for ifu in IFU:
    galaxy_ID_list.append(plate_ID+'-'+ifu)

Mass_list = []
for i in IFU:
    for j in range(len(master_table)):
        if master_table['MaNGA_plate'][j] == int(plate_ID) and master_table['MaNGA_IFU'][j] == int(i):
            Mass_list.append(master_table['NSA_Mstar'][j])

DTable_list = []
for i in range(len(galaxy_ID_list)):
    DTable_list.append(QTable.read(galaxy_ID_list[i]+'_rot_curve_data.txt', format='ascii.ecsv'))
################################################################################

################################################################################
for i in range(len(DTable_list)):

    # reading in radii and velocities
    r = DTable_list[i]['deprojected_distance'].data
    av = DTable_list[i]['rot_vel_avg'].data
    mav = DTable_list[i]['max_velocity'].data
    miv = DTable_list[i]['min_velocity'].data
    av_err = DTable_list[i]['rot_vel_avg_error'].data
    mav_err = DTable_list[i]['max_velocity_error'].data
    miv_err = DTable_list[i]['min_velocity_error'].data
    # If the object has a bulge
    # Plot from 0 to 5 times disk radii
    # Average Velocity
    av_fit = RC_fitting(r,Mass_list[i],av,av_err)
    RC_plotting(r,av,av_err,av_fit,galaxy_ID_list[i])
    mav_fit = RC_fitting(r,Mass_list[i],mav,av_err)
    RC_plotting(r,mav,mav_err,mav_fit,galaxy_ID_list[i])
    miv_fit = RC_fitting(r,Mass_list[i],np.abs(miv),miv_err)
    RC_plotting(r,np.abs(miv),miv_err,miv_fit,galaxy_ID_list[i])
################################################################################

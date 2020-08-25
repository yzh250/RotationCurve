################################################################################
# All the libraries used & constant values
# -------------------------------------------------------------------------------
import numpy as np

import matplotlib.pyplot as plt

from astropy.table import QTable

from scipy.optimize import minimize

from rotation_curve_functions import nloglike_Bur, v_co_Burket, v_co_Burket_nb,v_d, vel_h_Burket, vel_b, nloglike_Bur_nb, RC_fitting_Bur, RC_plotting_Bur
################################################################################

################################################################################
# Reading in the master table
master_table = QTable.read('Master_Table.txt', format='ascii.commented_header')
M_table = QTable.read('DRPall-master_file.txt', format='ascii.ecsv')
WF50 = M_table['WF50'].data
################################################################################

################################################################################
# Reading in all the objects with Plate No. 7443
# -------------------------------------------------------------------------------
plate_ID_list = ['8939','9507','8942','8940','8982','9493']
IFU_list = ['12704','12704','12703','12702','12702','12702']
galaxy_ID_list = []

# List of galaxy ID
for i in range(len(plate_ID_list)):
    galaxy_ID_list.append(plate_ID_list[i] + '-' + IFU_list[i])
print(galaxy_ID_list)

# Get stellar mass by galaxy ID
Mass_list = []
for i in range(len(galaxy_ID_list)):
    for j in range(len(master_table)):
        if master_table['MaNGA_plate'][j] == int(plate_ID_list[i]) and master_table['MaNGA_IFU'][j] == int(IFU_list[i]):
            Mass_list.append(master_table['NSA_Mstar'][j])
print(Mass_list)

# Get HI data by galaxy ID
WF50_list = []
for i in range(len(galaxy_ID_list)):
    for j in range(len(M_table)):
        if M_table['MaNGA_plate'][j] == int(plate_ID_list[i]) and M_table['MaNGA_IFU'][j] == int(IFU_list[i]):
            WF50_list.append(WF50[j])
print(WF50_list)

# Get rotation curve data by galaxy ID
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
    # Burket Model
    av_fit_Bur = RC_fitting_Bur(r,Mass_list[i],av,av_err,WF50_list[i])
    RC_plotting_Bur(r,av,av_err,av_fit_Bur,galaxy_ID_list[i])
    mav_fit_Bur = RC_fitting_Bur(r,Mass_list[i],mav,mav_err,WF50_list[i])
    RC_plotting_Bur(r,mav,mav_err,mav_fit_Bur,galaxy_ID_list[i])
    miv_fit_Bur = RC_fitting_Bur(r,Mass_list[i],np.abs(miv),miv_err,WF50_list[i])
    RC_plotting_Bur(r,np.abs(miv),miv_err,miv_fit_Bur,galaxy_ID_list[i])
################################################################################



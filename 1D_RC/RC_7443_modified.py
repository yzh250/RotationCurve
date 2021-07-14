################################################################################
# All the libraries used & constant values
# -------------------------------------------------------------------------------
import numpy as np

import matplotlib.pyplot as plt

from astropy.table import QTable

from scipy.optimize import minimize

from rotation_curve_functions import nloglike_Bur, v_co_Burket, v_co_Burket_nb,v_d, vel_h_Burket, vel_b, nloglike_Bur_nb, RC_fitting_Bur, RC_plotting_Bur, RC_fitting_Iso, RC_plotting_Iso

import dynesty
from dynesty import plotting as dyplot
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
print(len(Mass_list))

WF50_list = []
for i in IFU:
    for j in range(len(M_table)):
        if M_table['MaNGA_plate'][j] == int(plate_ID) and M_table['MaNGA_IFU'][j] == int(i):
            WF50_list.append(WF50[j])
print(WF50_list)
print(len(WF50_list))
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
    # Isothermal
    '''
    av_fit_Iso = RC_fitting_Iso(r, Mass_list[i], av, av_err)
    RC_plotting_Iso(r, av, av_err, av_fit_Iso, galaxy_ID_list[i])
    mav_fit_Iso = RC_fitting_Iso(r, Mass_list[i], mav, av_err)
    RC_plotting_Iso(r, mav, mav_err, mav_fit_Iso, galaxy_ID_list[i])
    miv_fit_Iso = RC_fitting_Iso(r, Mass_list[i], np.abs(miv), miv_err)
    RC_plotting_Iso(r, np.abs(miv), miv_err, miv_fit_Iso, galaxy_ID_list[i])
    '''
    # Burket
    av_fit_Bur = RC_fitting_Bur(r,Mass_list[i],av,av_err,WF50_list[i])
    RC_plotting_Bur(r,av,av_err,av_fit_Bur,galaxy_ID_list[i])
    mav_fit_Bur = RC_fitting_Bur(r,Mass_list[i],mav,av_err,WF50_list[i])
    RC_plotting_Bur(r,mav,mav_err,mav_fit_Bur,galaxy_ID_list[i])
    miv_fit_Bur = RC_fitting_Bur(r,Mass_list[i],np.abs(miv),miv_err,WF50_list[i])
    RC_plotting_Bur(r,np.abs(miv),miv_err,miv_fit_Bur,galaxy_ID_list[i])

################################################################################

################################################################################
# All the libraries used & constant values
# -------------------------------------------------------------------------------
import numpy as np

import matplotlib.pyplot as plt

from astropy.table import QTable

from scipy.optimize import minimize

from velocity_functions import nloglike_Bur, v_co_Burket, v_d, vel_h_Burket, vel_b

import dynesty
from dynesty import plotting as dyplot

################################################################################

# Initial guesses
p0 = [0.2,0.1,10.77897, 4, 0.02, 20]

param_bounds = [[0.2,1], # Scale Factor [unitless]
                [0,1000], # Bulge Scale Velocity [km/s]
                [0, 12],  # Disk mass [log(Msun)]
                [0, 10],  # Disk radius [kpc]
                [0, 1],  # Halo density [Msun/pc^2]
                [0, 100]]  # Halo radius [kpc]

################################################################################
# Reading in all the objects with Plate No. 7443
# -------------------------------------------------------------------------------
plate_ID = '7443'

IFU_list_1 = ['-1901','-1902']
IFU_list_2 = ['-3701','-3702','-3703','-3704']
IFU_list_3 = ['-6101','-6102','-6103','-6104']
IFU_list_4 = ['-9101','-9102']
IFU_list_5 = ['-12701','-12702','-12703','-12704','-12705']

DTable_list_1 = []
for i in range(len(IFU_list_1)):
    DTable_list_1.append(QTable.read(plate_ID+IFU_list_1[i]+'_rot_curve_data.txt', format='ascii.ecsv'))
DTable_list_2 = []
for i in range(len(IFU_list_2)):
    DTable_list_1.append(QTable.read(plate_ID+IFU_list_2[i]+'_rot_curve_data.txt', format='ascii.ecsv'))
DTable_list_3 = []
for i in range(len(IFU_list_3)):
    DTable_list_1.append(QTable.read(plate_ID+IFU_list_3[i]+'_rot_curve_data.txt', format='ascii.ecsv'))
DTable_list_4 = []
for i in range(len(IFU_list_3)):
    DTable_list_1.append(QTable.read(plate_ID+IFU_list_3[i]+'_rot_curve_data.txt', format='ascii.ecsv'))
DTable_list_5 = []
for i in range(len(IFU_list_3)):
    DTable_list_1.append(QTable.read(plate_ID+IFU_list_3[i]+'_rot_curve_data.txt', format='ascii.ecsv'))
################################################################################

################################################################################
# Reading in the radii, velocity data
# -------------------------------------------------------------------------------
r_list_1 = []
for i in range(len(DTable_list_1)):
    r_list_1.append(DTable_list_1[i]['deprojected_distance'].data)
r_list_2 = []
for i in range(len(DTable_list_2)):
    r_list_2.append(DTable_list_2[i]['deprojected_distance'].data)
r_list_3 = []
for i in range(len(DTable_list_3)):
    r_list_3.append(DTable_list_3[i]['deprojected_distance'].data)
r_list_4 = []
for i in range(len(DTable_list_4)):
    r_list_4.append(DTable_list_4[i]['deprojected_distance'].data)
r_list_5 = []
for i in range(len(DTable_list_5)):
    r_list_5.append(DTable_list_5[i]['deprojected_distance'].data)
av_list_1 = []
for i in range(len(DTable_list_1)):
    r_list_1.append(DTable_list_1[i]['rot_vel_avg'].data)
av_list_2 = []
for i in range(len(DTable_list_2)):
    r_list_2.append(DTable_list_2[i]['rot_vel_avg'].data)
av_list_3 = []
for i in range(len(DTable_list_3)):
    r_list_3.append(DTable_list_3[i]['rot_vel_avg'].data)
av_list_4 = []
for i in range(len(DTable_list_4)):
    r_list_4.append(DTable_list_4[i]['rot_vel_avg'].data)
av_list_5 = []
for i in range(len(DTable_list_5)):
    r_list_5.append(DTable_list_5[i]['rot_vel_avg'].data)
mav_list_1 = []
for i in range(len(DTable_list_1)):
    r_list_1.append(DTable_list_1[i]['max_velocity'].data)
mav_list_2 = []
for i in range(len(DTable_list_2)):
    r_list_2.append(DTable_list_2[i]['max_velocity'].data)
mav_list_3 = []
for i in range(len(DTable_list_3)):
    r_list_3.append(DTable_list_3[i]['max_velocity'].data)
mav_list_4 = []
for i in range(len(DTable_list_4)):
    r_list_4.append(DTable_list_4[i]['max_velocity'].data)
mav_list_5 = []
for i in range(len(DTable_list_5)):
    r_list_5.append(DTable_list_5[i]['max_velocity'].data)
miv_list_1 = []
for i in range(len(DTable_list_1)):
    r_list_1.append(DTable_list_1[i]['min_velocity'].data)
miv_list_2 = []
for i in range(len(DTable_list_2)):
    r_list_2.append(DTable_list_2[i]['min_velocity'].data)
miv_list_3 = []
for i in range(len(DTable_list_3)):
    r_list_3.append(DTable_list_3[i]['min_velocity'].data)
miv_list_4 = []
for i in range(len(DTable_list_4)):
    r_list_4.append(DTable_list_4[i]['min_velocity'].data)
miv_list_5 = []
for i in range(len(DTable_list_5)):
    r_list_5.append(DTable_list_5[i]['min_velocity'].data)
av_err_list_1 = []
for i in range(len(DTable_list_1)):
    r_list_1.append(DTable_list_1[i]['rot_vel_avg_error'].data)
av_err_list_2 = []
for i in range(len(DTable_list_2)):
    r_list_2.append(DTable_list_2[i]['rot_vel_avg_error'].data)
av_err_list_3 = []
for i in range(len(DTable_list_3)):
    r_list_3.append(DTable_list_3[i]['rot_vel_avg_error'].data)
av_err_list_4 = []
for i in range(len(DTable_list_4)):
    r_list_4.append(DTable_list_4[i]['rot_vel_avg_error'].data)
av_err_list_5 = []
for i in range(len(DTable_list_5)):
    r_list_5.append(DTable_list_5[i]['rot_vel_avg_error'].data)
mav_err_list_1 = []
for i in range(len(DTable_list_1)):
    r_list_1.append(DTable_list_1[i]['max_velocity_error'].data)
mav_err_list_2 = []
for i in range(len(DTable_list_2)):
    r_list_2.append(DTable_list_2[i]['max_velocity_error'].data)
mav_err_list_3 = []
for i in range(len(DTable_list_3)):
    r_list_3.append(DTable_list_3[i]['max_velocity_error'].data)
mav_err_list_4 = []
for i in range(len(DTable_list_4)):
    r_list_4.append(DTable_list_4[i]['max_velocity_error'].data)
mav_err_list_5 = []
for i in range(len(DTable_list_5)):
    r_list_5.append(DTable_list_5[i]['max_velocity_error'].data)
miv_err_list_1 = []
for i in range(len(DTable_list_1)):
    r_list_1.append(DTable_list_1[i]['min_velocity_error'].data)
miv_err_list_2 = []
for i in range(len(DTable_list_2)):
    r_list_2.append(DTable_list_2[i]['min_velocity_error'].data)
miv_err_list_3 = []
for i in range(len(DTable_list_3)):
    r_list_3.append(DTable_list_3[i]['min_velocity_error'].data)
miv_err_list_4 = []
for i in range(len(DTable_list_4)):
    r_list_4.append(DTable_list_4[i]['min_velocity_error'].data)
miv_err_list_5 = []
for i in range(len(DTable_list_5)):
    r_list_5.append(DTable_list_5[i]['min_velocity_error'].data)
################################################################################
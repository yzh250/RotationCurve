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
# Mass from Master Table
p0 = [0.2,0.1,10.77, 4, 0.02, 20]

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
    DTable_list_2.append(QTable.read(plate_ID+IFU_list_2[i]+'_rot_curve_data.txt', format='ascii.ecsv'))
DTable_list_3 = []
for i in range(len(IFU_list_3)):
    DTable_list_3.append(QTable.read(plate_ID+IFU_list_3[i]+'_rot_curve_data.txt', format='ascii.ecsv'))
DTable_list_4 = []
for i in range(len(IFU_list_4)):
    DTable_list_4.append(QTable.read(plate_ID+IFU_list_4[i]+'_rot_curve_data.txt', format='ascii.ecsv'))
DTable_list_5 = []
for i in range(len(IFU_list_5)):
    DTable_list_5.append(QTable.read(plate_ID+IFU_list_5[i]+'_rot_curve_data.txt', format='ascii.ecsv'))
################################################################################

################################################################################
# Reading in the radii, velocity data
# -------------------------------------------------------------------------------
# 7443-190~
r_list_1 = []
for i in range(len(DTable_list_1)):
    r_list_1.append(DTable_list_1[i]['deprojected_distance'].data)
av_list_1 = []
for i in range(len(DTable_list_1)):
    av_list_1.append(DTable_list_1[i]['rot_vel_avg'].data)
mav_list_1 = []
for i in range(len(DTable_list_1)):
    mav_list_1.append(DTable_list_1[i]['max_velocity'].data)
miv_list_1 = []
for i in range(len(DTable_list_1)):
    miv_list_1.append(DTable_list_1[i]['min_velocity'].data)
av_err_list_1 = []
for i in range(len(DTable_list_1)):
    av_err_list_1.append(DTable_list_1[i]['rot_vel_avg_error'].data)
mav_err_list_1 = []
for i in range(len(DTable_list_1)):
    mav_err_list_1.append(DTable_list_1[i]['max_velocity_error'].data)
miv_err_list_1 = []
for i in range(len(DTable_list_1)):
    miv_err_list_1.append(DTable_list_1[i]['min_velocity_error'].data)
# -------------------------------------------------------------------------------

# -------------------------------------------------------------------------------
# 7443-370~
r_list_2 = []
for i in range(len(DTable_list_2)):
    r_list_2.append(DTable_list_2[i]['deprojected_distance'].data)
av_list_2 = []
for i in range(len(DTable_list_2)):
    av_list_2.append(DTable_list_2[i]['rot_vel_avg'].data)
mav_list_2 = []
for i in range(len(DTable_list_2)):
    mav_list_2.append(DTable_list_2[i]['max_velocity'].data)
miv_list_2 = []
for i in range(len(DTable_list_2)):
    miv_list_2.append(DTable_list_2[i]['min_velocity'].data)
av_err_list_2 = []
for i in range(len(DTable_list_2)):
    av_err_list_2.append(DTable_list_2[i]['rot_vel_avg_error'].data)
mav_err_list_2 = []
for i in range(len(DTable_list_2)):
    mav_err_list_2.append(DTable_list_2[i]['max_velocity_error'].data)
miv_err_list_2 = []
for i in range(len(DTable_list_2)):
    miv_err_list_2.append(DTable_list_2[i]['min_velocity_error'].data)
# -------------------------------------------------------------------------------

# -------------------------------------------------------------------------------
# 7443-610~
r_list_3 = []
for i in range(len(DTable_list_3)):
    r_list_3.append(DTable_list_3[i]['deprojected_distance'].data)
av_list_3 = []
for i in range(len(DTable_list_3)):
    av_list_3.append(DTable_list_3[i]['rot_vel_avg'].data)
mav_list_3 = []
for i in range(len(DTable_list_3)):
    mav_list_3.append(DTable_list_3[i]['max_velocity'].data)
miv_list_3 = []
for i in range(len(DTable_list_3)):
    miv_list_3.append(DTable_list_3[i]['min_velocity'].data)
av_err_list_3 = []
for i in range(len(DTable_list_3)):
    av_err_list_3.append(DTable_list_3[i]['rot_vel_avg_error'].data)
mav_err_list_3 = []
for i in range(len(DTable_list_3)):
    mav_err_list_3.append(DTable_list_3[i]['max_velocity_error'].data)
miv_err_list_3 = []
for i in range(len(DTable_list_3)):
    miv_err_list_3.append(DTable_list_3[i]['min_velocity_error'].data)
# -------------------------------------------------------------------------------

# -------------------------------------------------------------------------------
# 7443-910~
r_list_4 = []
for i in range(len(DTable_list_4)):
    r_list_4.append(DTable_list_4[i]['deprojected_distance'].data)
av_list_4 = []
for i in range(len(DTable_list_4)):
    av_list_4.append(DTable_list_4[i]['rot_vel_avg'].data)
mav_list_4 = []
for i in range(len(DTable_list_4)):
    mav_list_4.append(DTable_list_4[i]['max_velocity'].data)
miv_list_4 = []
for i in range(len(DTable_list_4)):
    miv_list_4.append(DTable_list_4[i]['min_velocity'].data)
av_err_list_4 = []
for i in range(len(DTable_list_4)):
    av_err_list_4.append(DTable_list_4[i]['rot_vel_avg_error'].data)
mav_err_list_4 = []
for i in range(len(DTable_list_4)):
    mav_err_list_4.append(DTable_list_4[i]['max_velocity_error'].data)
miv_err_list_4 = []
for i in range(len(DTable_list_4)):
    miv_err_list_4.append(DTable_list_4[i]['min_velocity_error'].data)
# -------------------------------------------------------------------------------

# -------------------------------------------------------------------------------
# 7443-1270~
r_list_5 = []
for i in range(len(DTable_list_5)):
    r_list_5.append(DTable_list_5[i]['deprojected_distance'].data)
av_list_5 = []
for i in range(len(DTable_list_5)):
    av_list_5.append(DTable_list_5[i]['rot_vel_avg'].data)
mav_list_5 = []
for i in range(len(DTable_list_5)):
    mav_list_5.append(DTable_list_5[i]['max_velocity'].data)
miv_list_5 = []
for i in range(len(DTable_list_5)):
    miv_list_5.append(DTable_list_5[i]['min_velocity'].data)
av_err_list_5 = []
for i in range(len(DTable_list_5)):
    av_err_list_5.append(DTable_list_5[i]['rot_vel_avg_error'].data)
mav_err_list_5 = []
for i in range(len(DTable_list_5)):
    mav_err_list_5.append(DTable_list_5[i]['max_velocity_error'].data)
miv_err_list_5 = []
for i in range(len(DTable_list_5)):
    miv_err_list_5.append(DTable_list_5[i]['min_velocity_error'].data)
################################################################################

################################################################################
for i in range(len(DTable_list_1)):
    bestfit_av = minimize(nloglike_Bur, p0, args=(r_list_1[i], av_list_1[i], av_err_list_1[i]),
                      bounds=param_bounds)
    print('---------------------------------------------------')
    print('Average curve')
    print(bestfit_av)

    r_normalized = r_list_1[i] / bestfit_av.x[1]

    plt.errorbar(r_normalized, av_list_1[i], yerr=av_err_list_1[i], fmt='g*', label='Average')
    plt.plot(r_normalized, v_co_Burket(np.array(r_list_1[i]), bestfit_av.x), '--', label='fit')
    plt.plot(r_normalized, vel_b(np.array(r_list_1[i]) * 1000, bestfit_av.x[0], bestfit_av.x[1], bestfit_av.x[3] * 1000), label='bulge')
    plt.plot(r_normalized, v_d(np.array(r_list_1[i]) * 1000, 10 ** bestfit_av.x[2], bestfit_av.x[3] * 1000), label='disk')
    plt.plot(r_normalized, vel_h_Burket(np.array(r_list_1[i]) * 1000, bestfit_av.x[4], bestfit_av.x[5] * 1000), label='Burket halo')
    plt.legend()
    plt.xlabel('$r_{dep}$/$R_d$')
    plt.ylabel('$v_{rot}$ [km/s]')
    plt.title(plate_ID+IFU_list_1[i])
    plt.show()
    # -------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------
    # Positive rotation curve
    # -------------------------------------------------------------------------------
    # Initial guesses

    bestfit_mav = minimize(nloglike_Bur, p0, args=(r_list_1[i], mav_list_1[i], mav_err_list_1[i]),
                       bounds=param_bounds)
    print('---------------------------------------------------')
    print('Positive curve')
    print(bestfit_mav)

    r_normalized_mav = r_list_1[i] / bestfit_mav.x[1]

    plt.errorbar(r_normalized_mav, mav_list_1[i], yerr=mav_err_list_1[i], fmt='r*', label='data')
    plt.plot(r_normalized_mav, v_co_Burket(np.array(r_list_1[i]), bestfit_mav.x), '--', label='fit')
    plt.plot(r_normalized_mav, vel_b(np.array(r_list_1[i]) * 1000, bestfit_mav.x[0], bestfit_mav.x[1], bestfit_mav.x[3] * 1000), label='bulge')
    plt.plot(r_normalized_mav, v_d(np.array(r_list_1[i]) * 1000, 10 ** (bestfit_mav.x[2]), bestfit_mav.x[3] * 1000), label='disk')
    plt.plot(r_normalized_mav, vel_h_Burket(np.array(r_list_1[i]) * 1000, bestfit_mav.x[4], bestfit_mav.x[5] * 1000), label='Burket halo')
    plt.legend()
    plt.xlabel('$r_{dep}$/$R_d$')
    plt.ylabel('$v_{rot}$ [km/s]')
    plt.title(plate_ID+IFU_list_1[i])
    plt.show()
    # -------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------
    # Negative rotation curve
    # -------------------------------------------------------------------------------
    bestfit_miv = minimize(nloglike_Bur, p0, args=(r_list_1[i], np.abs(miv_list_1[i]), miv_err_list_1[i]),
                       bounds=param_bounds)
    print('---------------------------------------------------')
    print('Negative curve')
    print(bestfit_miv)

    r_normalized_miv = r_list_1[i] / bestfit_miv.x[1]
    plt.errorbar(r_normalized_miv, np.abs(miv_list_1[i]), yerr=miv_err_list_1[i], fmt='b*', label='data')
    plt.plot(r_normalized_miv, v_co_Burket(np.array(r_list_1[i]), bestfit_miv.x), '--', label='fit')
    plt.plot(r_normalized_miv, vel_b(np.array(r_list_1[i]) * 1000, bestfit_miv.x[0], bestfit_miv.x[1], bestfit_miv.x[3] * 1000), label='bulge')
    plt.plot(r_normalized_miv, v_d(np.array(r_list_1[i]) * 1000, 10 ** bestfit_miv.x[2], bestfit_miv.x[3] * 1000), label='disk')
    plt.plot(r_normalized_miv, vel_h_Burket(np.array(r_list_1[i]) * 1000, bestfit_miv.x[4], bestfit_miv.x[5] * 1000), label='Burket halo')
    plt.legend()
    plt.xlabel('$r_{dep}$/$R_d$')
    plt.ylabel('$v_{rot}$ [km/s]')
    plt.title(plate_ID+IFU_list_1[i])
    plt.show()
################################################################################

################################################################################
for i in range(len(DTable_list_2)):
    bestfit_av_2 = minimize(nloglike_Bur, p0, args=(r_list_2[i], av_list_2[i], av_err_list_2[i]),
                      bounds=param_bounds)
    print('---------------------------------------------------')
    print('Average curve')
    print(bestfit_av_2)

    r_normalized_2 = r_list_2[i] / bestfit_av_2.x[1]

    plt.errorbar(r_normalized_2, av_list_2[i], yerr=av_err_list_2[i], fmt='g*', label='Average')
    plt.plot(r_normalized_2, v_co_Burket(np.array(r_list_2[i]), bestfit_av_2.x), '--', label='fit')
    plt.plot(r_normalized_2, vel_b(np.array(r_list_2[i]) * 1000, bestfit_av_2.x[0], bestfit_av_2.x[1], bestfit_av_2.x[3] * 1000), label='bulge')
    plt.plot(r_normalized_2, v_d(np.array(r_list_2[i]) * 1000, 10 ** bestfit_av_2.x[2], bestfit_av_2.x[3] * 1000), label='disk')
    plt.plot(r_normalized_2, vel_h_Burket(np.array(r_list_2[i]) * 1000, bestfit_av_2.x[4], bestfit_av_2.x[5] * 1000), label='Burket halo')
    plt.legend()
    plt.xlabel('$r_{dep}$/$R_d$')
    plt.ylabel('$v_{rot}$ [km/s]')
    plt.title(plate_ID+IFU_list_2[i])
    plt.show()
    # -------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------
    # Positive rotation curve
    # -------------------------------------------------------------------------------
    # Initial guesses

    bestfit_mav_2 = minimize(nloglike_Bur, p0, args=(r_list_2[i], mav_list_2[i], mav_err_list_2[i]),
                       bounds=param_bounds)
    print('---------------------------------------------------')
    print('Positive curve')
    print(bestfit_mav_2)

    r_normalized_mav_2 = r_list_2[i] / bestfit_mav_2.x[1]

    plt.errorbar(r_normalized_mav_2, mav_list_2[i], yerr=mav_err_list_2[i], fmt='r*', label='data')
    plt.plot(r_normalized_mav_2, v_co_Burket(np.array(r_list_2[i]), bestfit_mav_2.x), '--', label='fit')
    plt.plot(r_normalized_mav_2, vel_b(np.array(r_list_2[i]) * 1000, bestfit_mav_2.x[0], bestfit_mav_2.x[1], bestfit_mav_2.x[3] * 1000), label='bulge')
    plt.plot(r_normalized_mav_2, v_d(np.array(r_list_2[i]) * 1000, 10 ** (bestfit_mav_2.x[2]), bestfit_mav_2.x[3] * 1000), label='disk')
    plt.plot(r_normalized_mav_2, vel_h_Burket(np.array(r_list_2[i]) * 1000, bestfit_mav_2.x[4], bestfit_mav_2.x[5] * 1000), label='Burket halo')
    plt.legend()
    plt.xlabel('$r_{dep}$/$R_d$')
    plt.ylabel('$v_{rot}$ [km/s]')
    plt.title(plate_ID+IFU_list_2[i])
    plt.show()
    # -------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------
    # Negative rotation curve
    # -------------------------------------------------------------------------------
    bestfit_miv_2 = minimize(nloglike_Bur, p0, args=(r_list_2[i], np.abs(miv_list_2[i]), miv_err_list_2[i]),
                       bounds=param_bounds)
    print('---------------------------------------------------')
    print('Negative curve')
    print(bestfit_miv_2)

    r_normalized_miv_2 = r_list_2[i] / bestfit_miv_2.x[1]
    plt.errorbar(r_normalized_miv_2, np.abs(miv_list_2[i]), yerr=miv_err_list_2[i], fmt='b*', label='data')
    plt.plot(r_normalized_miv_2, v_co_Burket(np.array(r_list_2[i]), bestfit_miv_2.x), '--', label='fit')
    plt.plot(r_normalized_miv_2, vel_b(np.array(r_list_2[i]) * 1000, bestfit_miv_2.x[0], bestfit_miv_2.x[1], bestfit_miv_2.x[3] * 1000), label='bulge')
    plt.plot(r_normalized_miv_2, v_d(np.array(r_list_2[i]) * 1000, 10 ** bestfit_miv_2.x[2], bestfit_miv_2.x[3] * 1000), label='disk')
    plt.plot(r_normalized_miv_2, vel_h_Burket(np.array(r_list_2[i]) * 1000, bestfit_miv_2.x[4], bestfit_miv_2.x[5] * 1000), label='Burket halo')
    plt.legend()
    plt.xlabel('$r_{dep}$/$R_d$')
    plt.ylabel('$v_{rot}$ [km/s]')
    plt.title(plate_ID + IFU_list_2[i])
    plt.show()
################################################################################

################################################################################
for i in range(len(DTable_list_3)):
    bestfit_av_3 = minimize(nloglike_Bur, p0, args=(r_list_3[i], av_list_3[i], av_err_list_3[i]),
                      bounds=param_bounds)
    print('---------------------------------------------------')
    print('Average curve')
    print(bestfit_av_3)

    r_normalized_3 = r_list_3[i] / bestfit_av_3.x[1]

    plt.errorbar(r_normalized_3, av_list_3[i], yerr=av_err_list_3[i], fmt='g*', label='Average')
    plt.plot(r_normalized_3, v_co_Burket(np.array(r_list_3[i]), bestfit_av_3.x), '--', label='fit')
    plt.plot(r_normalized_3, vel_b(np.array(r_list_3[i]) * 1000, bestfit_av_3.x[0], bestfit_av_3.x[1], bestfit_av_3.x[3] * 1000), label='bulge')
    plt.plot(r_normalized_3, v_d(np.array(r_list_3[i]) * 1000, 10 ** bestfit_av_3.x[2], bestfit_av_3.x[3] * 1000), label='disk')
    plt.plot(r_normalized_3, vel_h_Burket(np.array(r_list_3[i]) * 1000, bestfit_av_3.x[4], bestfit_av_3.x[5] * 1000), label='Burket halo')
    plt.legend()
    plt.xlabel('$r_{dep}$/$R_d$')
    plt.ylabel('$v_{rot}$ [km/s]')
    plt.title(plate_ID+IFU_list_3[i])
    plt.show()
    # -------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------
    # Positive rotation curve
    # -------------------------------------------------------------------------------
    # Initial guesses

    bestfit_mav_3 = minimize(nloglike_Bur, p0, args=(r_list_3[i], mav_list_3[i], mav_err_list_3[i]),
                       bounds=param_bounds)
    print('---------------------------------------------------')
    print('Positive curve')
    print(bestfit_mav_3)

    r_normalized_mav_3 = r_list_3[i] / bestfit_mav_3.x[1]

    plt.errorbar(r_normalized_mav_3, mav_list_3[i], yerr=mav_err_list_3[i], fmt='r*', label='data')
    plt.plot(r_normalized_mav_3, v_co_Burket(np.array(r_list_3[i]), bestfit_mav_3.x), '--', label='fit')
    plt.plot(r_normalized_mav_3, vel_b(np.array(r_list_3[i]) * 1000, bestfit_mav_3.x[0], bestfit_mav_3.x[1], bestfit_mav_3.x[3] * 1000), label='bulge')
    plt.plot(r_normalized_mav_3, v_d(np.array(r_list_3[i]) * 1000, 10 ** (bestfit_mav_3.x[2]), bestfit_mav_3.x[3] * 1000), label='disk')
    plt.plot(r_normalized_mav_3, vel_h_Burket(np.array(r_list_3[i]) * 1000, bestfit_mav_3.x[4], bestfit_mav_3.x[5] * 1000), label='Burket halo')
    plt.legend()
    plt.xlabel('$r_{dep}$/$R_d$')
    plt.ylabel('$v_{rot}$ [km/s]')
    plt.title(plate_ID+IFU_list_3[i])
    plt.show()
    # -------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------
    # Negative rotation curve
    # -------------------------------------------------------------------------------
    bestfit_miv_3 = minimize(nloglike_Bur, p0, args=(r_list_3[i], np.abs(miv_list_3[i]), miv_err_list_3[i]),
                       bounds=param_bounds)
    print('---------------------------------------------------')
    print('Negative curve')
    print(bestfit_miv_3)

    r_normalized_miv_3 = r_list_3[i] / bestfit_miv_3.x[1]
    plt.errorbar(r_normalized_miv_3, np.abs(miv_list_3[i]), yerr=miv_err_list_3[i], fmt='b*', label='data')
    plt.plot(r_normalized_miv_3, v_co_Burket(np.array(r_list_3[i]), bestfit_miv_3.x), '--', label='fit')
    plt.plot(r_normalized_miv_3, vel_b(np.array(r_list_3[i]) * 1000, bestfit_miv_3.x[0], bestfit_miv_3.x[1], bestfit_miv_3.x[3] * 1000), label='bulge')
    plt.plot(r_normalized_miv_3, v_d(np.array(r_list_3[i]) * 1000, 10 ** bestfit_miv_3.x[2], bestfit_miv_3.x[3] * 1000), label='disk')
    plt.plot(r_normalized_miv_3, vel_h_Burket(np.array(r_list_3[i]) * 1000, bestfit_miv_3.x[4], bestfit_miv_3.x[5] * 1000), label='Burket halo')
    plt.legend()
    plt.xlabel('$r_{dep}$/$R_d$')
    plt.ylabel('$v_{rot}$ [km/s]')
    plt.title(plate_ID+IFU_list_3[i])
    plt.show()
################################################################################

################################################################################
for i in range(len(DTable_list_4)):
    bestfit_av_4 = minimize(nloglike_Bur, p0, args=(r_list_4[i], av_list_4[i], av_err_list_4[i]),
                      bounds=param_bounds)
    print('---------------------------------------------------')
    print('Average curve')
    print(bestfit_av_4)

    r_normalized_4 = r_list_4[i] / bestfit_av_4.x[1]

    plt.errorbar(r_normalized_4, av_list_4[i], yerr=av_err_list_4[i], fmt='g*', label='Average')
    plt.plot(r_normalized_4, v_co_Burket(np.array(r_list_4[i]), bestfit_av_4.x), '--', label='fit')
    plt.plot(r_normalized_4, vel_b(np.array(r_list_4[i]) * 1000, bestfit_av_4.x[0], bestfit_av_4.x[1], bestfit_av_4.x[3] * 1000), label='bulge')
    plt.plot(r_normalized_4, v_d(np.array(r_list_4[i]) * 1000, 10 ** bestfit_av_4.x[2], bestfit_av_4.x[3] * 1000), label='disk')
    plt.plot(r_normalized_4, vel_h_Burket(np.array(r_list_4[i]) * 1000, bestfit_av_4.x[4], bestfit_av_4.x[5] * 1000), label='Burket halo')
    plt.legend()
    plt.xlabel('$r_{dep}$/$R_d$')
    plt.ylabel('$v_{rot}$ [km/s]')
    plt.title(plate_ID+IFU_list_4[i])
    plt.show()
    # -------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------
    # Positive rotation curve
    # -------------------------------------------------------------------------------
    # Initial guesses

    bestfit_mav_4 = minimize(nloglike_Bur, p0, args=(r_list_4[i], mav_list_4[i], mav_err_list_4[i]),
                       bounds=param_bounds)
    print('---------------------------------------------------')
    print('Positive curve')
    print(bestfit_mav_4)

    r_normalized_mav_4 = r_list_4[i] / bestfit_mav_4.x[1]

    plt.errorbar(r_normalized_mav_4, mav_list_4[i], yerr=mav_err_list_4[i], fmt='r*', label='data')
    plt.plot(r_normalized_mav_4, v_co_Burket(np.array(r_list_4[i]), bestfit_mav_4.x), '--', label='fit')
    plt.plot(r_normalized_mav_4, vel_b(np.array(r_list_4[i]) * 1000, bestfit_mav_4.x[0], bestfit_mav_4.x[1], bestfit_mav_4.x[3] * 1000), label='bulge')
    plt.plot(r_normalized_mav_4, v_d(np.array(r_list_4[i]) * 1000, 10 ** (bestfit_mav_4.x[2]), bestfit_mav_4.x[3] * 1000), label='disk')
    plt.plot(r_normalized_mav_4, vel_h_Burket(np.array(r_list_4[i]) * 1000, bestfit_mav_4.x[4], bestfit_mav_4.x[5] * 1000), label='Burket halo')
    plt.legend()
    plt.xlabel('$r_{dep}$/$R_d$')
    plt.ylabel('$v_{rot}$ [km/s]')
    plt.title(plate_ID+IFU_list_4[i])
    plt.show()
    # -------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------
    # Negative rotation curve
    # -------------------------------------------------------------------------------
    bestfit_miv_4 = minimize(nloglike_Bur, p0, args=(r_list_4[i], np.abs(miv_list_4[i]), miv_err_list_4[i]),
                       bounds=param_bounds)
    print('---------------------------------------------------')
    print('Negative curve')
    print(bestfit_miv_4)

    r_normalized_miv_4 = r_list_4[i] / bestfit_miv_4.x[1]
    plt.errorbar(r_normalized_miv_4, np.abs(miv_list_4[i]), yerr=miv_err_list_4[i], fmt='b*', label='data')
    plt.plot(r_normalized_miv_4, v_co_Burket(np.array(r_list_4[i]), bestfit_miv.x), '--', label='fit')
    plt.plot(r_normalized_miv_4, vel_b(np.array(r_list_4[i]) * 1000, bestfit_miv_4.x[0], bestfit_miv_4.x[1], bestfit_miv_4.x[3] * 1000), label='bulge')
    plt.plot(r_normalized_miv_4, v_d(np.array(r_list_4[i]) * 1000, 10 ** bestfit_miv_4.x[2], bestfit_miv_4.x[3] * 1000), label='disk')
    plt.plot(r_normalized_miv_4, vel_h_Burket(np.array(r_list_4[i]) * 1000, bestfit_miv_4.x[4], bestfit_miv_4.x[5] * 1000), label='Burket halo')
    plt.legend()
    plt.xlabel('$r_{dep}$/$R_d$')
    plt.ylabel('$v_{rot}$ [km/s]')
    plt.title(plate_ID+IFU_list_4[i])
    plt.show()
################################################################################

################################################################################
for i in range(len(DTable_list_5)):
    bestfit_av_5 = minimize(nloglike_Bur, p0, args=(r_list_5[i], av_list_5[i], av_err_list_5[i]),
                      bounds=param_bounds)
    print('---------------------------------------------------')
    print('Average curve')
    print(bestfit_av_5)

    r_normalized_5 = r_list_5[i] / bestfit_av_5.x[1]

    plt.errorbar(r_normalized_5, av_list_5[i], yerr=av_err_list_5[i], fmt='g*', label='Average')
    plt.plot(r_normalized_5, v_co_Burket(np.array(r_list_5[i]), bestfit_av_5.x), '--', label='fit')
    plt.plot(r_normalized_5, vel_b(np.array(r_list_5[i]) * 1000, bestfit_av_5.x[0], bestfit_av_5.x[1], bestfit_av_5.x[3] * 1000), label='bulge')
    plt.plot(r_normalized_5, v_d(np.array(r_list_5[i]) * 1000, 10 ** bestfit_av_5.x[2], bestfit_av_5.x[3] * 1000), label='disk')
    plt.plot(r_normalized_5, vel_h_Burket(np.array(r_list_5[i]) * 1000, bestfit_av_5.x[4], bestfit_av_5.x[5] * 1000), label='Burket halo')
    plt.legend()
    plt.xlabel('$r_{dep}$/$R_d$')
    plt.ylabel('$v_{rot}$ [km/s]')
    plt.title(plate_ID+IFU_list_5[i])
    plt.show()
    # -------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------
    # Positive rotation curve
    # -------------------------------------------------------------------------------
    # Initial guesses

    bestfit_mav_5 = minimize(nloglike_Bur, p0, args=(r_list_5[i], mav_list_5[i], mav_err_list_5[i]),
                       bounds=param_bounds)
    print('---------------------------------------------------')
    print('Positive curve')
    print(bestfit_mav_5)

    r_normalized_mav_5 = r_list_5[i] / bestfit_mav_5.x[1]

    plt.errorbar(r_normalized_mav_5, mav_list_5[i], yerr=mav_err_list_5[i], fmt='r*', label='data')
    plt.plot(r_normalized_mav_5, v_co_Burket(np.array(r_list_5[i]), bestfit_mav_5.x), '--', label='fit')
    plt.plot(r_normalized_mav_5, vel_b(np.array(r_list_5[i]) * 1000, bestfit_mav_5.x[0], bestfit_mav_5.x[1], bestfit_mav_5.x[3] * 1000), label='bulge')
    plt.plot(r_normalized_mav_5, v_d(np.array(r_list_5[i]) * 1000, 10 ** (bestfit_mav_5.x[2]), bestfit_mav_5.x[3] * 1000), label='disk')
    plt.plot(r_normalized_mav_5, vel_h_Burket(np.array(r_list_5[i]) * 1000, bestfit_mav_5.x[4], bestfit_mav_5.x[5] * 1000), label='Burket halo')
    plt.legend()
    plt.xlabel('$r_{dep}$/$R_d$')
    plt.ylabel('$v_{rot}$ [km/s]')
    plt.title(plate_ID+IFU_list_5[i])
    plt.show()
    # -------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------
    # Negative rotation curve
    # -------------------------------------------------------------------------------
    bestfit_miv_5 = minimize(nloglike_Bur, p0, args=(r_list_5[i], np.abs(miv_list_5[i]), miv_err_list_5[i]),
                       bounds=param_bounds)
    print('---------------------------------------------------')
    print('Negative curve')
    print(bestfit_miv_5)

    r_normalized_miv_5 = r_list_5[i] / bestfit_miv_5.x[1]
    plt.errorbar(r_normalized_miv_5, np.abs(miv_list_5[i]), yerr=miv_err_list_5[i], fmt='b*', label='data')
    plt.plot(r_normalized_miv_5, v_co_Burket(np.array(r_list_5[i]), bestfit_miv_5.x), '--', label='fit')
    plt.plot(r_normalized_miv_5, vel_b(np.array(r_list_5[i]) * 1000, bestfit_miv_5.x[0], bestfit_miv_5.x[1], bestfit_miv_5.x[3] * 1000), label='bulge')
    plt.plot(r_normalized_miv_5, v_d(np.array(r_list_5[i]) * 1000, 10 ** bestfit_miv_5.x[2], bestfit_miv_5.x[3] * 1000), label='disk')
    plt.plot(r_normalized_miv_5, vel_h_Burket(np.array(r_list_5[i]) * 1000, bestfit_miv_5.x[4], bestfit_miv_5.x[5] * 1000), label='Burket halo')
    plt.legend()
    plt.xlabel('$r_{dep}$/$R_d$')
    plt.ylabel('$v_{rot}$ [km/s]')
    plt.title(plate_ID+IFU_list_5[i])
    plt.show()
################################################################################
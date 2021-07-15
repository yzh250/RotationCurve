################################################################################
# All the libraries used & constant values
# -------------------------------------------------------------------------------
import numpy as np

import matplotlib.pyplot as plt

from astropy.table import QTable

from scipy.optimize import minimize

from velocity_functions import nloglike_Bur_nb, v_co_Burket_nb, v_d, vel_h_Burket, vel_b

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
print(len(DTable_list))
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

    # Initial Guess
    p0 = [np.log10(Mass_list[i]), max(np.array(r)) / 3, 0.02, 20]
    param_bounds = [[0, 12],  # Disk mass [log(Msun)]
                    [0, 10],  # Disk radius [kpc]
                    [0, 1],  # Halo density [Msun/pc^2]
                    [0, 100]]  # Halo radius [kpc]

    bestfit_av = minimize(nloglike_Bur_nb, p0, args=(r, av, av_err),bounds=param_bounds)
    print('---------------------------------------------------')
    print('Average curve')
    print(bestfit_av)

    r_normalized = r / bestfit_av.x[1]

    plt.errorbar(r_normalized, av, yerr=av_err, fmt='g*', label='Average')
    plt.plot(r_normalized, v_co_Burket_nb(np.array(r), bestfit_av.x), '--', label='fit')
    plt.plot(r_normalized, v_d(np.array(r) * 1000, 10 ** bestfit_av.x[0], bestfit_av.x[1] * 1000), label='disk')
    plt.plot(r_normalized, vel_h_Burket(np.array(r) * 1000, bestfit_av.x[2], bestfit_av.x[3] * 1000),label='Burket halo')
    plt.legend()
    plt.xlabel('$r_{dep}$/$R_d$')
    plt.ylabel('$v_{rot}$ [km/s]')
    plt.title(galaxy_ID_list[i])
    plt.show()
    # -------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------
    # Positive rotation curve
    # -------------------------------------------------------------------------------
    # Initial guesses

    bestfit_mav = minimize(nloglike_Bur_nb, p0, args=(r, mav, mav_err),bounds=param_bounds)
    print('---------------------------------------------------')
    print('Positive curve')
    print(bestfit_mav)

    r_normalized_mav = r / bestfit_mav.x[1]

    plt.errorbar(r_normalized_mav, mav, yerr=mav_err, fmt='r*', label='data')
    plt.plot(r_normalized_mav, v_co_Burket_nb(np.array(r), bestfit_mav.x), '--', label='fit')
    plt.plot(r_normalized_mav, v_d(np.array(r) * 1000, 10 ** (bestfit_mav.x[0]), bestfit_mav.x[1] * 1000),label='disk')
    plt.plot(r_normalized_mav, vel_h_Burket(np.array(r) * 1000, bestfit_mav.x[2], bestfit_mav.x[3] * 1000),label='Burket halo')
    plt.legend()
    plt.xlabel('$r_{dep}$/$R_d$')
    plt.ylabel('$v_{rot}$ [km/s]')
    plt.title(galaxy_ID_list[i])
    plt.show()
    # -------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------
    # Negative rotation curve
    # -------------------------------------------------------------------------------
    bestfit_miv = minimize(nloglike_Bur_nb, p0, args=(r, np.abs(miv), miv_err),bounds=param_bounds)
    print('---------------------------------------------------')
    print('Negative curve')
    print(bestfit_miv)

    r_normalized_miv = r / bestfit_miv.x[1]
    plt.errorbar(r_normalized_miv, np.abs(miv), yerr=miv_err, fmt='b*', label='data')
    plt.plot(r_normalized_miv, v_co_Burket_nb(np.array(r), bestfit_miv.x), '--', label='fit')
    plt.plot(r_normalized_miv, v_d(np.array(r) * 1000, 10 ** bestfit_miv.x[0], bestfit_miv.x[1] * 1000),label='disk')
    plt.plot(r_normalized_miv, vel_h_Burket(np.array(r) * 1000, bestfit_miv.x[2], bestfit_miv.x[3] * 1000),label='Burket halo')
    plt.legend()
    plt.xlabel('$r_{dep}$/$R_d$')
    plt.ylabel('$v_{rot}$ [km/s]')
    plt.title(galaxy_ID_list[i])
    plt.show()
################################################################################

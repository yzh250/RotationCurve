################################################################################
# All the libraries used & constant values
# -------------------------------------------------------------------------------
import numpy as np

import matplotlib.pyplot as plt

from astropy.table import Table

from scipy.optimize import minimize

from rotation_curve_functions import nloglike_Bur, v_co_Burket, v_d, vel_h_Burket, vel_b

import dynesty
from dynesty import plotting as dyplot
################################################################################

################################################################################
# Reading in the master table
master_table = Table.read('Master_Table.txt', format='ascii.commented_header')

################################################################################

################################################################################
# store the best fits
bestfits_av = []
bestfits_mav = []
bestfits_miv = []
for i in range(len(master_table)):
    # Initial Guess:
        # free parameter
        # scale velocity of the bulge
        # log10(Mdisk)
        # scale radius of disk
        # central radius of halo
        # scale radius of halo
    p0 = [0.2,0.1,10.77897, 4, 0.02, 20]

    # bounds

    param_bounds = [[0.2, 1],  # Scale Factor [unitless]
                    [0, 1000],  # Bulge Scale Velocity [km/s]
                    [0, 12],  # Disk mass [log(Msun)]
                    [0, 10],  # Disk radius [kpc]
                    [0, 1],  # Halo density [Msun/pc^2]
                    [0, 100]]  # Halo radius [kpc]

    # construct the galaxy ID in row i of the table
    galaxy_ID = master_table['MaNGA_plate'][i] + master_table['MaNGA_IFU'][i]

    # read in the data file for galaxy i

    r_av = master_table['avg_r_turn'][i].data
    r_mav = master_table['pos_r_turn'][i].data
    r_miv = master_table['neg_r_turn'][i].data
    av = master_table['avg_v_max'][i].data
    mav = master_table['pos_v_max'][i].data
    miv = master_table['neg_v_max'][i].data
    av_err = master_table['avg_v_max_sigma'][i].data
    mav_err = master_table['pos_v_max_sigma'][i].data
    miv_err = master_table['neg_v_max_sigma'][i].data

    # fit the rotation curve for galaxy i

    bestfit_av = minimize(nloglike_Bur, p0, args=(r_av, av, av_err),
                      bounds=param_bounds)
    print('---------------------------------------------------')
    print('Average curve')
    print(bestfit_av)
    bestfits_av.append(bestfit_av)

    r_normalized = r_av / bestfit_av.x[1]

    plt.errorbar(r_normalized, av, yerr=av_err, fmt='g*', label='Average')
    plt.plot(r_normalized, v_co_Burket(np.array(r), bestfit_av.x), '--', label='fit')
    plt.plot(r_normalized, vel_b(np.array(r) * 1000, bestfit_av.x[0], bestfit_av.x[1], bestfit_av.x[3] * 1000),
         label='bulge')
    plt.plot(r_normalized, v_d(np.array(r) * 1000, 10 ** bestfit_av.x[2], bestfit_av.x[3] * 1000), label='disk')
    plt.plot(r_normalized, vel_h_Burket(np.array(r) * 1000, bestfit_av.x[4], bestfit_av.x[5] * 1000), label='Burket halo')
    plt.legend()
    plt.xlabel('$r_{dep}$/$R_d$')
    plt.ylabel('$v_{rot}$ [km/s]')
    plt.title(galaxy_ID)
    plt.show()
    # -------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------
    # Positive rotation curve
    # -------------------------------------------------------------------------------
    # Initial guesses

    bestfit_mav = minimize(nloglike_Bur, p0, args=(r_mav, mav, mav_err),
                       bounds=param_bounds)
    print('---------------------------------------------------')
    print('Positive curve')
    print(bestfit_mav)
    bestfits_mav.append(bestfit_mav)

    r_normalized_mav = r_mav / bestfit_mav.x[1]

    plt.errorbar(r_normalized_mav, mav, yerr=mav_err, fmt='r*', label='data')
    plt.plot(r_normalized_mav, v_co_Burket(np.array(r), bestfit_mav.x), '--', label='fit')
    plt.plot(r_normalized_mav, vel_b(np.array(r) * 1000, bestfit_mav.x[0], bestfit_mav.x[1], bestfit_mav.x[3] * 1000),
         label='bulge')
    plt.plot(r_normalized_mav, v_d(np.array(r) * 1000, 10 ** (bestfit_mav.x[2]), bestfit_mav.x[3] * 1000), label='disk')
    plt.plot(r_normalized_mav, vel_h_Burket(np.array(r) * 1000, bestfit_mav.x[4], bestfit_mav.x[5] * 1000),
         label='Burket halo')
    plt.legend()
    plt.xlabel('$r_{dep}$/$R_d$')
    plt.ylabel('$v_{rot}$ [km/s]')
    plt.title(galaxy_ID)
    plt.show()
    # -------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------
    # Negative rotation curve
    # -------------------------------------------------------------------------------
    bestfit_miv = minimize(nloglike_Bur, p0, args=(r_miv, np.abs(miv), miv_err),
                       bounds=param_bounds)
    print('---------------------------------------------------')
    print('Negative curve')
    print(bestfit_miv)
    bestfits_av.append(bestfit_av)

    r_normalized_miv = r_miv / bestfit_miv.x[1]

    plt.errorbar(r_normalized_miv, np.abs(miv), yerr=miv_err, fmt='b*', label='data')
    plt.plot(r_normalized_miv, v_co_Burket(np.array(r), bestfit_miv.x), '--', label='fit')
    plt.plot(r_normalized_miv, vel_b(np.array(r) * 1000, bestfit_miv.x[0], bestfit_miv.x[1], bestfit_miv.x[3] * 1000),
         label='bulge')
    plt.plot(r_normalized_miv, v_d(np.array(r) * 1000, 10 ** bestfit_miv.x[2], bestfit_miv.x[3] * 1000), label='disk')
    plt.plot(r_normalized_miv, vel_h_Burket(np.array(r) * 1000, bestfit_miv.x[4], bestfit_miv.x[5] * 1000),
         label='Burket halo')
    plt.legend()
    plt.xlabel('$r_{dep}$/$R_d$')
    plt.ylabel('$v_{rot}$ [km/s]')
    plt.title(galaxy_ID)
    plt.show()
################################################################################
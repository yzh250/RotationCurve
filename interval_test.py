################################################################################
# All the libraries used & constant values
# -------------------------------------------------------------------------------
import numpy as np

import matplotlib.pyplot as plt

from astropy.table import QTable

from astropy.io import ascii

from scipy.optimize import minimize

from rotation_curve_functions import nloglike_Bur, v_co_Burket, v_co_Burket_nb,v_d, vel_h_Burket, vel_b, nloglike_Bur_nb

import math

import statistics
################################################################################

################################################################################
# Physical Constants
c = 3E5 # k * m * s ^1
h = 1 # reduced hubble constant
H_0 =  100 * h # km * s^-1 * Mpc^-1
################################################################################

################################################################################
# Import the Master Table

DTable1 = QTable.read('Master_Table.txt',format='ascii.commented_header')
DTable2 = QTable.read('DRPall-master_file.txt',format='ascii.ecsv')

# Get the Mass of stars & redshifts & angular resolution of r50
m = DTable1['NSA_Mstar'].data
z = DTable2['redshift'].data
r50_ang = DTable2['nsa_elpetro_th50_r'].data

# Obtain r50 for plate IFU 7443-12705
r50_spec = 0
z_spec = 0
for i in range(len(DTable2)):
    if DTable2['MaNGA_plate'][i] == 7443 and DTable2['MaNGA_IFU'][i] == 12705:
        r50_spec = r50_ang[i]
        z_spec = z[i]

# Obtain stellar mass of 7443-12705
m_spec = 0
for i in range(len(DTable1)):
    if DTable1['MaNGA_plate'][i] == 7443 and DTable1['MaNGA_IFU'][i] == 12705:
        m_spec = m[i]
print(np.log10(m_spec))

#  Calculate the recession velocity for 7443-12705
v_rec = z_spec * c # km/s

# Using Hubble's Law to calculate distance [kpc] for 7443-12705
d = v_rec/H_0 # Mpc
d *= 1E3 # kpc

# Using Small Angle Formula to calculate the actual value of r50 [kpc] for 7443-12705
theta = r50_spec/206265 # radian
r50 = theta * d # kpc
print(r50)
################################################################################

################################################################################
# Import 7443-12705 data
galaxy_ID = '7443-12705'

rot_curve_data_filename = galaxy_ID + '_rot_curve_data.txt'

DTable_spec = QTable.read(rot_curve_data_filename, format='ascii.ecsv')

r = DTable_spec['deprojected_distance'].data
print(max(np.array(r)))
av = DTable_spec['rot_vel_avg'].data
av_err = DTable_spec['rot_vel_avg_error'].data
################################################################################

################################################################################
# Bounds
param_bounds = [[np.log10(m_spec), 13],  # Disk mass [log(Msun)]
                [0, 20],  # Disk radius [kpc]
                [0, 0.01],  # Halo density [Msun/pc^2]
                [0, 100]]  # Halo radius [kpc]
################################################################################

################################################################################
# Interval Calculations (1 variable) --> Disk Radius

logM = np.linspace(np.log10(m_spec),13,5)
r_d = np.linspace(0.1*r50,2*r50,10) # initial guesses of r_d in terms of r50
rho_dc = np.linspace(0.0001,0.01,20)
r_h = np.linspace(0.1*r50,5*r50,10) # initial guesses of r_h in terms of r50

# Good fitting parameter
logM_good = []
logM_good_initial = []
rdisk_good = []
rdisk_good_initial = []
rho_good = []
rho_good_initial = []
rhalo_good = []
rhalo_good_initial = []


# Variables for bounds
logM_max = max(param_bounds[0])
logM_min = min(param_bounds[0])
rdisk_max = max(param_bounds[1])
rdisk_min = min(param_bounds[1])
rho_max = max(param_bounds[2])
rho_min = min(param_bounds[2])
rhalo_max = max(param_bounds[3])
rhalo_min = min(param_bounds[3])

for i in range(len(logM)):
    for j in range(len(r_d)):
        for k in range(len(rho_dc)):
            for l in range(len(r_h)):
                # Fit
                p0 = [logM[i], r_d[j], rho_dc[k], r_h[l]]
                bestfit_av = minimize(nloglike_Bur_nb, p0, args=(r, av, av_err, 250),
                                      bounds=param_bounds)

                print('---------------------------------------------------')
                print(bestfit_av)
                # Plotting Average
                if max(list(r)) < bestfit_av.x[1]:
                    r_plot = np.linspace(0, 3 * bestfit_av.x[1], 100)
                else:
                    r_plot = np.linspace(0, 3 * max(list(r)), 100)
                    plt.figure()
                    plt.errorbar(r, av, yerr=av_err, fmt='g*', label='data')
                    plt.plot(r_plot, v_co_Burket_nb(np.array(r_plot), bestfit_av.x), '--', label='fit')
                    plt.plot(r_plot, v_d(np.array(r_plot) * 1000, 10 ** bestfit_av.x[0], bestfit_av.x[1] * 1000),
                             color='orange',
                             label='disk')
                    plt.plot(r_plot, vel_h_Burket(np.array(r_plot) * 1000, bestfit_av.x[2], bestfit_av.x[3] * 1000),
                             color='blue',
                             label='Burket halo')
                    plt.legend()
                    plt.xlabel('$r_{dep}$ [kpc]')
                    plt.ylabel('$v_{rot}$ [km/s]')
                    plt.title('7443-12705')
                    plt.savefig('7443-12705'+str(i)+str(j)+str(k)+str(l)+'.png',format='png')
                    plt.close()

                # Determining if we have a good fit
                good_fit_flag = True
                if  (bestfit_av.x[0] == logM[i]) or (bestfit_av.x[0] == logM_min) or (bestfit_av.x[0] == logM_max):
                    good_fit_flag = False
                if (bestfit_av.x[1] == r_d[j]) or (bestfit_av.x[1] == rdisk_min) or (bestfit_av.x[1] == rdisk_max):
                    good_fit_flag = False
                if (bestfit_av.x[2] == rho_dc[k]) or (bestfit_av.x[2] == rho_min) or (bestfit_av.x[2] == rho_max):
                    good_fit_flag = False
                if (bestfit_av.x[3] == r_h[l]) or (bestfit_av.x[3] == rhalo_min) or (bestfit_av.x[3] == rhalo_max):
                    good_fit_flag = False
                if good_fit_flag:
                    logM_good.append(bestfit_av.x[0])
                    logM_good_initial.append(logM[i])
                    rdisk_good.append(bestfit_av.x[1])
                    rdisk_good_initial.append(r_d[j])
                    rho_good.append(bestfit_av.x[2])
                    rho_good_initial.append(rho_dc[k])
                    rhalo_good.append(bestfit_av.x[3])
                    rhalo_good_initial.append(r_h[l])

print('---------------------------------------------------')
print('$log_{10}(M_{disk})$ fits:')
print(logM_good)
print("length: " + str(len(logM_good)))
print('$log_{10}(M_{disk})$ good initial guesses:')
print(logM_good_initial)
print("length: " + str(len(logM_good_initial)))
print('---------------------------------------------------')

print('---------------------------------------------------')
print('$r_{disk}$ fits:')
print(rdisk_good)
print("length: " + str(len(rdisk_good)))
print('$r_{disk}$ good initial guesses:')
print(rdisk_good_initial)
print("length: " + str(len(rdisk_good_initial)))
print('---------------------------------------------------')

print('---------------------------------------------------')
print('$\\rho_{dc}$ fits:')
print(rho_good)
print("length: " + str(len(rho_good)))
print('$\\rho_{dc}$ good initial guesses:')
print(rho_good_initial)
print("length: " + str(len(rho_good_initial)))
print('---------------------------------------------------')

print('---------------------------------------------------')
print('$r_{halo}$ fits:')
print(rhalo_good)
print("length: " + str(len(rhalo_good)))
print('$r_{halo}$ good initial guesses:')
print(rho_good_initial)
print("length: " + str(len(rhalo_good_initial)))
print('---------------------------------------------------')

Good_fit_Table = QTable([logM_good, logM_good_initial, rdisk_good, rdisk_good_initial, rho_good, rho_good_initial, rhalo_good, rhalo_good_initial],
            names=('logM (log(M_sol))', 'logM_initial (log(M_sol))', 'R_disk (kpc)', 'R_disk_initial (kpc)','Rho_dc (M_sol/pc^3)','Rho_dc_initial (M_sol/pc^3)','R_halo (kpc)','R_halo_initial (kpc)'))
print(Good_fit_Table)

ascii.write([logM_good, logM_good_initial, rdisk_good, rdisk_good_initial, rho_good, rho_good_initial, rhalo_good, rhalo_good_initial],'good_fit.txt',format='ecsv',names=('logM (log(M_sol))', 'logM_initial (log(M_sol))', 'R_disk (kpc)', 'R_disk_initial (kpc)','Rho_dc (M_sol/pc^3)','Rho_dc_initial (M_sol/pc^3)','R_halo (kpc)','R_halo_initial (kpcx)'),overwrite=True)

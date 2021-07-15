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
# Only 2 parameters

logM = np.log10(m_spec) + 0.8
r_disk = 0.8*r50_spec
rho_dc =[0.006453502415458937,0.008778381642512078,0.00977596618357488]
r_halo = [27.0314253647587,23.32767286390475,34.56575415995706]

for i in range(len(rho_dc)):
    if max(list(r)) < r_disk:
        r_plot = np.linspace(0, 3 * r_disk, 100)
    else:
        r_plot = np.linspace(0, 3 * max(list(r)), 100)
        plt.figure()
        plt.errorbar(r, av, yerr=av_err, fmt='g*', label='data')
        plt.plot(r_plot, v_co_Burket_nb(np.array(r_plot), [logM,r_disk,rho_dc[i],r_halo[i]]), '--', label='fit')
        plt.plot(r_plot, v_d(np.array(r_plot) * 1000, 10 ** logM, r_disk * 1000),
                    color='orange',
                    label='disk')
        plt.plot(r_plot, vel_h_Burket(np.array(r_plot) * 1000, rho_dc[i], r_halo[i] * 1000),
                    color='blue',
                    label='Burket halo')
        plt.legend()
        plt.xlabel('$r_{dep}$ [kpc]')
        plt.ylabel('$v_{rot}$ [km/s]')
        plt.title('7443-12705')
        plt.show()
        #plt.savefig('7443-12705' + str(i) + str(j) + str(k) + str(l) + '.png', format='png')
        #plt.close()

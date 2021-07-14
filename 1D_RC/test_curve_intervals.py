################################################################################
# All the libraries used & constant values
# -------------------------------------------------------------------------------
import numpy as np

import matplotlib.pyplot as plt

from astropy.table import QTable

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
print(r50_spec)
################################################################################

################################################################################
# Import 7443-12705 data
galaxy_ID = '7443-12705'

rot_curve_data_filename = galaxy_ID + '_rot_curve_data.txt'

DTable_spec = QTable.read(rot_curve_data_filename, format='ascii.ecsv')

r = DTable_spec['deprojected_distance'].data
print(max(np.array(r)))
av = DTable_spec['rot_vel_avg'].data
mav = DTable_spec['max_velocity'].data
miv = DTable_spec['min_velocity'].data
av_err = DTable_spec['rot_vel_avg_error'].data
mav_err = DTable_spec['max_velocity_error'].data
miv_err = DTable_spec['min_velocity_error'].data
################################################################################


################################################################################
# Bounds
param_bounds = [[7, 12],  # Disk mass [log(Msun)]
                [0, 10],  # Disk radius [kpc]
                [0, 0.1],  # Halo density [Msun/pc^2]
                [0, 100]]  # Halo radius [kpc]
################################################################################


################################################################################
# Interval Calculations (1 variable) --> Disk Radius
# Initial Guesses [logM, rho_c, r_halo]
logM_guess = np.log10(m_spec) + 0.65
rho_dc_guess = 0.008
r_h_guess = max(list(r))*1.75

r_d_av_list = -1 * np.ones(20)
r_d_mav_list = -1 * np.ones(20)
r_d_miv_list = -1 * np.ones(20)
fit_av = -1 * np.ones(20)
fit_mav = -1 * np.ones(20)
fit_miv = -1 * np.ones(20)


# For disk radii
r_d = np.linspace(0.1*r50,2*r50,20) # Change this to intervals in terms of r50

for i in range(len(r_d)):
    p0 = [logM_guess,r_d[i],rho_dc_guess,r_h_guess]
    # Best fits for all three velocities
    bestfit_av = minimize(nloglike_Bur_nb, p0, args=(r, av, av_err, 250),
                       bounds=param_bounds)
    print('---------------------------------------------------')
    print(bestfit_av)
    r_d_av_list[i] = bestfit_av.x[1]/r50
    fit_av[i] = bestfit_av.fun

    bestfit_mav = minimize(nloglike_Bur_nb, p0, args=(r, mav, mav_err, 250),
                          bounds=param_bounds)
    print('---------------------------------------------------')
    print(bestfit_mav)
    r_d_mav_list[i] = bestfit_av.x[1]/r50
    fit_mav[i] = bestfit_mav.fun

    bestfit_miv = minimize(nloglike_Bur_nb, p0, args=(r, np.abs(miv), miv_err, 250),
                          bounds=param_bounds)
    print('---------------------------------------------------')
    print(bestfit_miv)
    r_d_miv_list[i] = bestfit_av.x[1]/r50
    fit_miv[i] = bestfit_miv.fun
    # Plotting Average
    if max(list(r)) < bestfit_av.x[1]:
        r_plot = np.linspace(0, 3 * bestfit_av.x[1], 100)
    else:
        r_plot = np.linspace(0, 3 * max(list(r)), 100)
    plt.errorbar(r, av, yerr=av_err, fmt='g*', label='data')
    plt.plot(r_plot, v_co_Burket_nb(np.array(r_plot), bestfit_av.x), '--', label='fit')
    plt.plot(r_plot, v_d(np.array(r_plot) * 1000, 10 ** bestfit_av.x[0], bestfit_av.x[1] * 1000), color='orange',
             label='disk')
    plt.plot(r_plot, vel_h_Burket(np.array(r_plot) * 1000, bestfit_av.x[2], bestfit_av.x[3] * 1000), color='blue',
             label='Burket halo')
    plt.legend()
    plt.xlabel('$r_{dep}$ [kpc]')
    plt.ylabel('$v_{rot}$ [km/s]')
    plt.title('7443-12705')
    plt.show()

    # Plotting Positive
    if max(list(r)) < bestfit_mav.x[1]:
        r_plot = np.linspace(0, 3 * bestfit_mav.x[1], 100)
    else:
        r_plot = np.linspace(0, 3 * max(list(r)), 100)
    plt.errorbar(r, mav, yerr=mav_err, fmt='g*', label='data')
    plt.plot(r_plot, v_co_Burket_nb(np.array(r_plot), bestfit_mav.x), '--', label='fit')
    plt.plot(r_plot, v_d(np.array(r_plot) * 1000, 10 ** bestfit_mav.x[0], bestfit_mav.x[1] * 1000), color='orange',
             label='disk')
    plt.plot(r_plot, vel_h_Burket(np.array(r_plot) * 1000, bestfit_mav.x[2], bestfit_mav.x[3] * 1000), color='blue',
             label='Burket halo')
    plt.legend()
    plt.xlabel('$r_{dep}$ [kpc]')
    plt.ylabel('$v_{rot}$ [km/s]')
    plt.title('7443-12705')
    plt.show()

    # Plotting Negative
    if max(list(r)) < bestfit_miv.x[1]:
        r_plot = np.linspace(0, 3 * bestfit_miv.x[1], 100)
    else:
        r_plot = np.linspace(0, 3 * max(list(r)), 100)
    plt.errorbar(r, np.abs(miv), yerr=miv_err, fmt='g*', label='data')
    plt.plot(r_plot, v_co_Burket_nb(np.array(r_plot), bestfit_miv.x), '--', label='fit')
    plt.plot(r_plot, v_d(np.array(r_plot) * 1000, 10 ** bestfit_miv.x[0], bestfit_miv.x[1] * 1000), color='orange',
             label='disk')
    plt.plot(r_plot, vel_h_Burket(np.array(r_plot) * 1000, bestfit_miv.x[2], bestfit_miv.x[3] * 1000), color='blue',
             label='Burket halo')
    plt.legend()
    plt.xlabel('$r_{dep}$ [kpc]')
    plt.ylabel('$v_{rot}$ [km/s]')
    plt.title('7443-12705')
    plt.show()

print(r_d_av_list)
print(r_d_mav_list)
print(r_d_miv_list)
print(fit_av)
print(fit_mav)
print(fit_miv)

plt.plot(r_d/r50,r_d_av_list,'b*')
plt.plot(np.linspace(0,2,20),np.linspace(0,2,20))
plt.plot(np.linspace(0,2,20),np.linspace(0,0,20)/r50,label='lower bound')
plt.plot(np.linspace(0,2,20),np.linspace(10,10,20)/r50,label='upper bound')
plt.xlabel('initial guess in $r_{50}$ [kpc]')
plt.ylabel('fitted parameters in $r_{50}$ [km/s]')
plt.title('$R_{disk}$ Average')
plt.legend()
plt.show()
plt.plot(r_d/r50,r_d_mav_list,'g*')
plt.plot(np.linspace(0,2,20),np.linspace(0,2,20))
plt.plot(np.linspace(0,2,20),np.linspace(0,0,20)/r50,label='lower bound')
plt.plot(np.linspace(0,2,20),np.linspace(10,10,20)/r50,label='upper bound')
plt.xlabel('initial guess in $r_{50}$ [kpc]')
plt.ylabel('fitted parameters in $r_{50}$ [kpc]')
plt.title('$R_{disk}$ Positive')
plt.legend()
plt.show()
plt.plot(r_d/r50,r_d_miv_list,'r*')
plt.plot(np.linspace(0,2,20),np.linspace(0,2,20))
plt.plot(np.linspace(0,2,20),np.linspace(0,0,20)/r50,label='lower bound')
plt.plot(np.linspace(0,2,20),np.linspace(10,10,20)/r50,label='upper bound')
plt.xlabel('initial guess in $r_{50}$ [kpc]')
plt.ylabel('fitted parameters in $r_{50}$ [kpc]')
plt.title('$R_{disk}$ Negative')
plt.legend()
plt.show()

r_h_av_list = -1 * np.ones(20)
r_h_mav_list = -1 * np.ones(20)
r_h_miv_list = -1 * np.ones(20)


# Interval Test with r_halo
r_h = np.linspace(0.001*r50,20*r50,20)
for i in range(len(r_h)):
    p0 = [logM_guess,0.68*r50,rho_dc_guess,r_h[i]]
    # Best fits for all three velocities
    bestfit_av = minimize(nloglike_Bur_nb, p0, args=(r, av, av_err, 250),
                       bounds=param_bounds)
    print('---------------------------------------------------')
    print(bestfit_av)
    r_h_av_list[i] = bestfit_av.x[3] / r50

    p0 = [logM_guess, 0.68*r50, rho_dc_guess, r_h[i]]
    bestfit_mav = minimize(nloglike_Bur_nb, p0, args=(r, mav, mav_err, 250),
                          bounds=param_bounds)
    print('---------------------------------------------------')
    print(bestfit_mav)
    r_h_mav_list[i] = bestfit_mav.x[3] / r50

    p0 = [logM_guess, 0.68*r50, rho_dc_guess, r_h[i]]
    bestfit_miv = minimize(nloglike_Bur_nb, p0, args=(r, np.abs(miv), miv_err, 250),
                          bounds=param_bounds)
    print('---------------------------------------------------')
    print(bestfit_miv)
    r_h_miv_list[i] = bestfit_miv.x[3] / r50

    # Plotting Average
    if max(list(r)) < bestfit_av.x[1]:
        r_plot = np.linspace(0, 3 * bestfit_av.x[1], 100)
    else:
        r_plot = np.linspace(0, 3 * max(list(r)), 100)
    plt.errorbar(r, av, yerr=av_err, fmt='g*', label='data')
    plt.plot(r_plot, v_co_Burket_nb(np.array(r_plot), bestfit_av.x), '--', label='fit')
    plt.plot(r_plot, v_d(np.array(r_plot) * 1000, 10 ** bestfit_av.x[0], bestfit_av.x[1] * 1000), color='orange',
             label='disk')
    plt.plot(r_plot, vel_h_Burket(np.array(r_plot) * 1000, bestfit_av.x[2], bestfit_av.x[3] * 1000), color='blue',
             label='Burket halo')
    plt.legend()
    plt.xlabel('$r_{dep}$ [kpc]')
    plt.ylabel('$v_{rot}$ [km/s]')
    plt.title('7443-12705')
    plt.show()

    # Plotting Positive
    if max(list(r)) < bestfit_mav.x[1]:
        r_plot = np.linspace(0, 3 * bestfit_mav.x[1], 100)
    else:
        r_plot = np.linspace(0, 3 * max(list(r)), 100)
    plt.errorbar(r, mav, yerr=mav_err, fmt='g*', label='data')
    plt.plot(r_plot, v_co_Burket_nb(np.array(r_plot), bestfit_mav.x), '--', label='fit')
    plt.plot(r_plot, v_d(np.array(r_plot) * 1000, 10 ** bestfit_mav.x[0], bestfit_mav.x[1] * 1000), color='orange',
            label='disk')
    plt.plot(r_plot, vel_h_Burket(np.array(r_plot) * 1000, bestfit_mav.x[2], bestfit_mav.x[3] * 1000), color='blue',
            label='Burket halo')
    plt.legend()
    plt.xlabel('$r_{dep}$ [kpc]')
    plt.ylabel('$v_{rot}$ [km/s]')
    plt.title('7443-12705')
    plt.show()

    # Plotting Negative
    if max(list(r)) < bestfit_miv.x[1]:
        r_plot = np.linspace(0, 3 * bestfit_miv.x[1], 100)
    else:
        r_plot = np.linspace(0, 3 * max(list(r)), 100)
    plt.errorbar(r, np.abs(miv), yerr=miv_err, fmt='g*', label='data')
    plt.plot(r_plot, v_co_Burket_nb(np.array(r_plot), bestfit_miv.x), '--', label='fit')
    plt.plot(r_plot, v_d(np.array(r_plot) * 1000, 10 ** bestfit_miv.x[0], bestfit_miv.x[1] * 1000), color='orange',
             label='disk')
    plt.plot(r_plot, vel_h_Burket(np.array(r_plot) * 1000, bestfit_miv.x[2], bestfit_miv.x[3] * 1000), color='blue',
             label='Burket halo')
    plt.legend()
    plt.xlabel('$r_{dep}$ [kpc]')
    plt.ylabel('$v_{rot}$ [km/s]')
    plt.title('7443-12705')
    plt.show()
print(r_h_av_list)
print(r_h_mav_list)
print(r_h_miv_list)

plt.plot(r_h/r50,r_h_av_list,'b*')
plt.plot(np.linspace(0,20,20),np.linspace(0,20,20))
plt.plot(np.linspace(0,20,20),np.linspace(0,0,20)/r50,label='lower bound')
plt.plot(np.linspace(0,20,20),np.linspace(100,100,20)/r50,label='upper bound')
plt.xlabel('initial guess in $r_{50}$ [kpc]')
plt.ylabel('fitted parameters in $r_{50}$ [km/s]')
plt.title('$R_{halo}$ Average')
plt.legend()
plt.show()
plt.plot(r_h/r50,r_h_mav_list,'g*')
plt.plot(np.linspace(0,20,20),np.linspace(0,20,20))
plt.plot(np.linspace(0,20,20),np.linspace(0,0,20)/r50,label='lower bound')
plt.plot(np.linspace(0,20,20),np.linspace(100,100,20)/r50,label='upper bound')
plt.xlabel('initial guess in $r_{50}$ [kpc]')
plt.ylabel('fitted parameters in $r_{50}$ [kpc]')
plt.title('$R_{halo}$ Positive')
plt.legend()
plt.show()
plt.plot(r_h/r50,r_h_miv_list,'r*')
plt.plot(np.linspace(0,20,20),np.linspace(0,20,20))
plt.plot(np.linspace(0,20,20),np.linspace(0,0,20)/r50,label='lower bound')
plt.plot(np.linspace(0,20,20),np.linspace(100,100,20)/r50,label='upper bound')
plt.xlabel('initial guess in $r_{50}$ [kpc]')
plt.ylabel('fitted parameters in $r_{50}$ [kpc]')
plt.title('$R_{halo}$ Negative')
plt.legend()
plt.show()

logM_av_list = -1 * np.ones(20)
logM_mav_list = -1 * np.ones(20)
logM_miv_list = -1 * np.ones(20)

# Intervals Test with logM
logM = np.linspace(np.log10(m_spec),np.log10(m_spec)+1,20)
for i in range(len(logM)):
    p0 = [logM[i],0.68*r50,rho_dc_guess,2.1*r50]
    # Best fits for all three velocities
    bestfit_av = minimize(nloglike_Bur_nb, p0, args=(r, av, av_err, 250),
                       bounds=param_bounds)
    print('---------------------------------------------------')
    print(bestfit_av)
    logM_av_list[i] = bestfit_av.x[0] - np.log10(m_spec)

    p0 = [logM[i], 0.68*r50, rho_dc_guess, 2.1*r50]
    bestfit_mav = minimize(nloglike_Bur_nb, p0, args=(r, mav, mav_err, 250),
                          bounds=param_bounds)
    print('---------------------------------------------------')
    print(bestfit_mav)
    logM_mav_list[i] = bestfit_mav.x[0] - np.log10(m_spec)

    p0 = [logM[i], 0.68*r50, rho_dc_guess, 2.1*r50]
    bestfit_miv = minimize(nloglike_Bur_nb, p0, args=(r, np.abs(miv), miv_err, 250),
                          bounds=param_bounds)
    print('---------------------------------------------------')
    print(bestfit_miv)
    logM_miv_list[i] = bestfit_miv.x[0] - np.log10(m_spec)

    # Plotting Average
    if max(list(r)) < bestfit_av.x[1]:
        r_plot = np.linspace(0, 3 * bestfit_av.x[1], 100)
    else:
        r_plot = np.linspace(0, 3 * max(list(r)), 100)
    plt.errorbar(r, av, yerr=av_err, fmt='g*', label='data')
    plt.plot(r_plot, v_co_Burket_nb(np.array(r_plot), bestfit_av.x), '--', label='fit')
    plt.plot(r_plot, v_d(np.array(r_plot) * 1000, 10 ** bestfit_av.x[0], bestfit_av.x[1] * 1000), color='orange',
             label='disk')
    plt.plot(r_plot, vel_h_Burket(np.array(r_plot) * 1000, bestfit_av.x[2], bestfit_av.x[3] * 1000), color='blue',
             label='Burket halo')
    plt.legend()
    plt.xlabel('$r_{dep}$ [kpc]')
    plt.ylabel('$v_{rot}$ [km/s]')
    plt.title('7443-12705')
    plt.show()
    
    # Plotting Positive
    if max(list(r)) < bestfit_mav.x[1]:
        r_plot = np.linspace(0, 3 * bestfit_mav.x[1], 100)
    else:
        r_plot = np.linspace(0, 3 * max(list(r)), 100)
    plt.errorbar(r, mav, yerr=mav_err, fmt='g*', label='data')
    plt.plot(r_plot, v_co_Burket_nb(np.array(r_plot), bestfit_mav.x), '--', label='fit')
    plt.plot(r_plot, v_d(np.array(r_plot) * 1000, 10 ** bestfit_mav.x[0], bestfit_mav.x[1] * 1000), color='orange',
             label='disk')
    plt.plot(r_plot, vel_h_Burket(np.array(r_plot) * 1000, bestfit_mav.x[2], bestfit_mav.x[3] * 1000), color='blue',
              label='Burket halo')
    plt.legend()
    plt.xlabel('$r_{dep}$ [kpc]')
    plt.ylabel('$v_{rot}$ [km/s]')
    plt.title('7443-12705')
    plt.show()

    # Plotting Negative
    if max(list(r)) < bestfit_miv.x[1]:
        r_plot = np.linspace(0, 3 * bestfit_miv.x[1], 100)
    else:
        r_plot = np.linspace(0, 3 * max(list(r)), 100)
    plt.errorbar(r, np.abs(miv), yerr=miv_err, fmt='g*', label='data')
    plt.plot(r_plot, v_co_Burket_nb(np.array(r_plot), bestfit_miv.x), '--', label='fit')
    plt.plot(r_plot, v_d(np.array(r_plot) * 1000, 10 ** bestfit_miv.x[0], bestfit_miv.x[1] * 1000), color='orange',
             label='disk')
    plt.plot(r_plot, vel_h_Burket(np.array(r_plot) * 1000, bestfit_miv.x[2], bestfit_miv.x[3] * 1000), color='blue',
            label='Burket halo')
    plt.legend()
    plt.xlabel('$r_{dep}$ [kpc]')
    plt.ylabel('$v_{rot}$ [km/s]')
    plt.title('7443-12705')
    plt.show()

print(logM_av_list)
print(logM_mav_list)
print(logM_miv_list)

for i in range(len(logM)):
    logM[i] -= np.log10(m_spec)

# Upper bound & Lower Bound
lb = np.linspace(7,7,20)
ub = np.linspace(12,12,20)
for i in range(len(lb)):
    lb[i] -= np.log10(m_spec)
for i in range(len(ub)):
    ub[i] -= np.log10(m_spec)

plt.plot(logM,logM_av_list,'b*')
plt.plot(np.linspace(0,1,20),np.linspace(0,1,20))
plt.plot(np.linspace(0,1,20),lb,label='lower bound')
plt.plot(np.linspace(0,1,20),ub,label='upper bound')
plt.xlabel('initial guess - [log(NSA_Mstar)]$')
plt.ylabel('fitted parameters - [log(NSA_Mstar)]')
plt.title('Average')
plt.legend()
plt.show()
plt.plot(logM,logM_mav_list,'g*')
plt.plot(np.linspace(0,1,20),np.linspace(0,1,20))
plt.plot(np.linspace(0,1,20),lb,label='lower bound')
plt.plot(np.linspace(0,1,20),ub,label='upper bound')
plt.xlabel('initial guess - [log(NSA_Mstar)]')
plt.ylabel('fitted parameters - [log(NSA_Mstar)] ')
plt.title('Positive')
plt.legend()
plt.show()
plt.plot(logM,logM_miv_list,'r*')
plt.plot(np.linspace(0,1,20),np.linspace(0,1,20))
plt.plot(np.linspace(0,1,20),lb,label='lower bound')
plt.plot(np.linspace(0,1,20),ub,label='upper bound')
plt.xlabel('initial guess - [log(NSA_Mstar)]')
plt.ylabel('fitted parameters - [log(NSA_Mstar)]')
plt.title('Negative')
plt.legend()
plt.show()

rho_dc_av_list = -1 * np.ones(20)
rho_dc_mav_list = -1 * np.ones(20)
rho_dc_miv_list = -1 * np.ones(20)


# Intervals Test with central density
rho_dc = np.linspace(0.0001,0.01,20)
for i in range(len(rho_dc)):
    p0 = [logM_guess,0.68*r50,rho_dc[i],2.1*r50]
    # Best fits for all three velocities
    bestfit_av = minimize(nloglike_Bur_nb, p0, args=(r, av, av_err, 250),
                       bounds=param_bounds)
    print('---------------------------------------------------')
    print(bestfit_av)
    rho_dc_av_list[i] = bestfit_av.x[2]

    p0 = [logM_guess,0.68*r50, rho_dc[i], 2.1*r50]
    bestfit_mav = minimize(nloglike_Bur_nb, p0, args=(r, mav, mav_err, 250),
                          bounds=param_bounds)
    print('---------------------------------------------------')
    print(bestfit_mav)
    rho_dc_mav_list[i] = bestfit_mav.x[2]

    p0 = [logM_guess,0.68*r50, rho_dc[i], 2.1*r50]
    bestfit_miv = minimize(nloglike_Bur_nb, p0, args=(r, np.abs(miv), miv_err, 250),
                          bounds=param_bounds)
    print('---------------------------------------------------')
    print(bestfit_miv)
    rho_dc_miv_list[i] = bestfit_miv.x[2]

    # Plotting Average
    if max(list(r)) < bestfit_av.x[1]:
        r_plot = np.linspace(0, 3 * bestfit_av.x[1], 100)
    else:
        r_plot = np.linspace(0, 3 * max(list(r)), 100)
    plt.errorbar(r, av, yerr=av_err, fmt='g*', label='data')
    plt.plot(r_plot, v_co_Burket_nb(np.array(r_plot), bestfit_av.x), '--', label='fit')
    plt.plot(r_plot, v_d(np.array(r_plot) * 1000, 10 ** bestfit_av.x[0], bestfit_av.x[1] * 1000), color='orange',
             label='disk')
    plt.plot(r_plot, vel_h_Burket(np.array(r_plot) * 1000, bestfit_av.x[2], bestfit_av.x[3] * 1000), color='blue',
             label='Burket halo')
    plt.legend()
    plt.xlabel('$r_{dep}$ [kpc]')
    plt.ylabel('$v_{rot}$ [km/s]')
    plt.title('7443-12705')
    plt.show()

    # Plotting Positive
    if max(list(r)) < bestfit_mav.x[1]:
        r_plot = np.linspace(0, 3 * bestfit_mav.x[1], 100)
    else:
        r_plot = np.linspace(0, 3 * max(list(r)), 100)
    plt.errorbar(r, mav, yerr=mav_err, fmt='g*', label='data')
    plt.plot(r_plot, v_co_Burket_nb(np.array(r_plot), bestfit_mav.x), '--', label='fit')
    plt.plot(r_plot, v_d(np.array(r_plot) * 1000, 10 ** bestfit_mav.x[0], bestfit_mav.x[1] * 1000), color='orange',
             label='disk')
    plt.plot(r_plot, vel_h_Burket(np.array(r_plot) * 1000, bestfit_mav.x[2], bestfit_mav.x[3] * 1000), color='blue',
              label='Burket halo')
    plt.legend()
    plt.xlabel('$r_{dep}$ [kpc]')
    plt.ylabel('$v_{rot}$ [km/s]')
    plt.title('7443-12705')
    plt.show()

    # Plotting Negative
    if max(list(r)) < bestfit_miv.x[1]:
        r_plot = np.linspace(0, 3 * bestfit_miv.x[1], 100)
    else:
        r_plot = np.linspace(0, 3 * max(list(r)), 100)
    plt.errorbar(r, np.abs(miv), yerr=miv_err, fmt='g*', label='data')
    plt.plot(r_plot, v_co_Burket_nb(np.array(r_plot), bestfit_miv.x), '--', label='fit')
    plt.plot(r_plot, v_d(np.array(r_plot) * 1000, 10 ** bestfit_miv.x[0], bestfit_miv.x[1] * 1000), color='orange',
             label='disk')
    plt.plot(r_plot, vel_h_Burket(np.array(r_plot) * 1000, bestfit_miv.x[2], bestfit_miv.x[3] * 1000), color='blue',
            label='Burket halo')
    plt.legend()
    plt.xlabel('$r_{dep}$ [kpc]')
    plt.ylabel('$v_{rot}$ [km/s]')
    plt.title('7443-12705')
    plt.show()

print(rho_dc_av_list)
print(rho_dc_mav_list)
print(rho_dc_miv_list)


plt.plot(rho_dc,rho_dc_av_list,'b*')
plt.plot(np.linspace(0,0.01,20),np.linspace(0,0.01,20))
plt.plot(np.linspace(0,0.01,20),np.linspace(0,0,20),label='lower bound')
plt.plot(np.linspace(0,0.01,20),np.linspace(.1,.1,20),label='lower bound')
plt.xlabel('initial guess [$\\frac{M_{\\odot}}{pc^3}$]')
plt.ylabel('fitted parameters [$\\frac{M_{\\odot}}{pc^3}$]')
plt.title('Average')
plt.legend()
plt.show()
plt.plot(rho_dc,rho_dc_mav_list,'g*')
plt.plot(np.linspace(0,0.01,20),np.linspace(0,0.01,20))
plt.plot(np.linspace(0,0.01,20),np.linspace(0,0,20),label='lower bound')
plt.plot(np.linspace(0,0.01,20),np.linspace(.1,.1,20),label='lower bound')
plt.xlabel('initial guess [$\\frac{M_{\\odot}}{pc^3}$]')
plt.ylabel('fitted parameters [$\\frac{M_{\\odot}}{pc^3}$] ')
plt.title('Positive')
plt.show()
plt.plot(rho_dc,rho_dc_miv_list,'r*')
plt.plot(np.linspace(0,0.01,20),np.linspace(0,0.01,20))
plt.plot(np.linspace(0,0.01,20),np.linspace(0,0,20),label='lower bound')
plt.plot(np.linspace(0,0.01,20),np.linspace(.1,.1,20),label='lower bound')
plt.xlabel('initial guess [$\\frac{M_{\\odot}}{pc^3}$]')
plt.ylabel('fitted parameters [$\\frac{M_{\\odot}}{pc^3}$]')
plt.title('Negative')
plt.legend()
plt.show()
################################################################################


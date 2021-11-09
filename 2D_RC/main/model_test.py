from scipy import integrate as inte
import numpy as np
import matplotlib.pyplot as plt
from numpy import log as ln

G = 6.67E-11  # m^3 kg^-1 s^-2
Msun = 1.989E30  # kg

from astropy.table import QTable
from scipy.optimize import minimize
import astropy.units as u
from scipy.special import kn
from scipy.special import iv

from rotation_curve_functions import disk_vel,\
                                     vel_h_iso,\
                                     vel_h_NFW,\
                                     vel_h_Burket

from galaxy_component_functions import  halo_vel_iso,\
                                        halo_vel_NFW,\
                                        halo_vel_bur,\
                                        vel_tot_iso_nb,\
                                        vel_tot_NFW_nb,\
                                        vel_tot_bur_nb

# range of radii

r = np.linspace(0.1,55,10)


# Disk
plt.plot(np.array(r),disk_vel(r*1000,1048,5000),label='disk')
plt.title('Disk')
plt.legend()
plt.show()
# Isothermal halo
# integral version
plt.plot(np.array(r),vel_h_iso(r*1000,130,30000),'o',label='integral')
plt.title('Isothermal')
plt.legend()
#plt.show()
# no integral version
plt.plot(np.array(r),halo_vel_iso(r*1000, 8e-4,30000),'*',label='no-integral')
plt.title('Isothermal')
plt.legend()
plt.show()
# NFW halo
# integral version
plt.plot(np.array(r),vel_h_NFW(r*1000, 8e-4,30000),'o',label='integral')
plt.title('NFW')
plt.legend()
#plt.show()
# no integral version
plt.plot(np.array(r), halo_vel_NFW(r*1000, 8e-4,30000),'*',label='no-integral')
plt.title('NFW')
plt.legend()
plt.show()
# Burket halo
# integral version
plt.plot(np.array(r),vel_h_Burket(r*1000, 8e-4,30000),'o',label='integral')
plt.title('Burket')
plt.legend()
#plt.show()
# no integral version
plt.plot(np.array(r),halo_vel_bur(r*1000, 8e-4,30000),'*',label='no-integral')
plt.title('Burket')
plt.legend()
plt.show()

# Example:
# No Bulge
Fit_nb = [1000, 5, 0.005, 25]
# Isothermal
plt.plot(np.array(r),vel_tot_iso_nb(r,Fit_nb),'*',label='Isothermal')
# NFW
plt.plot(np.array(r),vel_tot_NFW_nb(r,Fit_nb),'o',label='NFW')
# Burket
plt.plot(np.array(r),vel_tot_bur_nb(r,Fit_nb),'--',label='Burket')
plt.legend()
plt.show()
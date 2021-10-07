################################################################################
# All the libraries used & constant values
# ------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from rotation_curve_functions import vel_h_iso, vel_h_NFW, vel_h_Burket
################################################################################

r_halo = 15
r = np.linspace(0.1, 10 *r_halo,100)
roh = r/r_halo

################################################################################
# For the purpose of finding the position of maximum velocity and the halo scale radius
# Find maximum on a plot （Isothermal)
def findMax_Iso(r,v_inf,r_h):
    for i in range(len(r)):
            if vel_h_iso(r[i]*1000,v_inf,r_h*1000) == max(vel_h_iso(r*1000,v_inf,r_h*1000)):
                x_Max = r[i]/r_h
    return x_Max

# Find maximum on a plot （Burket）
def findMax_Bur(r,rho,r_h):
    for i in range(len(r)):
            if vel_h_Burket(r[i]*1000,rho,r_h*1000) == max(vel_h_Burket(r*1000,rho,r_h*1000)):
                x_Max = r[i]/r_h
    return x_Max

# Find maximum on a plot (NFW)
def findMax_NFW(r,rho,r_h):
    for i in range(len(r)):
            if vel_h_NFW(r[i]*1000,rho,r_h*1000) == max(vel_h_NFW(r*1000,rho,r_h*1000)):
                x_Max = r[i]/r_h
    return x_Max
################################################################################

################################################################################

################################################################################

################################################################################
# Isothermal Model
V_inf = 100
plt.plot(roh,vel_h_iso(r*1000,V_inf,r_halo*1000))
plt.xlabel('r/$R_h$')
plt.ylabel('$v_{rot}$')
plt.legend()
plt.grid(b=True)
print(findMax_Iso(r,V_inf,r_halo))
half_max = findMax_Iso(r,V_inf,r_halo)/2
print(half_max)
print(vel_h_iso(half_max*1000,V_inf,r_halo))

V_inf = 200
plt.plot(roh,vel_h_iso(r*1000,V_inf,r_halo*1000))
plt.xlabel('r/$R_h$')
plt.ylabel('$v_{rot}$')
plt.legend()
plt.grid(b=True)
print(findMax_Iso(r,V_inf,r_halo))

V_inf = 300
plt.plot(roh,vel_h_iso(r*1000,V_inf,r_halo*1000))
plt.xlabel('r/$R_h$')
plt.ylabel('$v_{rot}$')
plt.legend()
plt.grid(b=True)
print(findMax_Iso(r,V_inf,r_halo))

V_inf = 400
plt.plot(roh,vel_h_iso(r*1000,V_inf,r_halo*1000))
plt.xlabel('r/$R_h$')
plt.ylabel('$v_{rot}$')
plt.legend()
plt.grid(b=True)
print(findMax_Iso(r,V_inf,r_halo))
plt.show()
################################################################################

################################################################################
# NFW Model

rho_0_NFW = 3E-3
plt.plot(roh,vel_h_NFW(r*1000,rho_0_NFW,r_halo*1000))
plt.xlabel('r/$R_h$')
plt.ylabel('$v_{rot}$')
plt.legend()
plt.grid(b=True)
print(findMax_NFW(r,rho_0_NFW,r_halo))

rho_0_NFW = 9E-3
plt.plot(roh,vel_h_NFW(r*1000,rho_0_NFW,r_halo*1000))
plt.xlabel('r/$R_h$')
plt.ylabel('$v_{rot}$')
plt.legend()
plt.grid(b=True)
print(findMax_NFW(r,rho_0_NFW,r_halo))

rho_0_NFW = 3E-2
plt.plot(roh,vel_h_NFW(r*1000,rho_0_NFW,r_halo*1000))
plt.xlabel('r/$R_h$')
plt.ylabel('$v_{rot}$')
plt.legend()
plt.grid(b=True)
print(findMax_NFW(r,rho_0_NFW,r_halo))

rho_0_NFW = 9E-2
plt.plot(roh,vel_h_NFW(r*1000,rho_0_NFW,r_halo*1000))
plt.xlabel('r/$R_h$')
plt.ylabel('$v_{rot}$')
plt.legend()
plt.grid(b=True)
plt.show()
print(findMax_NFW(r,rho_0_NFW,r_halo))
################################################################################

################################################################################
# Burket Model

rho_0_Burket = 3E-3
plt.plot(roh,vel_h_Burket(r*1000,rho_0_Burket,r_halo*1000))
plt.xlabel('r/$R_h$')
plt.ylabel('$v_{rot}$')
plt.legend()
plt.grid(b=True)
print(findMax_Bur(r,rho_0_Burket,r_halo))


rho_0_Burket = 9E-3
plt.plot(roh,vel_h_Burket(r*1000,rho_0_Burket,r_halo*1000))
plt.xlabel('r/$R_h$')
plt.ylabel('$v_{rot}$')
plt.legend()
plt.grid(b=True)
print(findMax_Bur(r,rho_0_Burket,r_halo))

rho_0_Burket = 3E-2
plt.plot(roh,vel_h_Burket(r*1000,rho_0_Burket,r_halo*1000))
plt.xlabel('r/$R_h$')
plt.ylabel('$v_{rot}$')
plt.legend()
plt.grid(b=True)
print(findMax_Bur(r,rho_0_Burket,r_halo))

rho_0_Burket = 9E-2
plt.plot(roh,vel_h_Burket(r*1000,rho_0_Burket,r_halo*1000))
plt.xlabel('r/$R_h$')
plt.ylabel('$v_{rot}$')
plt.legend()
plt.show()
plt.grid(b=True)
print(findMax_Bur(r,rho_0_Burket,r_halo))
################################################################################

################################################################################
# Testing the scale radius of halo
'''
rho_0_Burket_1 = 3E-3
r_halo_1 = 5
r_1 = np.linspace(0,5*r_halo_1,100)
roh1 = r_1/r_halo_1
plt.plot(roh1,vel_h_Burket(r_1*1000,rho_0_Burket_1,r_halo_1*1000),color='brown',label='$R_h$ = 5 kpc')
plt.xlabel('r/$R_h$')
plt.ylabel('$v_{rot}$')
plt.legend()
plt.grid(b=True)

r_halo_1 = 10
r_1 = np.linspace(0,5*r_halo_1,100)
roh1 = r_1/r_halo_1
plt.plot(roh1,vel_h_Burket(r_1*1000,rho_0_Burket_1,r_halo_1*1000),color='blue',label='$R_h$ = 10 kpc')
plt.xlabel('r/$R_h$')
plt.ylabel('$v_{rot}$')
plt.legend()
plt.grid(b=True)

r_halo_1 = 15
r_1 = np.linspace(0,5*r_halo_1,100)
roh1 = r_1/r_halo_1
plt.plot(roh1,vel_h_Burket(r_1*1000,rho_0_Burket_1,r_halo_1*1000),color='green',label='$R_h$ = 15 kpc')
plt.xlabel('r/$R_h$')
plt.ylabel('$v_{rot}$')
plt.legend()
plt.grid(b=True)

r_halo_1 = 20
r_1 = np.linspace(0,5*r_halo_1,100)
roh1 = r_1/r_halo_1
plt.plot(roh1,vel_h_Burket(r_1*1000,rho_0_Burket_1,r_halo_1*1000),color='red',label='$R_h$ = 20 kpc')
plt.xlabel('r/$R_h$')
plt.ylabel('$v_{rot}$')
plt.legend()
plt.grid(b=True)
plt.show()
'''
################################################################################

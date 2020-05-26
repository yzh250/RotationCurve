from scipy import integrate as inte
import numpy as np
import matplotlib.pyplot as plt
from numpy import log as ln
import sympy as sym
import multiprocessing

G = 6.67*10**(-8)
R = np.linspace(0.005,100,10)

sigma_be = 10
r_b = 8
gamma = 3.3308
kappa = gamma*ln(10)

# Bulge

# Surface mass density
sigma_b = lambda r: sigma_be*np.exp(-1*kappa*(((r/r_b)**(0.25))-1))

# derivative of sigma with respect to x
dsdr = lambda r: sigma_b(r)*(-1*kappa*((1/r_b)**(0.25))*(0.25)*(r**(-1*(0.75))))

# integrand for getting rho (in terms of x)
integrand_d = lambda R, r: 4*np.pi*(R**2)*(1/(np.pi))*dsdr(r)*np.sqrt(1/(r**2-(R)**2))

res = np.zeros_like(R)
for i, val in enumerate(R):
    y, err = inte.dblquad(integrand_d, 0, val, lambda r: val, lambda r: 1000)
    res[i] = y

plt.plot(R,res)
plt.ylabel('m')
plt.xlabel('radius')
plt.show()

'''
def integrate(min_val, max_val, d_val, ifun):
    val_iter = min_val
    totals = 0
    while val_iter < max_val:
        totals += ifun(val_iter)
        val_iter += d_val
    totals = totals * d_val
    return totals

print (integrate(R+0.01, 1000, 0.01, integrand_s_1))

plt.plot(np.linspace(-100, 100000, 10000), integrand_s_1(np.linspace(-100, 100000, 10000)))
plt.show()
'''

'''
# Exponential Disk
sigma_dc = 10
r_d = 50
r = np.linspace(0.5,100,10)

def sigma_d(r):
    return sigma_dc*np.exp(-1*(r/r_d))

def integrand_d(r):
    return 2*np.pi*sigma_d(r)*r

def m_d(r):
    res = np.zeros_like(r)
    for i, val in enumerate(r):
        y, err = inte.quad(integrand_d, 0, val)
        res[i] = y
    return res

def v_d(r):
    return np.sqrt((m_d(r)*G)/r)

plt.plot(r,v_d(r))
plt.ylabel('velocity')
plt.xlabel('radius')
plt.show()

h = 50
# Halo 1
rho_0_NFW = 25

def rho_NFW(r):
    return rho_0_NFW/((r/h)*(1 + (r/h))**2)

def integrand_h_1(r):
    return 4*np.pi*rho_NFW(r)*r**2

def m_h_1(r):
    res = np.zeros_like(r)
    for i, val in enumerate(r):
        y, err = inte.quad(integrand_h_1, 0, val)
        res[i] = y
    return res

def v_h_1(r):
    return np.sqrt((m_h_1(r)*G)/r)

# Halo 2
rho_0_bur = 30

def rho_bur(r):
    return rho_0_bur/((1+(r/h))*(1 + (r/h)**2))

def integrand_h_2(r):
    return 4*np.pi*rho_bur(r)*r**2

def m_h_2(r):
    res = np.zeros_like(r)
    for i, val in enumerate(r):
        y, err = inte.quad(integrand_h_2, 0, val)
        res[i] = y
    return res

def v_h_2(r):
    return ((m_h_2(r)*G)/r)**(1/2)

# Plot combined for the two halo models
plt.plot(r,v_h_1(r))
plt.plot(r,v_h_2(r))
plt.ylabel('velocity')
plt.xlabel('radius')
plt.show()
'''
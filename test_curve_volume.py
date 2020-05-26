from scipy import integrate as inte
import numpy as np
import matplotlib.pyplot as plt
G = 6.67*10**(-8)
r = np.linspace(0.5,100,100)
M = 50

# Bulge (Constant density)
rho_0 = 10

def integrand_1(r):
    return 4*np.pi*rho_0*(r**2)

def m_1(r):
    res = np.zeros_like(r)
    for i, val in enumerate(r):
        y, err = inte.quad(integrand_1, 0, val)
        res[i] = y
    return res

def v_1(r):
    return ((m_1(r)*G)/r)**(1/2)

plt.plot(r,v_1(r))
plt.ylabel('velocity')
plt.xlabel('radius')
plt.show()

# Dark Matter Halo (1/r^2)
A_0 = 10

def rho(r):
    return A_0*(1/(r**2))

def integrand_2(r):
    return 4*np.pi*rho(r)*(r**2)

def m_2(r):
    res = np.zeros_like(r)
    for i,val in enumerate(r):
        y,err = inte.quad(integrand_2,0,val)
        res[i]=y
    return res

def v_2(r):
    return ((m_2(r)*G)/r)**(1/2)

plt.plot(r,v_2(r))
plt.show()

# Keplerian curve (point mass)

def v_kep(r):
    return ((M*G)/r)**(1/2)

plt.plot(r,v_kep(r))
plt.ylabel('velocity')
plt.xlabel('radius')
plt.show()

# Combined curve (all three)

def v_co(r):
    return (v_1(r)**2 + v_2(r)**2 + v_kep(r)**2)**(1/2)

plt.plot(r,v_co(r))
plt.ylabel('velocity')
plt.xlabel('radius')
plt.show()
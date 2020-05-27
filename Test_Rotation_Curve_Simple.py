#!/usr/bin/env python
# coding: utf-8

# In[61]:


from scipy import integrate as inte
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
G = 6.67*(10**(-8))*(1/(3.08*(10**18)))**3


# In[68]:


# Disk (Constant density)
R_d = np.linspace(10,3*10**3)
rho_0 = 8*1.988*10**33

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


# In[69]:


# Dark Matter Halo (1/r^2)
rho_0 = (5.1*(10**-3)*1.988*10**33)
r_0 = 12*10**3
R_h = np.linspace(10,20*10**3)

def rho(r):
    return rho_0*((r_0/r)**2)

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


# In[75]:


# Bulge (Keplerian)
R_b = np.linspace(10,0.5*10**3)
M_b = 1.8 * (10 ** 10) * 1.988*10**33

def v_kep(r):
    return ((M_b*G)/r)**(1/2)


# In[76]:


# Combined curve (all three)

def v_co(r):
    return (v_1(r)**2 + v_2(r)**2 + v_kep(r)**2)**(1/2)


# In[83]:


plt.plot(R_h/1000,v_co(R_h)*(3.08*10**13),label = 'velocity combined')
plt.plot(R_h/1000,v_kep(R_h)*(3.08*10**13),label = 'velocity bulge')
plt.plot(R_h/1000,v_2(R_h)*(3.08*10**13),label = 'velocity halo')
plt.plot(R_h/1000,v_1(R_h)*(3.08*10**13),label = 'velocity disk')
plt.ylabel('velocity [km/s]')
plt.xlabel('radius [kpc]')
plt.ylim(0,500)
plt.legend()
plt.show()
plt.savefig('rotation_curve_simple.png')


# In[ ]:





# In[ ]:





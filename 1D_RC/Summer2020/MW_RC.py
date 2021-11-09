from scipy import integrate as inte
import numpy as np
import matplotlib.pyplot as plt
from numpy import log as ln

G = 6.67 * (10 ** (-8)) * (1 / (3.08 * (10 ** 18))) ** 3
R = np.linspace(0.01, 30 * 10 ** 3)
h = 10 * 10 ** 3

# de Vaucouleurs bulge
sigma_be = 3.2 * (10 ** 3) * 1.988 * (10 ** 33)
r_b = 0.5 * 10 ** 3
gamma = 3.3308
kappa = gamma * ln(10)


def sigma_b(x):
    return sigma_be * np.exp(-1 * kappa * ((x / r_b) ** 0.25 - 1))


# derivative of sigma with respect to r
def dsdx(x):
    return sigma_b(x) * (-0.25 * kappa) * (r_b ** -0.25) * (x ** -0.75)


# integrand for getting denisty
def density_integrand(x, r):
    return -(1 / np.pi) * dsdx(x) / np.sqrt(x ** 2 - r ** 2)


def mass_integrand(r):
    vol_den, vol_den_err = inte.quad(density_integrand, r, np.inf, args=(r))
    return 4 * np.pi * vol_den * r ** 2


bulge_mass = np.zeros(len(R))
vel = np.zeros(len(R))

# getting a mass
for i, val in enumerate(R):
    bulge_mass[i], err = inte.quad(mass_integrand, 0, val)
    vel[i] = np.sqrt(bulge_mass[i] * G / val)

print(vel)
plt.plot(R / 1000, vel * (3.08 * 10 ** 13))
plt.ylabel('velocity [km/s]')
plt.xlabel('radius [kpc]')
plt.show()

# Exponential Disk
sigma_dc = 8.44 * (10 ** 2) * (1.988 * 10 ** 33)
r_d = 3.5 * 10 ** 3


def sigma_d(r):
    return sigma_dc * np.exp(-1 * (r / r_d))


def integrand_d(r):
    return 2 * np.pi * sigma_d(r) * r


def m_d(r):
    res = np.zeros_like(r)
    for i, val in enumerate(r):
        y, err = inte.quad(integrand_d, 0, val)
        res[i] = y
    return res


def v_d(r):
    return np.sqrt((m_d(r) * G) / r)


plt.plot(R / 1000, v_d(R) * (3.08 * 10 ** 13))
plt.ylabel('velocity [km/s]')
plt.xlabel('radius [kpc]')
plt.show()

# Isothermal
rho_0_iso = 0.74 * (h / (10 ** 3)) ** -2 * (1.988 * 10 ** 33)


def rho_iso(r):
    return rho_0_iso / (1 + (r / h) ** 2)


def integrand_h_iso(r):
    return 4 * np.pi * rho_iso(r) * r ** 2


def m_h_iso(r):
    res = np.zeros_like(r)
    for i, val in enumerate(r):
        y, err = inte.quad(integrand_h_iso, 0, val)
        res[i] = y
    return res


def v_h_iso(r):
    return np.sqrt((m_h_iso(r) * G) / r)


plt.plot(R / 1000, v_h_iso(R) * (3.08 * 10 ** 13))
plt.ylabel('velocity [km/s]')
plt.xlabel('radius [kpc]')
plt.show()

# NFW
rho_0_NFW = rho_0_bur = 1.3 * (h / (10 ** 3)) ** -2 * (1.988 * 10 ** 33)


def rho_NFW(r):
    return rho_0_NFW / ((r / h) * (1 + (r / h)) ** 2)


def integrand_h_NFW(r):
    return 4 * np.pi * rho_NFW(r) * r ** 2


def m_h_NFW(r):
    res = np.zeros_like(r)
    for i, val in enumerate(r):
        y, err = inte.quad(integrand_h_NFW, 0, val)
        res[i] = y
    return res


def v_h_NFW(r):
    return np.sqrt((m_h_NFW(r) * G) / r)


plt.plot(R / 1000, v_h_NFW(R) * (3.08 * 10 ** 13))
plt.ylabel('velocity [km/s]')
plt.xlabel('radius [kpc]')
plt.show()

# Burkert
rho_0_bur = 1.5 * (h / (10 ** 3)) ** -2 * (1.988 * 10 ** 33)


def rho_bur(r):
    return rho_0_bur / ((1 + (r / h)) * (1 + (r / h) ** 2))


def integrand_h_bur(r):
    return 4 * np.pi * rho_bur(r) * r ** 2


def m_h_bur(r):
    res = np.zeros_like(r)
    for i, val in enumerate(r):
        y, err = inte.quad(integrand_h_bur, 0, val)
        res[i] = y
    return res


def v_h_bur(r):
    return np.sqrt((m_h_bur(r) * G) / r)


plt.plot(R / 1000, v_h_bur(R) * (3.08 * 10 ** 13))
plt.ylabel('velocity [km/s]')
plt.xlabel('radius [kpc]')
plt.show()

plt.plot(R / 1000, v_h_NFW(R) * (3.08 * 10 ** 13), label='NFW')
plt.plot(R / 1000, v_h_iso(R) * (3.08 * 10 ** 13), label='isothermal')
plt.plot(R / 1000, v_h_bur(R) * (3.08 * 10 ** 13), label='Burket')
plt.ylabel('velocity [km/s]')
plt.xlabel('radius [kpc]')
plt.legend()
plt.show()

plt.plot(R / 1000, vel * (3.08 * 10 ** 13), label='Bulge')
plt.plot(R / 1000, v_d(R) * (3.08 * 10 ** 13), label='Disk')
plt.plot(R / 1000, v_h_NFW(R) * (3.08 * 10 ** 13), label='NFW')
plt.plot(R / 1000, v_h_iso(R) * (3.08 * 10 ** 13), label='isothermal')
plt.plot(R / 1000, v_h_bur(R) * (3.08 * 10 ** 13), label='Burket')
plt.ylabel('velocity [km/s]')
plt.xlabel('radius [kpc]')
plt.legend()
plt.show()

vel_t = np.sqrt(vel ** 2 + v_d(R) ** 2 + v_h_iso(R) ** 2)
vel_t_1 = np.sqrt(vel ** 2 + v_d(R) ** 2 + v_h_NFW(R) ** 2)
vel_t_2 = np.sqrt(vel ** 2 + v_d(R) ** 2 + v_h_bur(R) ** 2)

plt.plot(R / 1000, vel_t * (3.08 * 10 ** 13), label='Total Curve (w/ iso)')
plt.plot(R / 1000, vel_t_1 * (3.08 * 10 ** 13), label='Total Curve (w/ NFW)')
plt.plot(R / 1000, vel_t_2 * (3.08 * 10 ** 13), label='Total Curve (w/ Burket)')
plt.plot(R / 1000, vel * (3.08 * 10 ** 13), label='Bulge')
plt.plot(R / 1000, v_d(R) * (3.08 * 10 ** 13), label='Disk')
plt.plot(R / 1000, v_h_iso(R) * (3.08 * 10 ** 13), label='isothermal')
plt.plot(R / 1000, v_h_NFW(R) * (3.08 * 10 ** 13), label='NFW')
plt.plot(R / 1000, v_h_bur(R) * (3.08 * 10 ** 13), label='Burket')
plt.ylabel('velocity [km/s]')
plt.xlabel('radius [kpc]')
plt.legend(loc='upper left')
plt.show()

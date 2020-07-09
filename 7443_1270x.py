########################################################################################################################
# All the libraries used & constant values
from scipy import integrate as inte
import numpy as np
import matplotlib.pyplot as plt
from numpy import log as ln
G = 6.67E-11 # m^3 kg^-1 s^-2

from astropy.table import QTable
from scipy.optimize import curve_fit
import astropy.units as u
from scipy.special import kn
from scipy.special import iv
from scipy.optimize import minimize
import dynesty
from dynesty import plotting as dyplot
########################################################################################################################

########################################################################################################################
# Reading in all the objects with Plate No. 7443
DTable1 = QTable.read('7443-12701_rot_curve_data.txt', format='ascii.ecsv')
DTable2 = QTable.read('7443-12702_rot_curve_data.txt', format='ascii.ecsv')
DTable3 = QTable.read('7443-12703_rot_curve_data.txt', format='ascii.ecsv')
DTable4 = QTable.read('7443-12704_rot_curve_data.txt', format='ascii.ecsv')
DTable5 = QTable.read('7443-12705_rot_curve_data.txt', format='ascii.ecsv')
########################################################################################################################

########################################################################################################################
# bulge (Simpler Model)
def vel_b(r,a,b,c):
    '''
    :param r: The projected radius (pc)
    :param a: Some constant (unit less)
    :param b: the velocity when r approaches infinity (km/s)
    :param c: The scale radius of the disk (pc)
    :return: The rotational velocity of the bulge (km/s)
    '''
    v_b2 = a*(b**2)*((r/(0.2*c))**-1)
    return np.sqrt(v_b2)
########################################################################################################################

########################################################################################################################
# bulge (Not the simplest model)
''''
gamma = 3.3308 # unitless
kappa = gamma*ln(10) # unitless
def sigma_b(x,a,b):
    """
    parameters:
    x (projected radius): The projected radius  (pc)
    a (central density): The central density of the bulge (M_sol/pc^2)
    b (central radius): The central radius of the bulge (kpc)
    
    return: surface density of the bulge (g/pc^2)
    """
    return a*np.exp(-1*kappa*((x/b)**0.25-1)) #M_sol/pc^2
                           
# derivative of sigma with respect to r
def dsdx(x,a,b):
    """
    parameters:
    x (projected radius): The projected rdius  (pc)
    a (central density): The central density of the bulge (M/pc^2)
    b (central radius): The central radius of the bulge (kpc)
    
    return: derivative of sigma (g/pc^3)
    """
    return sigma_b(x,a,b)*(-0.25*kappa)*(b**-0.25)*(x**-0.75) # M_sol/pc^2

# integrand for getting denisty
def density_integrand(x,r,a,b):
    """
    parameters:
    x (projected radius): The projected rdius  (pc)   
    r (radius): The a distance from the centre (pc)
    a (central density): The central density of the bulge (M/pc^2)
    b (central radius): The central radius of the bulge (kpc)
    
    return: integrand for volume density of the bulge (g/pc^3)
    """
    return -(1/np.pi)*dsdx(x,a,b)/np.sqrt(x**2-r**2)

def mass_integrand(r,a,b):
    """
    parameters:
    x (projected radius): The projected rdius  (pc)   
    r (radius): The a distance from the centre (pc)
    a (central density): The central density of the bulge (M/pc^2)
    b (central radius): The central radius of the bulge (kpc)
    
    return: volume density of the bulge
    """
    vol_den, vol_den_err = inte.quad(density_integrand, r, np.inf, args=(r,a,b))
    return 4*np.pi*vol_den*r**2
    
# getting a velocity
def vel_b(r,a,b):
    """
    parameters:
    r (radius): The a distance from the centre (pc)
    a (central density): The central density of the bulge (M/pc^2)
    b (central radius): The central radius of the bulge (kpc)
    
    return: rotational velocity of the bulge (pc/s)
    """
    if isinstance(r, float):
        bulge_mass, m_err = inte.quad(mass_integrand, 0, r, args=(Sigma_be, Rb))
    else:
        bulge_mass = np.zeros(len(r))
        err = np.zeros(len(r))
        
        for i in range(len(r)):
            bulge_mass[i],err[i] = inte.quad(mass_integrand, 0, r[i], args=(a,b))
    vel = np.sqrt(G*(bulge_mass*1.988E30)/(r*3.08E16))
    vel /= 1000
    return vel
'''
########################################################################################################################

########################################################################################################################
# Disk velocity from Paolo 2018
def v_d(r,c,d):
    '''
    :param r: The a distance from the centre (pc)
    :param c: The total mass Xof the disk (M)
    :param d: The scale radius of the disk (pc)
    :return: The rotational velocity of the disk (km/s)
    '''
    bessel_component = (iv(0,r/(2*d))*kn(0,r/(2*d))-iv(1,r/(2*d))*kn(1,r/(2*d)))
    vel = ((0.5)*G*(1.988E30*c)*(r/d)**2/(d*3.08E16))*bessel_component
    return np.sqrt(vel)/1000
########################################################################################################################

########################################################################################################################
# halo (isothermal)
# e = rho_0_iso
# f = h

def rho0_iso(e,f):
    '''
    parameters:
    r (radius): The a distance from the centre (pc)
    e (rotational velocity): The rotational velocity when r approaches infinity (km/s)
    f (scale radius): The scale radius of the dark matter halo (pc)
    
    return: volume density of the isothermal halo (g/pc^3)
    '''
    return 0.740*(e/200)*(f)**(-2)

def rho_iso(r,e,f):
    '''
    parameters:
    r (radius): The a distance from the centre (pc)
    e (rotational velocity): The rotational velocity when r approaches infinity (km/s)
    f (scale radius): The scale radius of the dark matter halo (pc)
    
    return: volume density of the isothermal halo (g/pc^3)
    '''
    rho_0 = rho0_iso(e, f/1000)
    return rho_0/(1 + (r/f)**2)

def integrand_h_iso(r,e,f):
    '''
    parameters:
    r (radius): The a distance from the centre (pc)
    e (rotational velocity): The rotational velocity when r approaches infinity (km/s)
    f (scale radius): The scale radius of the dark matter halo (pc)

    return: integrand for getting the mass of the isothermal halo 
    '''
    
    return 4*np.pi*(rho_iso(r,e,f))*r**2

def mass_h_iso(r,e,f):
    '''
    parameters:
    r (radius): The a distance from the centre (pc)
    e (rotational velocity): The rotational velocity when r approaches infinity (km/s)
    f (scale radius): The scale radius of the dark matter halo (pc)
    
    return: mass of the isothermal halo (g)
    '''
    halo_mass, m_err = inte.quad(integrand_h_iso, 0, r, args=(e, f))
    return halo_mass

def vel_h_iso(r,e,f):
    '''
    parameters:
    r (radius): The a distance from the centre (pc)
    e (rotational velocity): The rotational velocity when r approaches infinity (km/s)
    f (scale radius): The scale radius of the dark matter halo (pc)
    
    return: rotational velocity of the isothermal halo (pc/s)
    '''
    if isinstance(r, float):
        halo_mass = mass_h_iso(r,e,f)
    else:
        halo_mass = np.zeros(len(r))
        for i in range(len(r)):
            halo_mass[i] = mass_h_iso(r[i],e,f)
    vel = np.sqrt(G*(halo_mass*1.988E30)/(r*3.08E16))
    vel /= 1000
    return vel
########################################################################################################################

########################################################################################################################
# halo (NFW)
# e = rho_0_NFW
# f = h

def rho_NFW(r,e,f):
    '''
    parameters:
    r (radius): The a distance from the centre (pc)
    e (central density): The central density of the halo (M_sol/pc^3)
    f (scale radius): The scale radius of the dark matter halo (pc)
    
    return: volume density of the isothermal halo (M/pc^3)
    '''
    return  (e)/((r/f)*((1+(r/f))**2))

def integrand_h_NFW(r,e,f):
    '''
    parameters:
    r (radius): The a distance from the centre (pc)
    e (central density): The central density of the halo (M_sol/pc^3)
    f (scale radius): The scale radius of the dark matter halo (pc)
    
    return: integrand for getting the mass of the isothermal halo 
    '''
    
    return 4*np.pi*(rho_NFW(r,e,f))*r**2

def mass_h_NFW(r,e,f):
    '''
    parameters:
    r (radius): The a distance from the centre (pc)
    e (central density): The central density of the halo (M_sol/pc^3)
    f (scale radius): The scale radius of the dark matter halo (pc)
    
    return: mass of the isothermal halo (g)
    '''
    halo_mass, m_err = inte.quad(integrand_h_NFW, 0, r, args=(e, f))
    return halo_mass

def vel_h_NFW(r,e,f):
    '''
    parameters:
    r (radius): The a distance from the centre (pc)
    e (central density): The central density of the halo (M_sol/pc^3)
    f (scale radius): The scale radius of the dark matter halo (pc)
    
    return: rotational velocity of the NFW halo (pc/s)
    '''
    if isinstance(r, float):
        halo_mass = mass_h_NFW(r,e,f)
    else:
        halo_mass = np.zeros(len(r))
        for i in range(len(r)):
            halo_mass[i] = mass_h_NFW(r[i],e,f)
    vel = np.sqrt(G*(halo_mass*1.98843E30)/(r*3.0857E16))
    vel /= 1000
    return vel
########################################################################################################################

########################################################################################################################
# halo (Burket)
# e = rho_0_Bur
# f = h

def rho_Burket(r,e,f):
    '''
    :param r: The distance from the centre (pc)
    :param e: The central density of the halo (M_sol/pc^3)
    :param f: The scale radius of the dark matter halo (pc)
    :return: volume density of the isothermal halo (M/pc^3)
    '''
    return  (e*f**3)/((r + f)*(r**2+f**2))

def integrand_h_Burket(r,e,f):
    '''
    :param r: The a distance from the centre (pc)
    :param e: The central density of the halo (M_sol/pc^3)
    :param f: The scale radius of the dark matter halo (pc)
    :return: integrand for getting the mass of the isothermal halo
    '''
    return 4*np.pi*(rho_Burket(r,e,f))*r**2

def mass_h_Burket(r,e,f):
    '''
    :param r: The a distance from the centre (pc)
    :param e: The central density of the halo (M_sol/pc^3)
    :param f: The scale radius of the dark matter halo (pc)
    :return: mass of the isothermal halo (g)
    '''
    halo_mass, m_err = inte.quad(integrand_h_Burket, 0, r, args=(e, f))
    return halo_mass

def vel_h_Burket(r,e,f):
    '''
    r (radius): The a distance from the centre (pc)
    e (central density): The central density of the halo (M_sol/pc^3)
    f (scale radius): The scale radius of the dark matter halo (pc)
    :return: rotational velocity of the Burket halo (pc/s)
    '''
    if isinstance(r, float):
        halo_mass = mass_h_Burket(r,e,f)
    else:
        halo_mass = np.zeros(len(r))
        for i in range(len(r)):
            halo_mass[i] = mass_h_Burket(r[i],e,f)
    vel = np.sqrt(G*(halo_mass*1.98843E30)/(r*3.0857E16))
    vel /= 1000
    return vel
########################################################################################################################

########################################################################################################################
# Combined Velocity of Isothermal Model
def v_co_iso(r,a,b,c,d,e,f):
    '''
    r (radius): The a distance from the centre (kpc)
    a (central density): The central density of the bulge (M_sol/pc^2)
    b (central radius): The central radius of the bulge (kpc)
    c (mass of disk): The total mass Xof the disk (M)
    d (radius radius): The central radius of the disk (kpc)
    e (rotational velocity): The rotational velocity when r approaches infinity (km/s)
    f (scale radius): The scale radius of the dark matter halo (kpc)
    '''
    
    return np.sqrt(vel_b(r,a,b)**2+v_d(r,c,d)**2 + vel_h_iso(r,e,f)**2) #km/s


# Combined Velocity of Isothermal Model (No Bulge)
def v_co_iso_nb(r,c,d,e,f):
    '''
    r (radius): The a distance from the centre (kpc)
    c (mass of disk): The total mass Xof the disk (M)
    d (radius radius): The central radius of the disk (kpc)
    e (rotational velocity): The rotational velocity when r approaches infinity (km/s)
    f (scale radius): The scale radius of the dark matter halo (kpc)
    '''
    return np.sqrt(v_d(r,c,d)**2 + vel_h_iso(r,e,f)**2) #km/s


# Combined Velocity of NFW Model
def v_co_NFW(r,a,b,c,d,e,f):
    '''
    r (radius): The a distance from the centre (kpc)
    a (central density): The central density of the bulge (M_sol/pc^2)
    b (central radius): The central radius of the bulge (kpc)
    c (mass of disk): The total mass Xof the disk (M)
    d (radius radius): The central radius of the disk (kpc)
    e (central density): The central density of the halo (M_sol/pc^3)
    f (scale radius): The scale radius of the dark matter halo (kpc)
    '''
    return np.sqrt(vel_b(r,a,b)**2+v_d(r,c,d)**2 + vel_h_NFW(r,e,f)**2) #km/s


# Combined Velocity of NFW Model (No Bulge)
def v_co_NFW_nb(r,c,d,e,f):
    '''
    r (radius): The a distance from the centre (kpc)
    c (mass of disk): The total mass Xof the disk (M)
    d (radius radius): The central radius of the disk (kpc)
    e (central density): The central density of the halo (M_sol/pc^3)
    f (scale radius): The scale radius of the dark matter halo (kpc)
    '''
    return np.sqrt(v_d(r*1000,10**c,d*1000)**2 + vel_h_NFW(r*1000,e,f*1000)**2) #km/s


# Combined Velocity of Burket Model
def v_co_Burket(r,a,b,c,d,e,f):
    '''
    :param r: The a distance from the centre (kpc)
    :param a: Some constant (unit less)
    :param b: the velocity when r approaches infinity (km/s)
    :param c: The total mass of the disk (M)
    :param d: The central radius of the disk (kpc)
    :param e: The central density of the halo (M_sol/pc^3)
    :param f: The scale radius of the dark matter halo (pc)
    :return: The combined rotational velocity using the Burket model (km/s)
    '''
    return np.sqrt(vel_b(r*1000,a,b,d*1000)**2+v_d(r*1000,10**c,d*1000)**2 + vel_h_Burket(r*1000,e,f*1000)**2) #km/s

# Combined Velocity of Burket Model (No Bulge)
def v_co_Burket_nb(r,c,d,e,f):
    '''
    :param r: The a distance from the centre (kpc)
    :param c: The total mass of the disk (M)
    :param d: The central radius of the disk (kpc)
    :param e: The central density of the halo (M_sol/pc^3)
    :param f: The scale radius of the dark matter halo (pc)
    :return: The combined rotational velocity using the Burket model (km/s)
    '''
    return np.sqrt(v_d(r*1000,10**c,d*1000)**2 + vel_h_Burket(r*1000,e,f*1000)**2) #km/s
########################################################################################################################

########################################################################################################################
# Reading in the radii, velocity data
r_1=DTable1['deprojected_distance'].data
r_2=DTable2['deprojected_distance'].data
r_3=DTable3['deprojected_distance'].data
r_4=DTable4['deprojected_distance'].data
r_5=DTable5['deprojected_distance'].data
av_1=DTable1['rot_vel_avg'].data
av_2=DTable2['rot_vel_avg'].data
av_3=DTable3['rot_vel_avg'].data
av_4=DTable4['rot_vel_avg'].data
av_5=DTable5['rot_vel_avg'].data
mav_1=DTable1['max_velocity'].data
mav_2=DTable2['max_velocity'].data
mav_3=DTable3['max_velocity'].data
mav_4=DTable4['max_velocity'].data
mav_5=DTable5['max_velocity'].data
miv_1=DTable1['min_velocity'].data
miv_2=DTable2['min_velocity'].data
miv_3=DTable3['min_velocity'].data
miv_4=DTable4['min_velocity'].data
miv_5=DTable5['min_velocity'].data
av_1_err=DTable1['rot_vel_avg_error'].data
av_2_err=DTable2['rot_vel_avg_error'].data
av_3_err=DTable3['rot_vel_avg_error'].data
av_4_err=DTable4['rot_vel_avg_error'].data
av_5_err=DTable5['rot_vel_avg_error'].data
mav_1_err=DTable1['max_velocity_error'].data
mav_2_err=DTable2['max_velocity_error'].data
mav_3_err=DTable3['max_velocity_error'].data
mav_4_err=DTable4['max_velocity_error'].data
mav_5_err=DTable5['max_velocity_error'].data
miv_1_err=DTable1['min_velocity_error'].data
miv_2_err=DTable2['min_velocity_error'].data
miv_3_err=DTable3['min_velocity_error'].data
miv_4_err=DTable4['min_velocity_error'].data
miv_5_err=DTable5['min_velocity_error'].data
########################################################################################################################

########################################################################################################################
#Plotting all the data
'''
plt.plot(r_1,av_1,'b*',label='Average')
plt.plot(r_1,mav_1,'r*',label='Maximum')
plt.plot(r_1,np.abs(miv_1),'g*',label='Minimum')
plt.legend()
plt.xlabel('$r_{dep}$ [kpc]')
plt.ylabel('$v_{rot}$ [km/s]')
plt.title('7443-12701')
#plt.show()

plt.plot(r_2,av_2,'b*',label='Average')
plt.plot(r_2,mav_2,'r*',label='Maximum')
plt.plot(r_2,np.abs(miv_2),'g*',label='Minimum')
plt.legend()
plt.xlabel('$r_{dep}$ [kpc]')
plt.ylabel('$v_{rot}$ [km/s]')
plt.title('7443-12702')
plt.show()

plt.plot(r_3,av_3,'b*',label='Average')
plt.plot(r_3,mav_3,'r*',label='Maximum')
plt.plot(r_3,np.abs(miv_3),'g*',label='Minimum')
plt.legend()
plt.xlabel('$r_{dep}$ [kpc]')
plt.ylabel('$v_{rot}$ [km/s]')
plt.title('7443-12703')
plt.show()

plt.plot(r_4,av_4,'b*',label='Average')
plt.plot(r_4,mav_4,'r*',label='Maximum')
plt.plot(r_4,np.abs(miv_4),'g*',label='Minimum')
plt.legend()
plt.xlabel('$r_{dep}$ [kpc]')
plt.ylabel('$v_{rot}$ [km/s]')
plt.title('7443-12704')
plt.show()

plt.plot(r_5,av_5,'b*',label='Average')
plt.plot(r_5,mav_5,'r*',label='Maximum')
plt.plot(r_5,np.abs(miv_5),'g*',label='Minimum')
plt.legend()
plt.xlabel('$r_{dep}$ [kpc]')
plt.ylabel('$v_{rot}$ [km/s]')
plt.title('7443-12705')
plt.show()
'''
########################################################################################################################

########################################################################################################################
#7443-12702
p2_0= [11,4,2E-2,2E2]
p2= [0.5,300,11,3.5,2E-4,5E2]

# No Bulge

# Loglike function
def loglike_Bur_ave_2(theta):
    a, b, c, d = theta
    model = v_co_Burket_nb(np.array(r_2), a, b, c, d)
    inv_sigma2 = 1.0 / (np.array(av_2_err) ** 2)

    return -0.5 * (np.sum((np.array(av_2) - model) ** 2 * inv_sigma2 - np.log(inv_sigma2)))

# Negative likelihood

nloglike_Bur_ave_2 = lambda theta: -loglike_Bur_ave_2(theta)

bestfit_av_2 = minimize(nloglike_Bur_ave_2,p2_0,bounds=((8,12),(0,10),(0,1),(0,1E2)))
bestfit_av_2

plt.plot(r_2,av_2,'b*',label='data')
plt.plot(r_2,v_co_Burket_nb(np.array(r_2),*bestfit_av_2.x),'--',label='fit')
plt.plot(r_2,v_d(np.array(r_2)*1000,10**(bestfit_av_2.x[0]),bestfit_av_2.x[1]*1000),label='disk')
plt.plot(r_2,vel_h_Burket(np.array(r_2)*1000,bestfit_av_2.x[2],bestfit_av_2.x[3]*1000),label='Burket halo')
plt.legend()
plt.show()

chi_square_ave_bur = np.zeros(len(np.array(av_2)))
for i in range(len(np.array(av_2))):
    chi_square_ave_bur[i] = ((np.array(av_2)[i] - v_co_Burket_nb(np.array(r_2),*bestfit_av_2.x)[i])/(np.array(av_2_err)[i]))**2
chi_square_bur_ave = np.sum(chi_square_ave_bur)
print(chi_square_bur_ave)

chi_square_Bur_ave_normalized = chi_square_bur_ave/(len(np.array(av_2))-4)
print(chi_square_Bur_ave_normalized)

# With Bulge

# Loglike function
def loglike_Bur_ave_2_2(theta):
    a,b,c,d,e,f = theta
    model = v_co_Burket(np.array(r_2),a,b,c,d,e,f)
    inv_sigma2 = 1.0 / (np.array(av_2_err)**2)
    
    return -0.5 * (np.sum((np.array(av_2)-model)**2 * inv_sigma2 - np.log(inv_sigma2)))

# Negative likelihood

nloglike_Bur_ave_2_2 = lambda theta: -loglike_Bur_ave_2_2(theta)

bestfit_av_2_2 = minimize(nloglike_Bur_ave_2_2,p2,bounds=((0.2,1),(100,1000),(8,12),(0,10),(0,1),(0,1E2)))
print(bestfit_av_2_2)

plt.plot(r_2,av_2,'b*',label='data')
plt.plot(r_2,v_co_Burket(np.array(r_2),*bestfit_av_2_2.x),'--',label='fit')
plt.plot(r_2,vel_b(np.array(r_2)*1000,bestfit_av_2_2.x[0],bestfit_av_2_2.x[1],bestfit_av_2_2.x[3]*1000),label='bulge')
plt.plot(r_2,v_d(np.array(r_2)*1000,10**(bestfit_av_2_2.x[2]),bestfit_av_2_2.x[3]*1000),label='disk')
plt.plot(r_2,vel_h_Burket(np.array(r_2)*1000,bestfit_av_2_2.x[4],bestfit_av_2_2.x[5]*1000),label='Burket halo')
plt.legend()
plt.show()

# Calculate the chi_2 value
chi_square_ave_bur_2 = np.zeros(len(np.array(av_2)))
for i in range(len(np.array(av_2))):
    chi_square_ave_bur_2[i] = ((np.array(av_2)[i] - v_co_Burket(np.array(r_2),*bestfit_av_2_2.x)[i])/(np.array(av_2_err)[i]))**2
chi_square_bur_ave_2 = np.sum(chi_square_ave_bur_2)
print(chi_square_bur_ave_2)

chi_square_Bur_ave_normalized_2 = chi_square_bur_ave_2/(len(np.array(av_2))-4)
print(chi_square_Bur_ave_normalized_2)

'''
p2_2= [10.7,4,1E-2,10]

# Loglike function
def loglike_Bur_mav_2(theta):
    a,b,c,d = theta
    model = v_co_Burket_nb(np.array(r_2)*1000,a,b,c,d)
    inv_sigma2 = 1.0 / (np.array(mav_2_err)**2)
    
    return -0.5 * (np.sum((np.array(mav_2)-model)**2 * inv_sigma2 - np.log(inv_sigma2)))

# Negative likelihood

nloglike_Bur_mav_2 = lambda theta: -loglike_Bur_mav_2(theta)

bestfit_mav_2 = minimize(nloglike_Bur_mav_2,p2_2,bounds=((8,12),(0.5,2),(0,12),(0,10),(0,0.1),(0,1E1)))
bestfit_mav_2

plt.plot(r_2,mav_2,'b*',label='data')
plt.plot(r_2,v_co_Burket_nb(np.array(r_2),*bestfit_mav_2.x),'--',label='fit')
plt.plot(r_2,v_d(np.array(r_2)*1000,10**(bestfit_mav_2.x[0]),bestfit_mav_2.x[1]*1000),label='disk')
plt.plot(r_2,vel_h_Burket(np.array(r_2)*1000,bestfit_mav_2.x[2],bestfit_mav_2.x[3]*1000),label='Burket halo')
plt.legend()
'''
'''
# Loglike function
def loglike_Bur_miv_2(theta):
    a,b,c,d = theta
    model = v_co_Burket_nb(np.array(r_2)*1000,a,b,c,d)
    inv_sigma2 = 1.0 / (np.array(miv_2_err)**2)
    
    return -0.5 * (np.sum((np.array(np.abs(miv_2))-model)**2 * inv_sigma2 - np.log(inv_sigma2)))

# Negative likelihood

nloglike_Bur_miv_2 = lambda theta: -loglike_Bur_miv_2(theta)

bestfit_miv_2 = minimize(nloglike_Bur_miv_2,p2_2,bounds=((0,12),(0,10),(0,0.1),(0,1E1)))
bestfit_miv_2

plt.plot(r_2,np.abs(miv_2),'b*',label='data')
plt.plot(r_2,v_co_Burket_nb(np.array(r_2),*bestfit_miv_2.x),'--',label='fit')
plt.plot(r_2,v_d(np.array(r_2)*1000,10**(bestfit_miv_2.x[0]),bestfit_miv_2.x[1]*1000),label='disk')
plt.plot(r_2,vel_h_Burket(np.array(r_2)*1000,bestfit_miv_2.x[2],bestfit_miv_2.x[3]*1000),label='Burket halo')
plt.legend()
'''
########################################################################################################################

########################################################################################################################
#7443-12704
'''
# With Bulge
# Bounds 
p4 = [0.5,200,10,2.99,2E-2,5E3]

# Loglike function
def loglike_Bur_ave_4(theta):
    a,b,c,d,e,f = theta
    model = v_co_Burket(np.array(r_4)*1000,a,b,c,d,e,f)
    inv_sigma2 = 1.0 / (np.array(av_4_err)**2)
    
    return -0.5 * (np.sum((np.array(av_4)-model)**2 * inv_sigma2 - np.log(inv_sigma2)))

# Negative likelihood

nloglike_Bur_ave_4 = lambda theta: -loglike_Bur_ave_4(theta)

bestfit_av_4 = minimize(nloglike_Bur_ave_4,p4,bounds=((0.2,1),(100,1000),(8,12),(0,10),(0,1),(0,1E2)))
print(bestfit_av_4)

plt.plot(r_4,av_4,'b*',label='data')
plt.plot(r_4,v_co_Burket(np.array(r_4),*bestfit_av_4.x),'--',label='fit')
plt.plot(r_4,vel_b(np.array(r_4)*1000,bestfit_av_4.x[0],bestfit_av_4.x[1],bestfit_av_4.x[3]*1000),label='bulge')
plt.plot(r_4,v_d(np.array(r_4)*1000,10**(bestfit_av_4.x[2]),bestfit_av_4.x[3]*1000),label='disk')
plt.plot(r_4,vel_h_Burket(np.array(r_4)*1000,bestfit_av_4.x[4],bestfit_av_4.x[5]*1000),label='Burket halo')
plt.legend()
plt.show()

# No bulge
p4_2 = [10,2.99,2E-2,5E1]


# Loglike function
def loglike_Bur_ave_4_2(theta):
    a, b, c, d = theta
    model = v_co_Burket_nb(np.array(r_4) * 1000, a, b, c, d)
    inv_sigma2 = 1.0 / (np.array(av_4_err) ** 2)

    return -0.5 * (np.sum((np.array(av_4) - model) ** 2 * inv_sigma2 - np.log(inv_sigma2)))

# Negative likelihood

nloglike_Bur_ave_4_2 = lambda theta: -loglike_Bur_ave_4_2(theta)

bestfit_av_4_2 = minimize(nloglike_Bur_ave_4_2,p4_2,bounds=((8,12),(0,10),(0,1),(0,1E2)))
print(bestfit_av_4_2)

plt.plot(r_4,av_4,'b*',label='data')
plt.plot(r_4,v_co_Burket_nb(np.array(r_4),*bestfit_av_4_2.x),'--',label='fit')
plt.plot(r_4,v_d(np.array(r_4)*1000,10**(bestfit_av_4_2.x[0]),bestfit_av_4_2.x[1]*1000),label='disk')
plt.plot(r_4,vel_h_Burket(np.array(r_4)*1000,bestfit_av_4_2.x[2],bestfit_av_4_2.x[3]*1000),label='Burket halo')
plt.legend()
plt.show()
'''
########################################################################################################################

########################################################################################################################
#7443-12705
# Bounds
p5= [10.77897,4,2E-2,2E1]

# Loglike function
def loglike_Bur_ave_5(theta):
    a,b,c,d = theta
    model = v_co_Burket_nb(np.array(r_5),a,b,c,d)
    inv_sigma2 = 1.0 / (np.array(av_5_err)**2)
    
    return -0.5 * (np.sum((np.array(av_5)-model)**2 * inv_sigma2 - np.log(inv_sigma2)))

# Negative likelihood

nloglike_Bur_ave_5 = lambda theta: -loglike_Bur_ave_5(theta)

bestfit_av_5 = minimize(nloglike_Bur_ave_5,p5,bounds=((8,12),(0,10),(0,1),(0,1E2)))
bestfit_av_5

plt.plot(r_5,av_5,'b*',label='data')
plt.plot(r_5,v_co_Burket_nb(np.array(r_5),*bestfit_av_5.x),'--',label='fit')
plt.plot(r_5,v_d(np.array(r_5)*1000,10**(bestfit_av_5.x[0]),bestfit_av_5.x[1]*1000),label='disk')
plt.plot(r_5,vel_h_Burket(np.array(r_5)*1000,bestfit_av_5.x[2],bestfit_av_5.x[3]*1000),label='Burket halo')
plt.legend()
plt.xlabel('$r_{dep}$ [kpc]')
plt.ylabel('$v_{rot}$ [km/s]')
plt.show()

# Bounds
p5_2= [10.77897,4,2E-2,1.5E1]

# Loglike function
def loglike_Bur_mav_5(theta):
    a,b,c,d = theta
    model = v_co_Burket_nb(np.array(r_5)*1000,a,b,c,d)
    inv_sigma2 = 1.0 / (np.array(mav_5_err)**2)
    
    return -0.5 * (np.sum((np.array(mav_5)-model)**2 * inv_sigma2 - np.log(inv_sigma2)))

# Negative likelihood

nloglike_Bur_mav_5 = lambda theta: -loglike_Bur_mav_5(theta)

bestfit_mav_5 = minimize(nloglike_Bur_mav_5,p5_2,bounds=((0,12),(0,10),(0,1),(0,1E2)))
bestfit_mav_5

plt.plot(r_5,mav_5,'b*',label='data')
plt.plot(r_5,v_co_Burket_nb(np.array(r_5),*bestfit_mav_5.x),'--',label='fit')
plt.plot(r_5,v_d(np.array(r_5)*1000,10**(bestfit_mav_5.x[0]),bestfit_mav_5.x[1]*1000),label='disk')
plt.plot(r_5,vel_h_Burket(np.array(r_5)*1000,bestfit_mav_5.x[2],bestfit_mav_5.x[3]*1000),label='Burket halo')
plt.legend()
plt.xlabel('$r_{dep}$ [kpc]')
plt.ylabel('$v_{rot}$ [km/s]')
plt.show()

# Loglike function
def loglike_Bur_miv_5(theta):
    a,b,c,d = theta
    model = v_co_Burket_nb(np.array(r_5)*1000,a,b,c,d)
    inv_sigma2 = 1.0 / (np.array(miv_5_err)**2)
    
    return -0.5 * (np.sum((np.array(np.abs(miv_5))-model)**2 * inv_sigma2 - np.log(inv_sigma2)))

# Negative likelihood

nloglike_Bur_miv_5 = lambda theta: -loglike_Bur_miv_5(theta)

bestfit_miv_5 = minimize(nloglike_Bur_miv_5,p5,bounds=((0,12),(0,10),(0,1),(0,1E2)))
bestfit_miv_5

plt.plot(r_5,np.abs(miv_5),'b*',label='data')
plt.plot(r_5,v_co_Burket_nb(np.array(r_5),*bestfit_miv_5.x),'--',label='fit')
plt.plot(r_5,v_d(np.array(r_5)*1000,10**bestfit_miv_5.x[0],bestfit_miv_5.x[1]*1000),label='disk')
plt.plot(r_5,vel_h_Burket(np.array(r_5)*1000,bestfit_miv_5.x[2],bestfit_miv_5.x[3]*1000),label='Burket halo')
plt.legend()
plt.xlabel('$r_{dep}$ [kpc]')
plt.ylabel('$v_{rot}$ [km/s]')
plt.show()
########################################################################################################################

########################################################################################################################
'''
def loglike_Bur(theta,r,vel,vel_err):
    a,b,c,d = theta
    model = v_co_Burket_nb(r*1000,a,b,c,d)
    model[model<=0] = np.finfo(dtype=np.float64).tiny
    inv_sigma2 = 1.0 / (vel_err**2)
    return -0.5 * (np.sum((vel-model)**2 * inv_sigma2 - np.log(inv_sigma2)))

# Random Distribution Functions

def uniform(a, b, u):
    """Given u in [0,1], return a uniform number in [a,b]."""
    return a + (b-a)*u
def jeffreys(a, b, u):
    #"""Given u in [0,1], return a Jeffreys random number in [a,b]."""
    return a**(1-u) * b**u

def ptform_Bur(u):
    """
    Priors for the 4 parameters of Burket rotation curve model. 
    Required by the dynesty sampler.
    Parameters
    ----------
    u : ndarray
        Array of uniform random numbers between 0 and 1.
    Returns
    -------
    priors : ndarray
        Transformed random numbers giving prior ranges on model parameters.
    """
    M_disk = uniform(1E8,1E11,u[0])
    R_disk = uniform(2, 10,u[1])
    Rho_halo = uniform(5E-4,5E-2,u[2])
    #R_halo = jeffreys(9E2,1E7,u[3])
    R_halo = jeffreys(10,500,u[3])
    return M_disk, R_disk, Rho_halo, R_halo
'''
########################################################################################################################

########################################################################################################################
#Running Dynamic Nested Sampler with the Burket Model
'''
dsampler = dynesty.DynamicNestedSampler(loglike_Bur, ptform_Bur, ndim=4,
                                        logl_args=(np.array(r_2), 
                                                   np.array(av_2), 
                                                   np.array(av_2_err)),
                                        nlive=2000,
                                        bound='multi',
                                        sample='auto')
dsampler.run_nested()
dres1 = dsampler.results

labels = ['$M_{disk}$', '$R_{disk}$', '$rho_{hc}$','$R_{halo}$']
nParams = len(labels)
fig, axes = dyplot.cornerplot(dres1, smooth=0.03,
                              labels=labels,
                              show_titles=True,
                              quantiles_2d=[1-np.exp(-0.5*r**2) for r in [1.,2.,3]],
                              quantiles=[0.16, 0.5, 0.84],
                              fig=plt.subplots(nParams, nParams, figsize=(2.5*nParams, 2.6*nParams)),
                              color='#1f77d4')

dsampler2 = dynesty.DynamicNestedSampler(loglike_Bur, ptform_Bur, ndim=4,
                                        logl_args=(np.array(r_2), 
                                                   np.array(mav_2), 
                                                   np.array(mav_2_err)),
                                        nlive=2000,
                                        bound='multi',
                                        sample='auto')
dsampler2.run_nested()
dres2 = dsampler.results

labels = ['$M_{disk}$', '$R_{disk}$', '$rho_{hc}$','$R_{halo}$']
nParams = len(labels)
fig, axes = dyplot.cornerplot(dres2, smooth=0.03,
                              labels=labels,
                              show_titles=True,
                              quantiles_2d=[1-np.exp(-0.5*r**2) for r in [1.,2.,3]],
                              quantiles=[0.16, 0.5, 0.84],
                              fig=plt.subplots(nParams, nParams, figsize=(2.5*nParams, 2.6*nParams)),
                              color='#1f77d4')

dsampler3 = dynesty.DynamicNestedSampler(loglike_Bur, ptform_Bur, ndim=4,
                                        logl_args=(np.array(r_2), 
                                                   np.array(miv_2), 
                                                   np.array(miv_2_err)),
                                        nlive=2000,
                                        bound='multi',
                                        sample='auto')
dsampler3.run_nested()
dres3 = dsampler.results

labels = ['$M_{disk}$', '$R_{disk}$', '$rho_{hc}$','$R_{halo}$']
nParams = len(labels)
fig, axes = dyplot.cornerplot(dres3, smooth=0.03,
                              labels=labels,
                              show_titles=True,
                              quantiles_2d=[1-np.exp(-0.5*r**2) for r in [1.,2.,3]],
                              quantiles=[0.16, 0.5, 0.84],
                              fig=plt.subplots(nParams, nParams, figsize=(2.5*nParams, 2.6*nParams)),
                              color='#1f77d4')
'''
########################################################################################################################



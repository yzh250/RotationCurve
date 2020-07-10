################################################################################
# All the libraries used & constant values
#-------------------------------------------------------------------------------
import numpy as np

import matplotlib.pyplot as plt

from astropy.table import QTable

from scipy.optimize import minimize

from velocity_functions import nloglike_Bur_nb, v_co_Burket_nb, v_d, vel_h_Burket

import dynesty
from dynesty import plotting as dyplot
################################################################################


################################################################################
# Reading in all the objects with Plate No. 7443
#-------------------------------------------------------------------------------
galaxy_ID = '7443-12705'

rot_curve_data_filename = galaxy_ID + '_rot_curve_data.txt'

DTable = QTable.read(rot_curve_data_filename, format='ascii.ecsv')
################################################################################


################################################################################
# Reading in the radii, velocity data
#-------------------------------------------------------------------------------
r = DTable['deprojected_distance'].data
av = DTable['rot_vel_avg'].data
mav = DTable['max_velocity'].data
miv = DTable['min_velocity'].data
av_err = DTable['rot_vel_avg_error'].data
mav_err = DTable['max_velocity_error'].data
miv_err = DTable['min_velocity_error'].data
################################################################################


################################################################################
# Plotting all the data
#-------------------------------------------------------------------------------
plt.errorbar(r, av, yerr=av_err, fmt='g.', label='Average')
plt.errorbar(r, mav, yerr=mav_err, fmt='r*', label='Maximum')
plt.errorbar(r, np.abs(miv), yerr=miv_err, fmt='b^', label='Minimum')
plt.legend()
plt.xlabel('$r_{dep}$ [kpc]')
plt.ylabel('$v_{rot}$ [km/s]')
plt.title(galaxy_ID)
plt.show()
################################################################################


################################################################################
# Fit rotation curves
#-------------------------------------------------------------------------------
# Average rotation curve
#-------------------------------------------------------------------------------
# Initial guesses
p0_avg = [10.77897, 4, 0.02, 20]

bestfit_av = minimize(nloglike_Bur_nb, p0_avg, args=(r, av, av_err), 
                        bounds=((8, 12), (0.1, 100), (0.001, 10), (0.1, 1000)))
print('---------------------------------------------------')
print('Average curve')
print(bestfit_av)

r_normalized = r/bestfit_av.x[1]

plt.errorbar(r_normalized, av, yerr=av_err, fmt='g*', label='Average')
plt.plot(r_normalized, v_co_Burket_nb(np.array(r), bestfit_av.x), '--', label='fit')
plt.plot(r_normalized, v_d(np.array(r)*1000, 10**bestfit_av.x[0], bestfit_av.x[1]*1000), label='disk')
plt.plot(r_normalized, vel_h_Burket(np.array(r)*1000, bestfit_av.x[2], bestfit_av.x[3]*1000),label='Burket halo')
plt.legend()
plt.xlabel('$r_{dep}$/$R_d$')
plt.ylabel('$v_{rot}$ [km/s]')
plt.title(galaxy_ID)
plt.show()
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Positive rotation curve
#-------------------------------------------------------------------------------
# Initial guesses
p0 = [10.77897, 4, 0.02, 15]

bestfit_mav = minimize(nloglike_Bur_nb, p0, args=(r, mav, mav_err), 
                         bounds=((8, 12), (0.1, 10), (0.001, 10), (0.1, 1000)))
print('---------------------------------------------------')
print('Positive curve')
print(bestfit_mav)

plt.errorbar(r, mav, yerr=mav_err, fmt='r*', label='data')
plt.plot(r, v_co_Burket_nb(np.array(r), bestfit_mav.x), '--', label='fit')
plt.plot(r, v_d(np.array(r)*1000, 10**(bestfit_mav.x[0]), bestfit_mav.x[1]*1000), label='disk')
plt.plot(r, vel_h_Burket(np.array(r)*1000, bestfit_mav.x[2], bestfit_mav.x[3]*1000), label='Burket halo')
plt.legend()
plt.xlabel('$r_{dep}$ [kpc]')
plt.ylabel('$v_{rot}$ [km/s]')
plt.title(galaxy_ID)
plt.show()
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Negative rotation curve
#-------------------------------------------------------------------------------
bestfit_miv = minimize(nloglike_Bur_nb, p0, args=(r, miv, miv_err), 
                         bounds=((8, 12), (0.1, 10), (0.001, 10), (0.1, 1000)))
print('---------------------------------------------------')
print('Negative curve')
print(bestfit_miv)

plt.errorbar(r, np.abs(miv), yerr=miv_err, fmt='b*', label='data')
plt.plot(r, v_co_Burket_nb(np.array(r), bestfit_miv.x), '--', label='fit')
plt.plot(r, v_d(np.array(r)*1000, 10**bestfit_miv.x[0], bestfit_miv.x[1]*1000), label='disk')
plt.plot(r, vel_h_Burket(np.array(r)*1000, bestfit_miv.x[2], bestfit_miv.x[3]*1000), label='Burket halo')
plt.legend()
plt.xlabel('$r_{dep}$ [kpc]')
plt.ylabel('$v_{rot}$ [km/s]')
plt.title(galaxy_ID)
plt.show()
################################################################################

'''
################################################################################
# 7443-12702
#-------------------------------------------------------------------------------
p2 = [11, 4, 0.02, 200]
p2_bulge = [0.5, 300, 11, 3.5, 0.0004, 500]

# No Bulge
bestfit_av_2 = minimize(nloglike_Bur_nb, p2, args=(r_2, av_2, av_2_err), 
                        bounds=((8,12), (0.1, 10), (0.001, 10), (0.1, 1000)))
print('-------------------------------------------------')
print(bestfit_av_2)

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
########################################################################################################################
'''
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



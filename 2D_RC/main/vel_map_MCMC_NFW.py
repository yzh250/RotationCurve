################################################################################
# Import modules
#-------------------------------------------------------------------------------
#import matplotlib.pyplot as plt

import numpy as np

import emcee
#import corner

#import pickle

from Velocity_Map_Functions import loglikelihood_NFW_flat_constraints

from RC_2D_Fit_Functions import Galaxy_Data
################################################################################




################################################################################
# Constants
#-------------------------------------------------------------------------------
G = 6.674E-11  # m^3 kg^-1 s^-2
Msun = 1.989E30  # kg

# Specific for galaxy 7443-6101
scale = 0.22471093                                           
################################################################################




################################################################################
# Data location
#-------------------------------------------------------------------------------
#manga = '/home/yzh250/Documents/UR_Stuff/Research_UR/SDSS/dr16/manga/spectro/'
manga =  '/Users/richardzhang/Documents/UR_Stuff/Research_UR/SDSS/dr16/manga/spectro/'
#manga = '/Users/kellydouglass/Documents/Research/data/SDSS/dr16/manga/spectro/'
################################################################################




################################################################################
# Import galaxy data
#-------------------------------------------------------------------------------
data_maps, gshape, x_center_guess, y_center_guess = Galaxy_Data('7443-6101', 
                                                                manga)
################################################################################




################################################################################
# Define MCMC functions
#-------------------------------------------------------------------------------
def log_prior(params):

    log_rhob0,Rb,SigD,Rd,log_rhoh0,Rh,inclination,phi,center_x,center_y,vsys = params

    logP = 0

    if -7 < log_rhob0 < 2 and 0 < Rb < 5 and 100 < SigD < 3000 and 1 < Rd < 30\
     and -7 < log_rhoh0 < 2 and 0.01 < Rh< 500 and 0 < inclination < np.pi*0.436 and 0 < phi < 2*np.pi\
     and 10 < center_x < 50 and 10 < center_y < 50 and -100 < vsys < 100:
        logP = 0

    # setting constraints on the radii
    elif Rh < Rb or Rh < Rd or Rd < Rb:
        logP = -np.inf
    else:
    	logP = -np.inf

    return logP



def log_prob_NFW(params, scale, shape, vdata, ivar, mask):

    lp = log_prior(params)

    logL = loglikelihood_NFW_flat_constraints(params, 
                                              scale, 
                                              shape, 
                                              vdata.compressed(), 
                                              ivar.compressed(), 
                                              mask)

    if not np.isfinite(lp) or not np.isfinite(logL):
        return -np.inf 
    else:
        return lp + logL
################################################################################




################################################################################
# Best-fit parameter values from scipy.optimize.minimize
#-------------------------------------------------------------------------------
'''
mini_soln = [np.log10(5.315237789),
             0.1472824,
             417.348003,
             11.71427151,
             0.003456733,
             20.53074275,
             0.865871265,
             1.969515296,
             25.83823628,
             27.65695241,
             4.516715936]
'''
mini_soln = [-1.55039138e+00,  
             2.60121905e+00,  
             5.41233872e+02,  
             1.45222297e+00,
             -3.65723641e+00,  
             3.81443437e+02,  
             5.92785613e-01,  
             1.95488595e+00,
             2.64660673e+01,  
             2.75112718e+01,  
             2.51683515e-01]
################################################################################




################################################################################
# NFW
#-------------------------------------------------------------------------------
pos = np.array(mini_soln) + np.random.uniform(low=-1e-3*np.ones(len(mini_soln)), 
                                              high=1e-3*np.ones(len(mini_soln)), 
                                              size=(64,11))
#pos = np.random.uniform(low=[-6,0.00001,200,0.1,2e-5,0.1,0,0,15,15,-50], 
#                        high=[2,5,2500,25,0.1,500,0.436*np.pi,2*np.pi,45,45,50], 
#                        size=(64,11))

nwalkers, ndim = pos.shape

bad_sampler_NFW = emcee.EnsembleSampler(nwalkers, 
                                        ndim, 
                                        log_prob_NFW, 
                                        args=(scale, 
                                              gshape, 
                                              data_maps['vmasked'], 
                                              data_maps['ivar_masked'], 
                                              data_maps['Ha_vel_mask']))
bad_sampler_NFW.run_mcmc(pos, 10000, progress=True)
bad_samples_NFW = bad_sampler_NFW.get_chain(discard=500)

np.save('bad_samples_NFW.npy', bad_samples_NFW)

good_walkers_NFW = bad_sampler_NFW.acceptance_fraction > 0
np.save('good_walkers_NFW.npy', good_walkers_NFW)

'''
labels = ['rho_b','R_b', 'Sigma_d','R_d','rho_h','R_h','i','phi','x','y','vsys']

for i in range(ndim):
    ax = axes_NFW[i]
    ax.plot(bad_samples_NFW[:5000,:,i], 'k', alpha=0.3)
    ax.set(xlim=(0,5000), ylabel=labels[i])
    ax.yaxis.set_label_coords(-0.11, 0.5)

axes_NFW[-1].set_xlabel('step number')
#fig_NFW.tight_layout()
plt.savefig('mcmc_NFW.png',format='png')
plt.close()
####################################################################

####################################################################
bad_samples_NFW = bad_sampler_NFW.get_chain(discard=500)[:,good_walkers_NFW,:]
np.save('bad_samples_NFW.npy',bad_samples_NFW)
ns_NFW, nw_NFW, nd_NFW = bad_samples_NFW.shape
flat_bad_samples_NFW = bad_samples_NFW.reshape(ns_NFW*nw_NFW, nd_NFW)
np.save('flat_bad_samples_NFW.npy',flat_bad_samples_NFW)
flat_bad_samples_NFW.shape
####################################################################

####################################################################
corner.corner(flat_bad_samples_NFW, labels=labels,
                    range=[(-7,2), (0,5), (0,2000),(1,20),(0.0001,0.01),(5,200),(0,np.pi/2),(0,1.5),(30,40),(30,40),(-100,100)], bins=30, #smooth=1,
                    truths=[np.log10(0.05812451),3.601276359,385.2756031,6.748078457,0.002449669,30.24921674,1.080172553,0.69825044,36.61004742,37.67680252,11.81343922], truth_color='#ff4444',
                    levels=(1-np.exp(-0.5), 1-np.exp(-2)), 
                    quantiles=(0.16, 0.84),
                    hist_kwargs={'histtype':'stepfilled', 'alpha':0.3, 'density':True},
                    color='blue', plot_datapoints=False,
                    fill_contours=True)
plt.savefig('corner_NFW.png',format='png')
plt.close()
####################################################################

for i, label in enumerate(labels):
    x = mini_soln[i]
    x16, x84 = np.percentile(flat_bad_samples_iso[:,i], [16,84])
    dlo = x - x16
    dhi = x84 - x
    print('{:3s} = {:5.2f} + {:4.2f} - {:4.2f}'.format(label, x, dhi, dlo))
    print('    = ({:5.2f}, {:5.2f})'.format(x16, x84))

####################################################################
# Dumping out put
#out_directory = '/Users/richardzhang/Documents/UR_Stuff/Research_UR/2D_RC/MCMC_folder/'
#temp_outfile = open(out_directory + 'results.pickle', 'wb')
#pickle.dump((flat_bad_samples_iso, flat_bad_samples_NFW, flat_bad_samples_bur), temp_outfile)
#temp_outfile.close()
'''



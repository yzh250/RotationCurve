################################################################################
# Import modules
#-------------------------------------------------------------------------------
#import matplotlib.pyplot as plt

import numpy as np

import emcee
#import corner

#import pickle

from Velocity_Map_Functions import loglikelihood_bur_flat_constraints

from RC_2D_Fit_Functions import Galaxy_Data
################################################################################




################################################################################
# Constants
#-------------------------------------------------------------------------------
G = 6.674E-11  # m^3 kg^-1 s^-2
Msun = 1.989E30  # kg

# Specific for galaxy 7443-6101
scale = 0.2101665333430296                                        
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
data_maps, gshape, x_center_guess, y_center_guess = Galaxy_Data('7495-12704', 
                                                                manga)
################################################################################




################################################################################
# Define MCMC functions
#-------------------------------------------------------------------------------
def log_prior(params):

    log_rhob0,Rb,SigD,Rd,log_rhoh0,Rh,inclination,phi,center_x,center_y,vsys = params

    logP = 0

    rhob_check = -7 < log_rhob0 < 1
    #rhob_check = 0 < log_rhob0 < 10
    Rb_check = 0 < Rb < 5

    SigD_check = 0.1 < SigD < 3000
    Rd_check = 0.1 < Rd < 30

    rhoh_check = -7 < log_rhoh0 < 2
    #rhoh_check = 0 < log_rhoh0 < 100
    Rh_check = 0.01 < Rh < 500

    i_check = 0 < inclination < np.pi*0.436
    phi_check = 0 < phi < 2*np.pi

    x_check = 10 < center_x < 50
    y_check = 10 < center_y < 50

    v_check = -100 < vsys < 100

    if rhob_check and Rb_check and SigD_check and Rd_check and rhoh_check and Rh_check and i_check and phi_check and x_check and y_check and v_check:
        logP = 0

    # setting constraints on the radii
    elif Rh < Rb or Rh < Rd or Rd < Rb:
        logP = -np.inf
    else:
    	logP = -np.inf

    return logP



def log_prob_bur(params, scale, shape, vdata, ivar, mask):

    lp = log_prior(params)

    logL = loglikelihood_bur_flat_constraints(params, 
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
# 7443-12705
mini_soln = [np.log10(5.36E-05),
             2.811046162,
             978.7934831,
             6.493085395,
             4.10E-05,
             999.8669552,
             0.858228903,
             0.752910577,
             38.25051586,
             37.23417255,
             -0.685352448]
'''
mini_soln = [0.999956293,
             4.469934154,
             2999.999855,
             7.423787826,
             0.048158806,
             127.6522708,
             0.571032036,
             2.018775432,
             41.30157734,
             7.728779602,
             99.99993356]
################################################################################




################################################################################
# Burket
#-------------------------------------------------------------------------------
pos = np.array(mini_soln) + np.random.uniform(low=-1e-3*np.ones(len(mini_soln)), 
                                              high=1e-3*np.ones(len(mini_soln)), 
                                              size=(64,11))
#pos = np.random.uniform(low=[-6,0.00001,200,0.1,2e-5,0.1,0,0,15,15,-50], 
#                        high=[2,5,2500,25,0.1,500,0.436*np.pi,2*np.pi,45,45,50], 
#                        size=(64,11))

nwalkers, ndim = pos.shape

bad_sampler_bur = emcee.EnsembleSampler(nwalkers, 
                                        ndim, 
                                        log_prob_bur, 
                                        args=(scale, 
                                              gshape, 
                                              data_maps['vmasked'], 
                                              data_maps['ivar_masked'], 
                                              data_maps['Ha_vel_mask']))
bad_sampler_bur.run_mcmc(pos, 10000, progress=True)
bad_samples_bur = bad_sampler_bur.get_chain()
#bad_samples_bur = bad_sampler_bur.get_chain(discard=500)

np.save('bad_samples_bur.npy', bad_samples_bur)

good_walkers_bur = bad_sampler_bur.acceptance_fraction > 0
np.save('good_walkers_bur.npy', good_walkers_bur)

'''
labels = ['rho_b','R_b', 'Sigma_d','R_d','rho_h','R_h','i','phi','x','y','vsys']

for i in range(ndim):
    ax = axes_bur[i]
    ax.plot(bad_samples_bur[:5000,:,i], 'k', alpha=0.3)
    ax.set(xlim=(0,5000), ylabel=labels[i])
    ax.yaxis.set_label_coords(-0.11, 0.5)

axes_bur[-1].set_xlabel('step number')
#fig_bur.tight_layout()
plt.savefig('mcmc_bur.png',format='png')
plt.close()
####################################################################

####################################################################
bad_samples_bur = bad_sampler_bur.get_chain(discard=500)[:,good_walkers_bur,:]
np.save('bad_samples_bur.npy',bad_samples_bur)
ns_bur, nw_bur, nd_bur = bad_samples_bur.shape
flat_bad_samples_bur = bad_samples_bur.reshape(ns_bur*nw_bur, nd_bur)
np.save('flat_bad_samples_bur.npy',flat_bad_samples_bur)
flat_bad_samples_bur.shape
####################################################################

####################################################################
corner.corner(flat_bad_samples_bur, labels=labels,
                    range=[(0,100), (0,5), (0,2000),(1,20),(0.0001,0.01),(5,200),(0,np.pi/2),(0,1.5),(30,40),(30,40),(-100,100)], bins=30, #smooth=1,
                    truths=[np.log10(5.36E-05),2.811046162,978.7934831,6.493085395,4.10E-05,999.8669552,0.858228903,0.752910577,38.25051586,37.23417255,-0.685352448], truth_color='#ff4444',
                    levels=(1-np.exp(-0.5), 1-np.exp(-2)), 
                    quantiles=(0.16, 0.84),
                    hist_kwargs={'histtype':'stepfilled', 'alpha':0.3, 'density':True},
                    color='blue', plot_datapoints=False,
                    fill_contours=True)
plt.savefig('corner_bur.png',format='png')
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



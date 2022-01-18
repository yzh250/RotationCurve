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

# scaling for different galaxies
scale_7443_6101 = 0.224801833                                     
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
data_maps_7443_6101, gshape_7443_6101, x_center_guess_7443_6101, y_center_guess_7443_6101 = Galaxy_Data('7443-6101', 
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

# 7443-6101

initial_guesses_7443_6101 = [-1, 1, 1000, 4, -3, 25, 0.7488906714558082, 1.7935701525194527, 34, 24, 0]

model_guesses_7443_6101 = [-1, 1, 1000, 4, -3, 25]

geo_guesses_7443_6101 = [0.7488906714558082, 1.7935701525194527, 34, 24, 0]

################################################################################




################################################################################
# Burket
#-------------------------------------------------------------------------------
#pos = np.array(mini_soln) + np.random.uniform(low=-1e-3*np.ones(len(mini_soln)), 
#                                             high=1e-3*np.ones(len(mini_soln)), 
#                                              size=(64,11))

# 7443-6101

pos_rand_7443_6101 = np.random.uniform(low=[-7,0,0.1,0.1,-7,0.001,0,0,10,10,-100], 
                        high=[1,5,3000,30,-2,500,0.436*np.pi,2*np.pi,50,50,100], 
                        size=(64,11))

# Seeding around initial guess

pos_init_7443_6101 = initial_guesses_7443_6101 + np.random.uniform(np.random.uniform(low=-1e-3*np.ones(len(initial_guesses_7443_6101)), 
                                              high=1e-3*np.ones(len(initial_guesses_7443_6101)), 
                                              size=(64,len(initial_guesses_7443_6101))))

# Combined
pos_model_7443_6101 = np.random.uniform(low=[-7,0.00001,200,0.1,2e-5,0.1], 
                        high=[1,5,2500,25,0.1,500], 
                        size=(64,6))

pos_geo_7443_6101 = np.array(geo_guesses_7443_6101) + np.random.uniform(np.random.uniform(low=-1e-3*np.ones(len(geo_guesses_7443_6101)), 
                                              high=1e-3*np.ones(len(geo_guesses_7443_6101)), 
                                              size=(64,len(geo_guesses_7443_6101))))

pos_combined_7443_6101 = np.column_stack((pos_model_7443_6101,pos_geo_7443_6101))

# random walker

nwalkers, ndim = pos_rand_7443_6101.shape

bad_sampler_bur = emcee.EnsembleSampler(nwalkers, 
                                        ndim, 
                                        log_prob_bur, 
                                        args=(scale_7443_6101, 
                                              gshape_7443_6101, 
                                              data_maps_7443_6101['vmasked'], 
                                              data_maps_7443_6101['ivar_masked'], 
                                              data_maps_7443_6101['Ha_vel_mask']))

bad_sampler_bur.run_mcmc(pos_rand_7443_6101, 10000, progress=True)
bad_samples_bur = bad_sampler_bur.get_chain()
#bad_samples_bur = bad_sampler_bur.get_chain(discard=500)

np.save('bad_samples_bur_7443_6101_rand.npy', bad_samples_bur)

good_walkers_bur = bad_sampler_bur.acceptance_fraction > 0
np.save('good_walkers_bur_7443_6101_rand.npy', good_walkers_bur)

# seeding around initial guess

nwalkers, ndim = pos_init_7443_6101.shape

bad_sampler_bur = emcee.EnsembleSampler(nwalkers, 
                                        ndim, 
                                        log_prob_bur, 
                                        args=(scale_7443_6101, 
                                              gshape_7443_6101, 
                                              data_maps_7443_6101['vmasked'], 
                                              data_maps_7443_6101['ivar_masked'], 
                                              data_maps_7443_6101['Ha_vel_mask']))

bad_sampler_bur.run_mcmc(pos_init_7443_6101, 10000, progress=True)
bad_samples_bur = bad_sampler_bur.get_chain()
#bad_samples_bur = bad_sampler_bur.get_chain(discard=500)

np.save('bad_samples_bur_7443_6101_init.npy', bad_samples_bur)

good_walkers_bur = bad_sampler_bur.acceptance_fraction > 0
np.save('good_walkers_bur_7443_6101_init.npy', good_walkers_bur)

# Combined

nwalkers, ndim = pos_combined_7443_6101.shape

bad_sampler_bur = emcee.EnsembleSampler(nwalkers, 
                                        ndim, 
                                        log_prob_bur, 
                                        args=(scale_7443_6101, 
                                              gshape_7443_6101, 
                                              data_maps_7443_6101['vmasked'], 
                                              data_maps_7443_6101['ivar_masked'], 
                                              data_maps_7443_6101['Ha_vel_mask']))

bad_sampler_bur.run_mcmc(pos_combined_7443_6101, 10000, progress=True)
bad_samples_bur = bad_sampler_bur.get_chain()
#bad_samples_bur = bad_sampler_bur.get_chain(discard=500)

np.save('bad_samples_bur_7443_6101_comb.npy', bad_samples_bur)

good_walkers_bur = bad_sampler_bur.acceptance_fraction > 0
np.save('good_walkers_bur_7443_6101_comb.npy', good_walkers_bur)


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
#pickle.dump((flat_bad_samples_iso, flat_bad_samples_bur
, flat_bad_samples_bur), temp_outfile)
#temp_outfile.close()
'''



################################################################################
# Import modules
#-------------------------------------------------------------------------------
#import matplotlib.pyplot as plt

import numpy as np

import emcee
#import corner

#import pickle

from Velocity_Map_Functions import loglikelihood_iso_flat_constraints,\
                                   loglikelihood_NFW_flat_constraints,\
                                   loglikelihood_bur_flat_constraints

from RC_2D_Fit_Functions import Galaxy_Data
################################################################################




################################################################################
# Constants
#-------------------------------------------------------------------------------
G = 6.674E-11  # m^3 kg^-1 s^-2
Msun = 1.989E30  # kg

# scaling for different galaxies
scale = 0.586842411
################################################################################




################################################################################
# Data location
#-------------------------------------------------------------------------------
#manga = '/home/yzh250/Documents/UR_Stuff/Research_UR/SDSS/dr16/manga/spectro/'
manga = '/Users/richardzhang/Documents/UR_Stuff/Research_UR/SDSS/dr16/manga/spectro/'
#manga = '/Users/kellydouglass/Documents/Research/data/SDSS/dr16/manga/spectro/'
################################################################################




################################################################################
# Import galaxy data
#-------------------------------------------------------------------------------
gal_ID = '7815-6104'
data_maps, gshape, x_center_guess, y_center_guess = Galaxy_Data(gal_ID, 
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
    elif (Rh < Rb) or (Rh < Rd) or (Rd < Rb):
        logP = -np.inf

    else:
    	logP = -np.inf

    return logP



def log_prob_iso(params, scale, shape, vdata, ivar, mask):

    lp = log_prior(params)

    logL = loglikelihood_iso_flat_constraints(params, 
                                              scale, 
                                              shape, 
                                              vdata.compressed(), 
                                              ivar.compressed(), 
                                              mask)

    if not np.isfinite(lp) or not np.isfinite(logL):
        return -np.inf 
    else:
        return lp + logL

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
mini_soln = [0.999956293,
             5.985783412,
             2999.999855,
             5.896416606,
             1.830766403,
             0.101792515,
             0.400013985,
             2.086430968,
             42.75920041,
             8.822686101,
             99.99993356]
'''

# 7443-6101

initial_guesses = [-1, 1, 1000, 4, -3, 25, 0.34798028637557554, 4.614072838290597, 26, 26, 0]

model_guesses = [-1, 1, 1000, 4, -3, 25]

geo_guesses = [0.34798028637557554, 4.614072838290597, 26, 26, 0]
################################################################################

'''
################################################################################
# random walker

pos_rand = np.random.uniform(low=[-7,0,0.1,0.1,-7,0.001,0,0,10,10,-100], 
                        high=[1,5,3000,30,-2,500,0.436*np.pi,2*np.pi,50,50,100], 
                        size=(64,11))

#-------------------------------------------------------------------------------

nwalkers, ndim = pos_rand.shape

bad_sampler_iso = emcee.EnsembleSampler(nwalkers, 
                                        ndim, 
                                        log_prob_iso, 
                                        args=(scale, 
                                              gshape, 
                                              data_maps['vmasked'], 
                                              data_maps['ivar_masked'], 
                                              data_maps['Ha_vel_mask']))

bad_sampler_iso.run_mcmc(pos_rand, 10000, progress=True)
bad_samples_iso = bad_sampler_iso.get_chain()
#bad_samples_iso = bad_sampler_iso.get_chain(discard=500)

np.save('bad_samples_iso_' + gal_ID + '_rand.npy', bad_samples_iso)

good_walkers_iso = bad_sampler_iso.acceptance_fraction > 0
np.save('good_walkers_iso_' + gal_ID + '_rand.npy', good_walkers_iso)

#-------------------------------------------------------------------------------

bad_sampler_NFW = emcee.EnsembleSampler(nwalkers, 
                                        ndim, 
                                        log_prob_NFW, 
                                        args=(scale, 
                                              gshape, 
                                              data_maps['vmasked'], 
                                              data_maps['ivar_masked'], 
                                              data_maps['Ha_vel_mask']))

bad_sampler_NFW.run_mcmc(pos_rand, 10000, progress=True)
bad_samples_NFW = bad_sampler_NFW.get_chain()
#bad_samples_NFW = bad_sampler_NFW.get_chain(discard=500)

np.save('bad_samples_NFW_' + gal_ID + '_rand.npy', bad_samples_NFW)

good_walkers_NFW = bad_sampler_NFW.acceptance_fraction > 0
np.save('good_walkers_NFW_' + gal_ID + '_rand.npy', good_walkers_NFW)

#-------------------------------------------------------------------------------

bad_sampler_bur = emcee.EnsembleSampler(nwalkers, 
                                        ndim, 
                                        log_prob_bur, 
                                        args=(scale, 
                                              gshape, 
                                              data_maps['vmasked'], 
                                              data_maps['ivar_masked'], 
                                              data_maps['Ha_vel_mask']))

bad_sampler_bur.run_mcmc(pos_rand, 10000, progress=True)
bad_samples_bur = bad_sampler_bur.get_chain()
#bad_samples_bur = bad_sampler_bur.get_chain(discard=500)

np.save('bad_samples_bur_' + gal_ID + '_rand.npy', bad_samples_bur)

good_walkers_bur = bad_sampler_bur.acceptance_fraction > 0
np.save('good_walkers_bur_' + gal_ID + '_rand.npy', good_walkers_bur)

################################################################################

################################################################################

# seeding around initial guess

pos_init = initial_guesses + np.random.uniform(np.random.uniform(low=-1e-3*np.ones(len(initial_guesses)), 
                                              high=1e-3*np.ones(len(initial_guesses)), 
                                              size=(64,len(initial_guesses))))

#-------------------------------------------------------------------------------

nwalkers, ndim = pos_init.shape

bad_sampler_iso = emcee.EnsembleSampler(nwalkers, 
                                        ndim, 
                                        log_prob_iso, 
                                        args=(scale, 
                                              gshape, 
                                              data_maps['vmasked'], 
                                              data_maps['ivar_masked'], 
                                              data_maps['Ha_vel_mask']))

bad_sampler_iso.run_mcmc(pos_init, 10000, progress=True)
bad_samples_iso = bad_sampler_iso.get_chain()
#bad_samples_iso = bad_sampler_iso.get_chain(discard=500)

np.save('bad_samples_iso_' + gal_ID + '_init.npy', bad_samples_iso)

good_walkers_iso = bad_sampler_iso.acceptance_fraction > 0
np.save('good_walkers_iso_' + gal_ID + '_init.npy', good_walkers_iso)

#-------------------------------------------------------------------------------

bad_sampler_NFW = emcee.EnsembleSampler(nwalkers, 
                                        ndim, 
                                        log_prob_NFW, 
                                        args=(scale, 
                                              gshape, 
                                              data_maps['vmasked'], 
                                              data_maps['ivar_masked'], 
                                              data_maps['Ha_vel_mask']))

bad_sampler_NFW.run_mcmc(pos_init, 10000, progress=True)
bad_samples_NFW = bad_sampler_NFW.get_chain()
#bad_samples_NFW = bad_sampler_NFW.get_chain(discard=500)

np.save('bad_samples_NFW_' + gal_ID + '_init.npy', bad_samples_NFW)

good_walkers_NFW = bad_sampler_NFW.acceptance_fraction > 0
np.save('good_walkers_NFW_' + gal_ID + '_init.npy', good_walkers_NFW)

#-------------------------------------------------------------------------------

bad_sampler_bur = emcee.EnsembleSampler(nwalkers, 
                                        ndim, 
                                        log_prob_bur, 
                                        args=(scale, 
                                              gshape, 
                                              data_maps['vmasked'], 
                                              data_maps['ivar_masked'], 
                                              data_maps['Ha_vel_mask']))

bad_sampler_bur.run_mcmc(pos_init, 10000, progress=True)
bad_samples_bur = bad_sampler_bur.get_chain()
#bad_samples_bur = bad_sampler_bur.get_chain(discard=500)

np.save('bad_samples_bur_' + gal_ID + '_init.npy', bad_samples_bur)

good_walkers_bur = bad_sampler_bur.acceptance_fraction > 0
np.save('good_walkers_bur_' + gal_ID + '_init.npy', good_walkers_bur)

################################################################################
'''

################################################################################

# Combined

pos_model = np.random.uniform(low=[-7,0.00001,200,0.1,2e-5,0.1], 
                        high=[1,5,2500,25,0.1,500], 
                        size=(64,6))

pos_geo = np.array(geo_guesses) + np.random.uniform(np.random.uniform(low=-1e-3*np.ones(len(geo_guesses)), 
                                              high=1e-3*np.ones(len(geo_guesses)), 
                                              size=(64,len(geo_guesses))))

pos_combined = np.column_stack((pos_model,pos_geo))

#-------------------------------------------------------------------------------

nwalkers, ndim = pos_combined.shape

bad_sampler_iso = emcee.EnsembleSampler(nwalkers, 
                                        ndim, 
                                        log_prob_iso, 
                                        args=(scale, 
                                              gshape, 
                                              data_maps['vmasked'], 
                                              data_maps['ivar_masked'], 
                                              data_maps['Ha_vel_mask']))

bad_sampler_iso.run_mcmc(pos_combined, 10000, progress=True)
bad_samples_iso = bad_sampler_iso.get_chain()
#bad_samples_iso = bad_sampler_iso.get_chain(discard=500)

np.save('bad_samples_iso_' + gal_ID + '_comb.npy', bad_samples_iso)

good_walkers_iso = bad_sampler_iso.acceptance_fraction > 0
np.save('good_walkers_iso_' + gal_ID + '_comb.npy', good_walkers_iso)

#-------------------------------------------------------------------------------

bad_sampler_NFW = emcee.EnsembleSampler(nwalkers, 
                                        ndim, 
                                        log_prob_NFW, 
                                        args=(scale, 
                                              gshape, 
                                              data_maps['vmasked'], 
                                              data_maps['ivar_masked'], 
                                              data_maps['Ha_vel_mask']))

bad_sampler_NFW.run_mcmc(pos_combined, 10000, progress=True)
bad_samples_NFW = bad_sampler_NFW.get_chain()
#bad_samples_NFW = bad_sampler_NFW.get_chain(discard=500)

np.save('bad_samples_NFW_' + gal_ID + '_comb.npy', bad_samples_NFW)

good_walkers_NFW = bad_sampler_NFW.acceptance_fraction > 0
np.save('good_walkers_NFW_' + gal_ID + '_comb.npy', good_walkers_NFW)

#-------------------------------------------------------------------------------

bad_sampler_bur = emcee.EnsembleSampler(nwalkers, 
                                        ndim, 
                                        log_prob_bur, 
                                        args=(scale, 
                                              gshape, 
                                              data_maps['vmasked'], 
                                              data_maps['ivar_masked'], 
                                              data_maps['Ha_vel_mask']))

bad_sampler_bur.run_mcmc(pos_combined, 10000, progress=True)
bad_samples_bur = bad_sampler_bur.get_chain()
#bad_samples_bur = bad_sampler_bur.get_chain(discard=500)

np.save('bad_samples_bur_' + gal_ID + '_comb.npy', bad_samples_bur)

good_walkers_bur = bad_sampler_bur.acceptance_fraction > 0
np.save('good_walkers_bur_' + gal_ID + '_comb.npy', good_walkers_bur)

################################################################################


################################################################################
# Import modules
#-------------------------------------------------------------------------------
#import matplotlib.pyplot as plt

import numpy as np

import emcee
#import corner

#import pickle

from Velocity_Map_Functions import loglikelihood_iso_flat_constraints

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
manga = '/Users/richardzhang/Documents/UR_Stuff/Research_UR/SDSS/dr16/manga/spectro/'
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
################################################################################




################################################################################
# Best-fit parameter values from scipy.optimize.minimize
#-------------------------------------------------------------------------------
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
################################################################################




################################################################################
# Isothermal
#-------------------------------------------------------------------------------

pos = np.array(mini_soln) + np.random.uniform(low=-1e-3*np.ones(len(mini_soln)), 
                                              high=1e-3*np.ones(len(mini_soln)), 
                                              size=(64,11))
'''
pos = np.random.uniform(low=[0, 0, 0.1, 0.1, 0, 0.1, 0.1, 0, 15, 15, -100], 
                        high=[10, 5, 3000, 25, 100, 500, 0.436*np.pi, 2*np.pi, 45, 45, 100], 
                        size=(64,11))
'''
nwalkers, ndim = pos.shape

bad_sampler_iso = emcee.EnsembleSampler(nwalkers, 
                                        ndim, 
                                        log_prob_iso, 
                                        args=(scale, 
                                              gshape, 
                                              data_maps['vmasked'], 
                                              data_maps['ivar_masked'], 
                                              data_maps['Ha_vel_mask']))
bad_sampler_iso.run_mcmc(pos, 10000, progress=True)
bad_samples_iso = bad_sampler_iso.get_chain()
#bad_samples_iso = bad_sampler_iso.get_chain(discard=500)

np.save('bad_samples_iso.npy', bad_samples_iso)

good_walkers_iso = bad_sampler_iso.acceptance_fraction > 0
np.save('good_walkers_iso.npy', good_walkers_iso)

'''
fig_iso, axes_iso = plt.subplots(11,1, figsize=(20, 14), sharex=True,
                                 gridspec_kw={'hspace':0.1})
bad_samples_iso = bad_sampler_iso.get_chain()[:,good_walkers_iso,:]
np.save('bad_samples_iso.npy',bad_samples_iso)


labels = ['rho_b','R_b', 'Sigma_d','R_d','rho_h','R_h','i','phi','x','y','vsys']
for i in range(ndim):
    ax = axes_iso[i]
    ax.plot(bad_samples_iso[:5000,:,i], 'k', alpha=0.3)
    ax.set(xlim=(0,5000), ylabel=labels[i])
    ax.yaxis.set_label_coords(-0.11, 0.5)

axes_iso[-1].set_xlabel('step number')
#fig_iso.tight_layout()
plt.savefig('mcmc_iso.png',format='png')
plt.close()
################################################################################

################################################################################
bad_samples_iso = bad_sampler_iso.get_chain(discard=500)[:,good_walkers_iso,:]
np.save('bad_samples_iso.npy',bad_samples_iso)
ns_iso, nw_iso, nd_iso = bad_samples_iso.shape
flat_bad_samples_iso = bad_samples_iso.reshape(ns_iso*nw_iso, nd_iso)
np.save('flat_bad_samples_iso.npy',flat_bad_samples_iso)
flat_bad_samples_iso.shape
################################################################################

################################################################################
corner.corner(flat_bad_samples_iso, labels=labels,
                    range=[(-7,2), (0,5), (0,2000),(1,20),(0.0001,0.01),(5,200),(0,np.pi/2),(0,1.5),(30,40),(30,40),(-100,100)], bins=30, #smooth=1,
                    truths=[np.log10(0.048688757),2.549862293,748.5940907,5.617303041,0.002927534,0.100051148,1.070928683,0.699892835,36.61461409,37.68004929,11.37083843], truth_color='#ff4444',
                    levels=(1-np.exp(-0.5), 1-np.exp(-2)), 
                    quantiles=(0.16, 0.84),
                    hist_kwargs={'histtype':'stepfilled', 'alpha':0.3, 'density':True},
                    color='blue', plot_datapoints=False,
                    fill_contours=True)
plt.savefig('corner_iso.png',format='png')
plt.close()
################################################################################

for i, label in enumerate(labels):
    x = mini_soln[i]
    x16, x84 = np.percentile(flat_bad_samples_iso[:,i], [16,84])
    dlo = x - x16
    dhi = x84 - x
    print('{:3s} = {:5.2f} + {:4.2f} - {:4.2f}'.format(label, x, dhi, dlo))
    print('    = ({:5.2f}, {:5.2f})'.format(x16, x84))

################################################################################
# Dumping output
#-------------------------------------------------------------------------------
#out_directory = '/Users/richardzhang/Documents/UR_Stuff/Research_UR/2D_RC/MCMC_folder/'
#temp_outfile = open(out_directory + 'results.pickle', 'wb')
#pickle.dump((flat_bad_samples_iso, flat_bad_samples_NFW, flat_bad_samples_bur), temp_outfile)
#temp_outfile.close()
'''



####################################################################
#import matplotlib.pyplot as plt

import numpy as np

import numdifftools as ndt

import emcee
import corner

import pickle

from Velocity_Map_Functions import loglikelihood_iso_flat

from RC_2D_Fit_Functions import Galaxy_Data    

G = 6.674E-11  # m^3 kg^-1 s^-2
Msun = 1.989E30  # kg
scale = 0.46886408261217366                                                                    
####################################################################

#manga = '/home/yzh250/Documents/UR_Stuff/Research_UR/SDSS/dr16/manga/spectro/'
manga =  '/Users/richardzhang/Documents/UR_Stuff/Research_UR/SDSS/dr16/manga/spectro/'

####################################################################
# 7443-12705
data_maps, gshape, x_center_guess, y_center_guess = Galaxy_Data('7443-6101',manga)
####################################################################

####################################################################
# loglikelihood
def log_prior(params):
    log_rhob0,Rb,SigD,Rd,rho_h,Rh,inclination,phi,center_x,center_y,vsys= params
    logP = 0
    if -7 < log_rhob0 < 2 and 0 < Rb < 5 and 100 < SigD < 3000 and 1 < Rd < 30\
     and 1e-5 < rho_h < 0.1 and 0.01 < Rh< 500 and 0 < inclination < np.pi*0.436 and 0 < phi < 2*np.pi\
     and 10 < center_x < 50 and 10 < center_y < 50 and -100 < vsys < 100:
        logP = 0
    # setting constraints on the radii
    elif Rh < Rb or Rh < Rd or Rd < Rd:
        logP = -np.inf
    else:
    	logP = -np.inf
    return logP

def log_prob_iso(params, scale, shape, vdata, ivar, mask):
    lp = log_prior(params)
    logL = loglikelihood_iso_flat(params, scale, shape, vdata.compressed(), ivar.compressed(), mask)
    if not np.isfinite(lp) or not np.isfinite(logL):
        return -np.inf 
    else:
        return lp + logL
####################################################################

mini_soln = [np.log10(11.66291723),2.69E-05,1031.023329,1.838768634,0.083546044,0.102759719,0.553854733,1.951500683,26.37172472,27.44266793,0.907424538]

####################################################################
# Isothermal

pos = np.array(mini_soln) + np.random.uniform(low=-1e-4*np.ones(len(mini_soln)), high=1e-4*np.ones(len(mini_soln)), size=(64,11))
#pos = np.random.uniform(low=[-6,0.00001,200,0.1,2e-5,0.1,0,0,15,15,-50], high=[2,5,2500,25,0.1,500,0.436*np.pi,2*np.pi,45,45,50], size=(64,11))

nwalkers, ndim = pos.shape

bad_sampler_iso = emcee.EnsembleSampler(nwalkers, ndim, log_prob_iso, args=(scale, gshape, data_maps['vmasked'], data_maps['ivar_masked'], data_maps['Ha_vel_mask']))
bad_sampler_iso.run_mcmc(pos, 5000, progress=True)
bad_samples_iso = bad_sampler_iso.get_chain()
good_walkers_iso = bad_sampler_iso.acceptance_fraction > 0
np.save('bad_samples_iso.npy',bad_samples_iso)
np.save('good_walkers_iso.npy',good_walkers_iso)

#good_walkers_iso = bad_sampler_iso.acceptance_fraction > 0
#np.save('good_walkers_iso.npy',good_walkers_iso)

#fig_iso, axes_iso = plt.subplots(11,1, figsize=(20, 14), sharex=True,
                         #gridspec_kw={'hspace':0.1})
#bad_samples_iso = bad_sampler_iso.get_chain()[:,good_walkers_iso,:]
#np.save('bad_samples_iso.npy',bad_samples_iso)

'''
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
####################################################################

####################################################################
bad_samples_iso = bad_sampler_iso.get_chain(discard=500)[:,good_walkers_iso,:]
np.save('bad_samples_iso.npy',bad_samples_iso)
ns_iso, nw_iso, nd_iso = bad_samples_iso.shape
flat_bad_samples_iso = bad_samples_iso.reshape(ns_iso*nw_iso, nd_iso)
np.save('flat_bad_samples_iso.npy',flat_bad_samples_iso)
flat_bad_samples_iso.shape
####################################################################

####################################################################
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



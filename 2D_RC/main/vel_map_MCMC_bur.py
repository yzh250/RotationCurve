####################################################################
#import matplotlib.pyplot as plt

import numpy as np

import numdifftools as ndt

import emcee
import corner

import pickle

from Velocity_Map_Functions import loglikelihood_bur_flat

from RC_2D_Fit_Functions import Galaxy_Data    

G = 6.674E-11  # m^3 kg^-1 s^-2
Msun = 1.989E30  # kg
scale = 0.46886408261217366                                                                     
####################################################################

####################################################################
# 7443-12705
r_band, Ha_vel, Ha_vel_ivar, Ha_vel_mask, Ha_flux, Ha_flux_ivar, Ha_flux_mask, vmasked, Ha_flux_masked, ivar_masked, gshape, x_center_guess, y_center_guess = Galaxy_Data('7443-12705')
####################################################################

####################################################################
# loglikelihood
def log_prior(params):
    rho_b,Rb,SigD,Rd,rho_h,Rh,inclination,phi,center_x,center_y,vsys= params
    logP = 0
    if 0 < rho_b < 100 and 0 < Rb < 5 and 100 < SigD < 3000 and 1 < Rd < 30\
     and 1e-5 < rho_h < 0.1 and 0.01 < Rh< 500 and 0 < inclination < 0.436*np.pi and 0 < phi < 2*np.pi\
     and 20 < center_x < 40 and 20 < center_y < 40 and -100 < vsys < 100:
        logP = 0
    # setting constraints on the radii
    elif Rh < Rb or Rh < Rd or Rd < Rd:
        logP = -np.inf
    else:
        logP = -np.inf
    return logP

def log_prob_bur(params, scale, shape, vdata, ivar, mask):
    lp = log_prior(params)
    logL = loglikelihood_bur_flat(params, scale, shape, vdata.compressed(), ivar.compressed(), mask)
    if not np.isfinite(lp) or not np.isfinite(logL):
        return -np.inf 
    else:
        return lp + logL
####################################################################

mini_soln = [np.log10(5.36E-05),2.811046162,978.7934831,6.493085395,4.10E-05,999.8669552,0.858228903,0.752910577,38.25051586,37.23417255,-0.685352448]

####################################################################
# Burket

pos = np.array(mini_soln) + np.random.uniform(low=-1e-6*np.ones(len(mini_soln)), high=1e-6*np.ones(len(mini_soln)), size=(64,11))
nwalkers, ndim = pos.shape

bad_sampler_bur = emcee.EnsembleSampler(nwalkers, ndim, log_prob_bur, args=(scale, gshape, data_maps['vmasked'], data_maps['ivar_masked'], data_maps['Ha_vel_mask']))
bad_sampler_bur.run_mcmc(pos, 5000, progress=True)

good_walkers_bur = bad_sampler_bur.acceptance_fraction > 0
np.save('good_walkers_bur.npy',good_walkers_bur)

fig_bur, axes_bur = plt.subplots(11,1, figsize=(20, 14), sharex=True,
                         gridspec_kw={'hspace':0.1})
bad_samples_bur = bad_sampler_bur.get_chain()[:,good_walkers_bur,:]
np.save('bad_samples_bur.npy',bad_samples_bur)

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



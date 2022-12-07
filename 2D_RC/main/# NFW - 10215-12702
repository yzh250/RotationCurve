# NFW - 10215-12702

plt.figure(figsize=(15,7)) #tight_layout=True)

################################################################################
# Original data
#-------------------------------------------------------------------------------
plt.subplot(231)

data = plt.imshow(dmap_10215_12702, 
                      origin='lower', 
                      cmap='RdBu_r', 
                      vmin=-125, 
                      vmax=125)

plt.xlabel('spaxel')
plt.ylabel('spaxel')

cbar = plt.colorbar(data)
cbar.set_label('km/s')

plt.title('10215-12702 data')
################################################################################

################################################################################
# minimize model
#-------------------------------------------------------------------------------
plt.subplot(232)

map_minimize = plt.imshow(mini_map_10215_12702, 
                              origin='lower',
                              cmap='RdBu_r', 
                              vmin=-125, 
                              vmax=125)

plt.xlabel('spaxel')
plt.ylabel('spaxel')

cbar = plt.colorbar(map_minimize)
cbar.set_label('km/s')

plt.title('10215-12702 minimize NFW halo model')
################################################################################


################################################################################
# MCMC model
#-------------------------------------------------------------------------------
plt.subplot(233)

map_mcmc = plt.imshow(mcmc_map_10215_12702, 
                              origin='lower',
                              cmap='RdBu_r', 
                              vmin=-125, 
                              vmax=125)

plt.xlabel('spaxel')
plt.ylabel('spaxel')

cbar = plt.colorbar(map_mcmc)
cbar.set_label('km/s')

plt.title('10215-12702 MCMC NFW halo model')
################################################################################

################################################################################
# Minimize 1D rotation Curve
#-------------------------------------------------------------------------------
plt.subplot(235)

plt.plot(r_mini_10215_12702, v_mini_10215_12702, 'c',label='v tot')
plt.plot(r_mini_10215_12702, v_b_mini_10215_12702, '--',label='bulge')
plt.plot(r_mini_10215_12702, v_d_mini_10215_12702,'.-',label='disk')
plt.plot(r_mini_10215_12702, v_h_mini_10215_12702,':',label='halo')

vmax = np.max(np.abs(v_mini_10215_12702))

plt.xlim([0,np.max(r_mini_10215_12702)])
plt.ylim([0,1.25*vmax])
plt.xlabel('Deprojected radius [kpc/h]')
plt.ylabel('Rotational velocity [km/s]')
plt.legend()
################################################################################

################################################################################
# MCMC model
#-------------------------------------------------------------------------------
plt.subplot(236)

plt.plot(r_mcmc_10215_12702, v_mcmc_10215_12702, 'c',label='v tot')
plt.plot(r_mcmc_10215_12702, v_b_mcmc_10215_12702, '--',label='bulge')
plt.plot(r_mcmc_10215_12702, v_d_mcmc_10215_12702,'.-',label='disk')
plt.plot(r_mcmc_10215_12702, v_h_mcmc_10215_12702,':',label='halo')

vmax = np.max(np.abs(v_mcmc_10215_12702))

plt.xlim([0,np.max(r_mcmc_10215_12702)])
plt.ylim([0,1.25*vmax])
plt.xlabel('Deprojected radius [kpc/h]')
plt.ylabel('Rotational velocity [km/s]')
plt.legend()
################################################################################
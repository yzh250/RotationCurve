################################################################################
# Import modules
#-------------------------------------------------------------------------------
import numpy as np
from astropy.table import Table, QTable
from astropy.io import ascii

from mass_calc_functions import bulge_mass,\
								disk_mass,\
								halo_mass_bur
################################################################################

################################################################################
# Physics Constants
#-------------------------------------------------------------------------------
c = 3E5 # km * s ^1
h = 1 # reduced hubble constant
H_0 =  100 * h # km * s^-1 * Mpc^-1
q0 = 0.2 # minimum inclination value
################################################################################

################################################################################
# Used files (local)
#-------------------------------------------------------------------------------
MANGA_FOLDER = '/Users/richardzhang/Documents/UR_Stuff/Research_UR/SDSS/dr16/manga/spectro/'

DRP_FILENAME = MANGA_FOLDER + 'redux/v3_1_1/drpall-v3_1_1.fits'

# Can't really use this anymore
VEL_MAP_FOLDER = MANGA_FOLDER + 'analysis//v3_1_1/2.1.1/HYB10-GAU-MILESHC/'

MORPH_FOLDER = '/Users/richardzhang/Documents/UR_Stuff/Research_UR/SDSS/dr16/manga/morph/'

fits_file = '/Users/richardzhang/Documents/UR_Stuff/Research_UR/RotationCurve/2D_RC/main/Notebooks/'

main_file = '/Users/richardzhang/Documents/UR_Stuff/Research_UR/RotationCurve/2D_RC/main/'

r90_file = main_file + 'r90.csv'
################################################################################

################################################################################
# DRP all table
DTable =  Table.read(DRP_FILENAME, format='fits')

z = DTable['nsa_z']

f_r90 = ascii.read(r90_file,'r')

r90_list = f_r90['r90']

DRP_index = {}

for i in range(len(DTable)):
    gal_ID = DTable['plateifu'][i]

    DRP_index[gal_ID] = i

galaxy_ID = []
plateifu = DTable['plateifu'].data

for i in range(len(plateifu)):
    galaxy_ID.append(str(plateifu[i],'utf-8'))

stellar_mass = DTable['nsa_elpetro_mass']
################################################################################

################################################################################
# Reading in all the files
# Isothermal
fit_mini_iso_name = fits_file + 'iso_mini.csv'
fit_mini_iso = ascii.read(fit_mini_iso_name,'r')
# NFW
fit_mini_nfw_name = fits_file + 'nfw_mini.csv'
fit_mini_nfw = ascii.read(fit_mini_nfw_name,'r')
#Burket
fit_mini_bur_name = fits_file + 'bur_mini.csv'
fit_mini_bur = ascii.read(fit_mini_bur_name,'r')
################################################################################

c_table = Table()
c_table['galaxy_ID'] = galaxy_ID
c_table['Rh_iso'] = np.nan
c_table['Rh_nfw'] = np.nan
c_table['Rh_bur'] = np.nan
c_table['R90'] = np.nan
################################################################################

for i in range(len(fit_mini_iso)):
	gal_fit_iso = list(fit_mini_iso[i])
	gal_fit_nfw = list(fit_mini_nfw[i])
	gal_fit_bur = list(fit_mini_bur[i])

	Isothermal_fit_mini = gal_fit_iso[1:-2]
	NFW_fit_mini = gal_fit_nfw[1:-2]
	Burket_fit_mini = gal_fit_bur[1:-2]

	j = DRP_index[gal_fit_iso[0]]
	redshift = z[j]
	velocity = redshift * c
	distance = (velocity / H_0) * 1000
	r90 = distance * (r90_list[i]/206265)

	rho_0_iso = float(Isothermal_fit_mini[0])
	Rb_iso = float(Isothermal_fit_mini[1])
	SigD_iso = float(Isothermal_fit_mini[2])
	Rd_iso = float(Isothermal_fit_mini[3])
	rho0_h_iso = float(Isothermal_fit_mini[4])
	Rh_iso = float(Isothermal_fit_mini[5])
	rho_0_nfw = float(NFW_fit_mini[0])
	Rb_nfw = float(NFW_fit_mini[1])
	SigD_nfw = float(NFW_fit_mini[2])
	Rd_nfw = float(NFW_fit_mini[3])
	rho0_h_nfw = float(NFW_fit_mini[4])
	Rh_nfw = float(NFW_fit_mini[5])
	rho_0_bur = float(Burket_fit_mini[0])
	Rb_bur = float(Burket_fit_mini[1])
	SigD_bur = float(Burket_fit_mini[2])
	Rd_bur = float(Burket_fit_mini[3])
	rho0_h_bur = float(Burket_fit_mini[4])
	Rh_bur = float(Burket_fit_mini[5])

	if np.isfinite(gal_fit_iso[-1]) and gal_fit_iso[-1] < 150 and np.isfinite(gal_fit_nfw[-1]) and gal_fit_nfw[-1] < 150 and np.isfinite(gal_fit_bur[-1]) and gal_fit_bur[-1] < 150:
		if round(rho_0_iso) == -7 or round(rho_0_iso) == 1 or round(rho_0_nfw) == -7 or round(rho_0_nfw) == 1 or round(rho_0_bur) == -7 or round(rho_0_bur) == 1\
		   or round(Rb_iso) == 0 or round(Rb_iso) == 10 or round(Rb_nfw) == 0 or round(Rb_nfw) == 10 or round(Rb_bur) == 0 or round(Rb_bur) == 10\
		   or round(SigD_iso,1) == 0.1 or round(SigD_iso,1) == 3000 or round(SigD_nfw,1) == 0.1 or round(SigD_nfw,1) == 3000 or round(SigD_bur,1) == 0.1 or round(SigD_bur,1) == 3000\
		   or round(Rd_iso,1) == 0.1 or round(Rd_iso) == 30 or round(Rd_nfw,1) == 0.1 or round(Rd_nfw) == 30 or round(Rd_bur,1) == 0.1 or round(Rd_bur) == 30\
		   or round(rho0_h_iso) == -7 or round(rho0_h_iso) == 2 or round(rho0_h_nfw) == -7 or round(rho0_h_nfw) == 2 or round(rho0_h_bur) == -7 or round(rho0_h_bur) == 2\
		   or round(Rh_iso,1) == 0.1 or round(Rh_iso) == 1000 or round(Rh_nfw,1) == 0.1 or round(Rh_nfw) == 1000 or round(Rh_bur,1) == 0.1 or round(Rh_bur) == 1000:
		   print('One or more paramter is hitting the bound.')
		else:
			c_table['Rh_iso'][i] = Rh_iso
			c_table['Rh_nfw'][i] = Rh_nfw
			c_table['Rh_bur'][i] = Rh_bur
			c_table['R90'][i] = r90
c_table.write('rh_r90.csv', format='ascii.csv', overwrite=True)
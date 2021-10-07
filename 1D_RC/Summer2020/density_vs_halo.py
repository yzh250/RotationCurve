import numpy as np

import matplotlib.pyplot as plt

from astropy.table import QTable

Dtable = QTable.read('good_fit_re.txt',format='ascii.ecsv')

plt.figure()

plt.plot(Dtable["Rho_dc (M_sol/pc^3)"],Dtable["R_halo (kpc)"],'.')
plt.xlabel("Rho_dc (M_sol/pc^3)")
plt.ylabel("R_halo (kpc)")
plt.show()


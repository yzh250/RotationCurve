from galpy import potential
import matplotlib.pyplot as plt
lp= potential.LogarithmicHaloPotential(amp=1.,q=0.7)
lp.plot(justcontours=True,rmin=-1.5,rmax=1.5,nrs=100,nzs=100)
plt.xlabel(r'$r$')
plt.ylabel(r'$z$')
plt.show()
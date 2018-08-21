import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from mpl_toolkits.axes_grid1 import make_axes_locatable
import dtw 

# parameters
maxLag = 80
dt = 0.004
b = 1
lvec = np.arange(-maxLag,maxLag+1)*dt
df = pd.read_csv('/Users/thclements/Desktop/DynamicWarping-python/exampleData/sineShiftData.csv',names=['st','u0','u1'])
st, u0, u1 = df['st'].values, df['u0'].values, df['u1'].values
npts = len(u0)
tvec   = np.arange(npts) * dt
stTime = st * dt

err = dtw.computeErrorFunction( u1, u0, npts, maxLag )
direction = 1

dist  = dtw.accumulateErrorFunction( direction, err, npts, maxLag, b )
stbar = dtw.backtrackDistanceFunction( -1*direction, dist, err, -maxLag, b )

stbarTime = stbar * dt   # convert from samples to time
tvec2     = tvec + stbarTime # make the warped time axis

fig, ax = plt.subplots(2,1,figsize=(20,10))
dist_mat = ax[0].imshow(dist.T,aspect='auto',extent=[tvec[0],tvec[-1],lvec[-1],lvec[0]])
ax[0].set_title('Distance function')
ax[0].set_xlabel('Time [s]')
ax[0].set_ylabel(r'$\tau$ [s]')
ax[0].invert_yaxis()  
cax = fig.add_axes([0.93, 0.55, 0.03, 0.3])
fig.colorbar(dist_mat,cax=cax)
# plot real shifts against estimated shifts
ax[1].plot(tvec,stTime,'ko',label='Actual')
ax[1].plot(tvec,stbarTime,'r+',label='Estimated') 
ax[1].legend(fontsize=12)
ax[1].set_title('Estimated shifts')
ax[1].set_xlabel('Time [s]')
ax[1].set_ylabel(r'$\tau$ [s]')
ax[1].set_xlim([tvec[0],tvec[-1]])
plt.autoscale(enable=True, tight=True)
plt.savefig('SINEdistance.png')
plt.show()
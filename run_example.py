import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
import dtw 

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

matplotlib.rc('font', **font)

# This script runs an example of dynamic time warping as an introduction to
# the method.

example = 2 # shift function (1=step, 2=sine)

# you can try two test cases. Watch the 'maxLag' parameter. The two shift
# functions have different magnitudes so if you don't go to large enough
# lags in the sine model, you won't converge to the correction solution.

# in the case of the sine wave, you must have b=1 because the strains are
# on the order of dt!! Play with 'b' to see how you get stuck in the
# wrong minimum because you're not allowing large enough jumps.

# in the case of the step function, we break the DTW because the step shift
# is larger than dt. We would need a different step pattern to correctly
# solve for this step. It's possible, just not implemented in this version
# of the dynamic warping because don't expect strains > 100%.


# parameters
maxLag = 80 #  max nuber of points to search forward and backward (can be npts, just takes longer and is unrealistic)
dt = 0.004
b = 1 # b-value to limit strain
# impose a strain limit: 1 dt is b=1, half dt is b=2, etc.
# if you mess with b you will see the results in the estimated shifts

# load the data file and plot
if example == 1:
        df = pd.read_csv('exampleData/stepShiftData.csv',names=['st','u0','u1'])
elif example == 2:
        df = pd.read_csv('exampleData/sineShiftData.csv',names=['st','u0','u1'])


lvec = np.arange(-maxLag,maxLag+1)*dt # lag array for plotting below
st, u0, u1 = df['st'].values, df['u0'].values, df['u1'].values
npts = len(u0) # number of samples
tvec   = np.arange(npts) * dt # make the time axis
stTime = st * dt # shift vector in time

fig, ax = plt.subplots(2,1,figsize=(20,10))
# plot shift function
ax[0].plot(tvec,stTime)
ax[0].set_xlabel('Time [s]')
ax[0].set_ylabel(r'$\tau$ [s]')
ax[0].set_title(r'Shift applied to $u_{0}$(t)')
# plot original and shifted traces
ax[1].plot(tvec,u0,label=r'$u_{0}$(t)')
ax[1].plot(tvec,u1, label='u(t)')
ax[1].set_xlabel('Time [s]')
ax[1].set_ylabel('Amplitude [a.u.]')
ax[1].set_title('Traces')
ax[1].legend(loc='best',frameon=False)
plt.tight_layout()

# compute error function and plot
err = dtw.computeErrorFunction( u1, u0, npts, maxLag ) # compute error function over lags
# note the error function is independent of strain limit 'b'.

plt.figure(figsize=(10,10))
plt.imshow(np.flipud(np.log10(err.T + 1e-16)),aspect='auto',cmap=plt.cm.gray,extent=[tvec[0],tvec[-1],lvec[-1],lvec[0]])
plt.xlabel('Time [s]')
plt.ylabel('Lag')
plt.title('Error Function')
plt.colorbar()
plt.tight_layout()

direction = 1
# direction to accumulate errors (1=forward, -1=backward)
# it is instructive to flip the sign of +/-1 here to see how the function
# changes as we start the backtracking on different sides of the traces.
# Also change 'b' to see how this influences the solution for stbar. You
# want to make sure you're doing things in the proper directions in each
# step!!!

dist  = dtw.accumulateErrorFunction( direction, err, npts, maxLag, b )
stbar = dtw.backtrackDistanceFunction( -1*direction, dist, err, -maxLag, b )

stbarTime = stbar * dt   # convert from samples to time
tvec2     = tvec + stbarTime # make the warped time axis

# plot the results
fig, ax = plt.subplots(2,1,figsize=(20,10))
dist_mat = ax[0].imshow(dist.T,aspect='auto',extent=[tvec[0],tvec[-1],lvec[-1],lvec[0]])
ax[0].set_title('Distance function')
ax[0].set_xlabel('Time [s]')
ax[0].set_ylabel(r'$\tau$ [s]')
ax[0].invert_yaxis()  
cax = fig.add_axes([0.65, 0.5, 0.3, 0.02])
fig.colorbar(dist_mat,cax=cax,orientation='horizontal')
# plot real shifts against estimated shifts
ax[1].plot(tvec,stTime,'ko',label='Actual')
ax[1].plot(tvec,stbarTime,'r+',label='Estimated') 
ax[1].legend(fontsize=12)
ax[1].set_title('Estimated shifts')
ax[1].set_xlabel('Time [s]')
ax[1].set_ylabel(r'$\tau$ [s]')
ax[1].set_xlim([tvec[0],tvec[-1]])
plt.autoscale(enable=True, tight=True)
fig.tight_layout()

# plot the input traces 
fig, ax = plt.subplots(2,1,figsize=(20,10),sharex=True)
ax[0].plot(tvec,u0,'b',label='Raw')
ax[0].plot(tvec,u1,'r--',label='Shifted')
ax[0].legend(loc='best',frameon=False)
ax[0].set_title('Input traces for dynamic time warping')
ax[0].set_xlabel('Time (s)')
ax[0].set_ylabel('Amplitude (a.u.)')
# plot warped trace to compare
ax[1].plot(tvec,u0,'b',label='Raw')
ax[1].plot(tvec2,u1,'r--',label='Shifted')
ax[1].legend(loc='best',frameon=False)
ax[1].set_title('Output traces for dynamic time warping')
ax[1].set_xlabel('Time (s)')
ax[1].set_ylabel('Amplitude (a.u.)')
plt.tight_layout()

# Apply dynamic time warping in both directions to smooth. (Follwoing example in Hale 2013)
dist1 = dtw.accumulateErrorFunction( -1, err, npts, maxLag, b ) # forward accumulation to make distance function
dist2 = dtw.accumulateErrorFunction( 1, err, npts, maxLag, b ); # backwward accumulation to make distance function

dist  = dist1 + dist2 - err; # add them and remove 'err' to not count twice (see Hale's paper)
stbar = dtw.backtrackDistanceFunction( -1, dist, err, -maxLag, b ); # find shifts
# !! Notice now that you can backtrack in either direction and get the same
# result after you smooth the distance function in this way.

# plot the results
stbarTime = stbar * dt      # convert from samples to time
tvec2     = tvec + stbarTime # make the warped time axis

fig, ax = plt.subplots(2,1,figsize=(20,10))
dist_mat = ax[0].imshow(dist.T,aspect='auto',extent=[tvec[0],tvec[-1],lvec[-1],lvec[0]])
ax[0].set_title('Distance function')
ax[0].set_xlabel('Time [s]')
ax[0].set_ylabel(r'$\tau$ [s]')
ax[0].invert_yaxis()  
cax = fig.add_axes([0.65, 0.5, 0.3, 0.02])
fig.colorbar(dist_mat,cax=cax,orientation='horizontal')
# plot real shifts against estimated shifts
ax[1].plot(tvec,stTime,'ko',label='Actual')
ax[1].plot(tvec,stbarTime,'r+',label='Estimated') 
ax[1].legend(fontsize=12)
ax[1].set_title('Estimated shifts')
ax[1].set_xlabel('Time [s]')
ax[1].set_ylabel(r'$\tau$ [s]')
ax[1].set_xlim([tvec[0],tvec[-1]])
plt.autoscale(enable=True, tight=True)
fig.tight_layout()

fig,ax = plt.subplots(2,1,figsize=(20,10))
ax[0].plot(tvec,u0,'b',label='Raw')
ax[0].plot(tvec,u1,'r--',label='Shifted')
ax[0].legend(loc='best',frameon=False)
ax[0].set_title('Input traces for dynamic time warping')
ax[0].set_xlabel('Time (s)')
ax[0].set_ylabel('Amplitude (a.u.)')
# plot warped trace to compare
ax[1].plot(tvec,u0,'b',label='Raw')
ax[1].plot(tvec2,u1,'r--',label='Shifted')
ax[1].legend(loc='best',frameon=False)
ax[1].set_title('Output traces for dynamic time warping')
ax[1].set_xlabel('Time (s)')
ax[1].set_ylabel('Amplitude (a.u.)')
plt.tight_layout()

plt.show()
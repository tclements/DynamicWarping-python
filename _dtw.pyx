#cython: boundscheck=False, wraparound=False, nonecheck=False

cimport cython 
import cython
import numpy as np
cimport numpy as np

DTYPE64 = np.float64
DTYPE32 = np.int
ctypedef np.int_t DTYPE32_t
ctypedef np.float64_t DTYPE64_t


cpdef computeErrorFunction(np.ndarray[np.float64_t, ndim=1] u1, 
                         np.ndarray[np.float64_t, ndim=1] u0,
                         int nSample, 
                         int lag, 
                         str norm='L2'):
    """
    USAGE: err = computeErrorFunction( u1, u0, nSample, lag )
    
    INPUT:
        u1      = trace that we want to warp; size = (nsamp,1)
        u0      = reference trace to compare with: size = (nsamp,1)
        nSample = numer of points to compare in the traces
        lag     = maximum lag in sample number to search
        norm    = 'L2' or 'L1' (default is 'L2')
    OUTPUT:
        err = the 2D error function; size = (nsamp,2*lag+1)
    
    The error function is equation 1 in Hale, 2013. You could uncomment the
    L1 norm and comment the L2 norm if you want on Line 29
    
    Original by Di Yang
    Last modified by Dylan Mikesell (25 Feb. 2015)
    Translated to python by Tim Clements (17 Aug. 2018)

    """

    # allocate variables 
    cdef int ll, thisLag, ii
    cdef np.ndarray[np.float64_t, ndim=2] err = np.zeros([nSample, 2 * lag + 1],dtype=DTYPE64)

    if lag >= nSample: 
        raise ValueError('computeErrorFunction:lagProblem','lag must be smaller than nSample')

    # initial error calculation 
    # loop over lags
    for ll in range(-lag,lag + 1):
        thisLag = ll + lag 

        # loop over samples 
        for ii in range(nSample):
            
            # skip corners for now, we will come back to these
            if (ii + ll >= 0) & (ii + ll < nSample):
                err[ii,thisLag] = u1[ii] - u0[ii + ll]

    
    if norm == 'L2':
        err = err**2
    elif norm == 'L1':
        err = np.abs(err)

    # Now fix corners with constant extrapolation
    for ll in range(-lag,lag + 1):
        thisLag = ll + lag 

        for ii in range(nSample):
            if ii + ll < 0:
                err[ii, thisLag] = err[-ll, thisLag]

            elif ii + ll > nSample - 1:
                err[ii,thisLag] = err[nSample - ll - 1,thisLag]
    
    return err

cpdef accumulateErrorFunction(int dir, 
                              np.ndarray[np.float64_t, ndim=2] err, 
                              int nSample, 
                              int lag, 
                              int b):
    """
    USAGE: d = accumulation_diw_mod( dir, err, nSample, lag, b )

    INPUT:
        dir = accumulation direction ( dir > 0 = forward in time, dir <= 0 = backward in time)
        err = the 2D error function; size = (nsamp,2*lag+1)
        nSample = numer of points to compare in the traces
        lag = maximum lag in sample number to search
        b = strain limit (integer value >= 1)
    OUTPUT:
        d = the 2D distance function; size = (nsamp,2*lag+1)
    
    The function is equation 6 in Hale, 2013.

    Original by Di Yang
    Last modified by Dylan Mikesell (25 Feb. 2015)

    Translated to python by Tim Clements (17 Aug. 2018)

    """

    # allocate variables
    cdef int iBegin, iEnd, iInc, ii, ji, jb, ll, lMinus1, lPlus1, kb, count
    # number of lags from [ -lag : +lag ]
    cdef int nLag = ( 2 * lag ) + 1
    cdef float distLminus1, distL, distLplus1
    cdef np.ndarray[np.float64_t, ndim=2] d = np.zeros([nSample, nLag], dtype=DTYPE64)

    # Setup indices based on forward or backward accumulation direction
    if dir > 0: # FORWARD
        iBegin, iEnd, iInc = 0, nSample - 1, 1
    else: # BACKWARD
        iBegin, iEnd, iInc = nSample - 1, 0, -1 

    # Loop through all times ii in forward or backward direction
    for count in range(nSample):

        ii = iBegin + count * iInc

        # min/max to account for the edges/boundaries
        ji = max([0, min([nSample - 1, ii - iInc])])
        jb = max([0, min([nSample - 1, ii - iInc * b])])

        # loop through all lag 
        for ll in range(nLag):

            # check limits on lag indices 
            lMinus1 = ll - 1

            # check lag index is greater than 0
            if lMinus1 < 0:
                lMinus1 = 0 # make lag = first lag

            lPlus1 = ll + 1# lag at l+1
            
            # check lag index less than max lag
            if lPlus1 > nLag - 1: 
                lPlus1 = nLag - 1
            
            # get distance at lags (ll-1, ll, ll+1)
            distLminus1 = d[jb, lMinus1] # minus:  d[i-b, j-1]
            distL = d[ji,ll] # actual d[i-1, j]
            distLplus1 = d[jb, lPlus1] # plus d[i-b, j+1]

            if ji != jb: # equation 10 in Hale, 2013
                for kb in range(ji,jb + iInc - 1, -iInc): 
                    distLminus1 += err[kb, lMinus1]
                    distLplus1 += err[kb, lPlus1]
            
            # equation 6 (if b=1) or 10 (if b>1) in Hale (2013) after treating boundaries
            d[ii, ll] = err[ii,ll] + min([distLminus1, distL, distLplus1])

    return d


cpdef backtrackDistanceFunction(int dir,
                                np.ndarray[np.float64_t, ndim=2] d, 
                                np.ndarray[np.float64_t, ndim=2] err, 
                                int lmin, 
                                int b):
    """
    USAGE: stbar = backtrackDistanceFunction( dir, d, err, lmin, b )

    INPUT:
        dir   = side to start minimization ( dir > 0 = front, dir <= 0 =  back)
        d     = the 2D distance function; size = (nsamp,2*lag+1)
        err   = the 2D error function; size = (nsamp,2*lag+1)
        lmin  = minimum lag to search over
        b     = strain limit (integer value >= 1)
    OUTPUT:
        stbar = vector of integer shifts subject to |u(i)-u(i-1)| <= 1/b

    The function is equation 2 in Hale, 2013.

    Original by Di Yang
    Last modified by Dylan Mikesell (19 Dec. 2014)

    Translated to python by Tim Clements (17 Aug. 2018)

    """

    cdef int nSample, nLag, iBegin, iEnd, iInc, ll, ii, ji, jb, lMinus1, lPlus1, kb
    cdef float distLminus1, distL, distLplus1, dl, 
    nSample, nLag = d.shape[0], d.shape[1]
    cdef np.ndarray[np.float64_t, ndim=1] stbar = np.zeros(nSample)

    # Setup indices based on forward or backward accumulation direction
    if dir > 0: # FORWARD
        iBegin, iEnd, iInc = 0, nSample - 1, 1
    else: # BACKWARD
        iBegin, iEnd, iInc = nSample - 1, 0, -1 

    # start from the end (front or back)
    ll = np.argmin(d[iBegin,:]) # find minimum accumulated distance at front or back depending on 'dir'
    stbar[iBegin] = ll + lmin # absolute value of integer shift

    # move through all time samples in forward or backward direction
    ii = iBegin

    while ii != iEnd: 

        # min/max for edges/boundaries
        ji = max([0, min([nSample - 1, ii + iInc])])
        jb = max([0, min([nSample - 1, ii + iInc * b])])

        # check limits on lag indices 
        lMinus1 = ll - 1

        if lMinus1 < 0: # check lag index is greater than 1
            lMinus1 = 0 # make lag = first lag

        lPlus1 = ll + 1

        if lPlus1 > nLag - 1: # check lag index less than max lag
            lPlus1 = nLag - 1

        # get distance at lags (ll-1, ll, ll+1)
        distLminus1 = d[jb, lMinus1] # minus:  d[i-b, j-1]
        distL = d[ji,ll] # actual d[i-1, j]
        distLplus1 = d[jb, lPlus1] # plus d[i-b, j+1]

        # equation 10 in Hale (2013)
        # sum errors over i-1:i-b+1
        if ji != jb:
            for kb in range(ji, jb - iInc - 1, iInc):
                distLminus1 = distLminus1 + err[kb, lMinus1]
                distLplus1  = distLplus1  + err[kb, lPlus1]
        
        # update minimum distance to previous sample
        dl = min([distLminus1, distL, distLplus1 ])

        if dl != distL: # then ll ~= ll and we check forward and backward
            if dl == distLminus1:
                ll = lMinus1
            else: 
                ll = lPlus1
        
        # assume ii = ii - 1
        ii += iInc 

        # absolute integer of lag
        stbar[ii] = ll + lmin 

        # now move to correct time index, if smoothing difference over many
        # time samples using 'b'
        if (ll == lMinus1) | (ll == lPlus1): # check edges to see about b values
            if ji != jb: # if b>1 then need to move more steps
                for kb in range(ji, jb - iInc - 1, iInc):
                    ii = ii + iInc # move from i-1:i-b-1
                    stbar[ii] = ll + lmin  # constant lag over that time

    return stbar


cpdef computeDTWerror(np.ndarray[np.float64_t, ndim=2] Aerr, 
                      np.ndarray[np.int_t, ndim=1] u, 
                      int lag0):
    """

    Compute the accumulated error along the warping path for Dynamic Time Warping.

    USAGE: function error = computeDTWerror( Aerr, u, lag0 )

    INPUT:
        Aerr = error MATRIX (equation 13 in Hale, 2013)
        u    = warping function (samples) VECTOR
        lag0 = value of maximum lag (samples) SCALAR

    Written by Dylan Mikesell
    Last modified: 25 February 2015
    Translated to python by Tim Clements (17 Aug. 2018)
    """

    cdef int npts, ii, idx
    cdef float error 

    npts = len(u)

    if Aerr.shape[0] != npts:
        print('Funny things with dimensions of error matrix: check inputs.')
        Aerr = Aerr.T

    error = 0.
    for ii in range(npts):
        idx = lag0 + 1 + u[ii] # index of lag 
        error = error + Aerr[ii,idx]

    return error 
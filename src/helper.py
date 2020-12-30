import numpy as np

def make_sdf(sT, sID, allID, t0, tmax, dt, sigma):
    n= int((tmax-t0)/dt)
    sdfs= np.zeros((len(allID),n))
    kwdt= 3*sigma
    i= 0
    x= np.arange(-kwdt,kwdt,dt)
    x= np.exp(-np.power(x,2)/(2*sigma*sigma))
    while i < len(sT):
        left= int((sT[i]-t0-kwdt)/dt)
        right= int((sT[i]-t0+kwdt)/dt)
        sdfs[sID[i],left:right]+=x
        i= i+1

    return sdfs
                  

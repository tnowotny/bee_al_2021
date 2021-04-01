import matplotlib.pyplot as plt
import numpy as np
from helper import *
import sys

def plot_raster(spike_t, spike_ID,t_total,N,bname,display= True):
    plt.figure()
    if spike_t is not None:
        plt.plot(spike_t, spike_ID, '.', markersize= 1)
    plt.title(bname)
    plt.xlim([0, t_total])
    plt.ylim([0, N])
    plt.savefig(bname+"_spikes.png",dpi=300)
    if display:
        plt.show()

if __name__ == "__main__":

    argv= sys.argv
    if len(argv) != 4:
        print("usage: python plot_raster.py <base name> <t_total> <N>")
        exit()

    bname= argv[1]
    t_total= float(argv[2])
    N= int(argv[3])
    spike_t= np.load(bname+"_spike_t.npy")
    spike_ID= np.load(bname+"_spike_ID.npy")
    plot_raster(spike_t, spike_ID, t_total, N, bname)
    

import matplotlib.pyplot as plt
import numpy as np
from helper import *
import sys

def plot_raster(spike_t, spike_ID,t_total,N,label,pop,display= True):
    plt.figure()
    if spike_t is not None:
        plt.plot(spike_t, spike_ID, '.', markersize= 1)
    plt.title(pop)
    plt.xlim([0, t_total])
    plt.ylim([0, N])
    plt.savefig(label+"_"+pop+"_spikes.png",dpi=300)
    if display:
        plt.show()

if __name__ == "__main__":

    argv= sys.argv
    if len(argv) != 5:
        print("usage: python plot_raster.py <label> <pop name> <t_total> <N>")
        exit()

    label= argv[1]
    pop= argv[2]
    t_total= float(argv[3])
    N= int(argv[4])
    spike_t= np.load(label+"_"+pop+"_spike_t.npy")
    spike_ID= np.load(label+"_"+pop+"_spike_ID.npy")
    plot_raster(spike_t, spike_ID, t_total, N, label, pop)
    
import matplotlib.pyplot as plt
import numpy as np
from helper import *
import sys

def plot_avg_sdf(spike_t, spike_ID,t_total,N,label,pop,dt_sdf= 5.0, sigma_sdf= 20.0, percentile= 100, display= True):
    sdfs= make_sdf(spike_t, spike_ID, np.arange(0,N), -3*sigma_sdf, t_total+3*sigma_sdf, dt_sdf, sigma_sdf)
    plt.figure()
    plt.imshow(sdfs, extent=[-3*sigma_sdf,t_total+3*sigma_sdf,0,N], aspect='auto')
    plt.title(pop)
    plt.colorbar()
    plt.savefig(label+"_"+pop+"_sdfmap.png",dpi=300)

    if percentile < 100:
        t_avg= np.mean(sdfs,1)
        p= np.percentile(t_avg,percentile)
        print(p)
        print(t_avg)
        ind= np.argwhere(t_avg > p).flatten()
        plt.figure()
        print(ind)
        plt.plot(ind, t_avg[ind],'o')
        plt.gca().set_xlim(0,N)
        avgsdf= np.mean(sdfs[ind,:],0)
        print(sdfs.shape)
        print(avgsdf.shape)
    else:
        avgsdf= np.mean(sdfs,0)
    plt.figure()
    plt.plot(avgsdf)
    print(avgsdf.shape)
    plt.gca().set_title("average SDF of "+pop)
    plt.savefig(label+"_"+pop+"_avgsdf.png",dpi=300)
    if display:
        plt.show()
    
if __name__ == "__main__":

    argv= sys.argv
    if len(argv) != 6:
        print("usage: python plot_avg_sdf.py <label> <pop name> <t_total> <N> <percentile>")
        exit()

    label= argv[1]
    pop= argv[2]
    t_total= float(argv[3])
    N= int(argv[4])
    perc= float(argv[5])
    spike_t= np.load(label+"_"+pop+"_spike_t.npy")
    spike_ID= np.load(label+"_"+pop+"_spike_ID.npy")
    plot_avg_sdf(spike_t, spike_ID, t_total, N, label, pop, dt_sdf= 2, sigma_sdf= 50.0, percentile= perc)

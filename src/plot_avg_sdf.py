import matplotlib.pyplot as plt
import numpy as np
from helper import *
import sys

def plot_avg_sdf(spike_t, spike_ID, t0, t_max,N,bname,dt_sdf= 5.0, sigma_sdf= 20.0, percentile= 100, display= True):
    sdfs= make_sdf(spike_t, spike_ID, np.arange(0,N), t0, t_max, dt_sdf, sigma_sdf)
    plt.figure()
    plt.imshow(sdfs, extent=[t0,t_max,0,N], aspect='auto')
    plt.title(bname)
    plt.colorbar()
    plt.savefig(bname+"_sdfmap.png",dpi=300)

    avgsdf= []
    if percentile < 100:
        wds= int((t_max-t0)//30000)
        # extract percentile of highest responders per odour width episode
        for i in range(wds):
            left= (i*30000)//dt_sdf
            right= left+30000//dt_sdf
            lsdfs= sdfs[:, left:right]
            t_avg= np.mean(lsdfs,1)
            p= np.percentile(t_avg,percentile)
            print(p)
            ind= np.argwhere(t_avg > p).flatten()
            mn= np.mean(lsdfs[ind,:],0)
            avgsdf.append(mn)
        tavgsdf= np.hstack(avgsdf)
    else:
        tavgsdf= np.mean(sdfs,0)
    plt.figure()
    plt.plot(tavgsdf)
    print(tavgsdf.shape)
    plt.gca().set_title("average SDF of "+bname)
    plt.savefig(bname+"_avgsdf.png",dpi=300)
    if display:
        plt.show()
    return sdfs

if __name__ == "__main__":

    argv= sys.argv
    if len(argv) != 6:
        print("usage: python plot_avg_sdf.py <base name> <t0> <t_max> <N> <percentile>")
        exit()

    bname= argv[1]
    t0= float(argv[2])
    t_max= float(argv[3])
    N= int(argv[4])
    perc= float(argv[5])
    spike_t= np.load(bname+"_spike_t.npy")
    spike_ID= np.load(bname+"_spike_ID.npy")
    plot_avg_sdf(spike_t, spike_ID, t0, t_max, N, bname, dt_sdf= 1, sigma_sdf= 50.0, percentile= perc)

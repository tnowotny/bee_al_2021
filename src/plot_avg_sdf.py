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

    avgsdf= []
    if percentile < 100:
        wds= int(t_total//30000)
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
    plt.gca().set_title("average SDF of "+pop)
    plt.savefig(label+"_"+pop+"_avgsdf.png",dpi=300)
    if display:
        plt.show()
    return sdfs

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
    plot_avg_sdf(spike_t, spike_ID, t_total, N, label, pop, dt_sdf= 1, sigma_sdf= 50.0, percentile= perc)

import matplotlib.pyplot as plt
import numpy as np
from helper import *

def exp1_plots(state_bufs, spike_t, spike_ID, plot_raster, plot_sdf, t_total, n_glo, n, N, dirname, label):
    dt_sdf= 1.0
    sigma_sdf= 300.0

    # Doing some plots
    for pop in plot_raster:
        plt.figure()
        if spike_t[pop] is not None:
            plt.plot(spike_t[pop], spike_ID[pop], '.', markersize= 1)
        plt.title(pop)
        plt.xlim([0, t_total])
        plt.ylim([0, N[pop]])
        plt.savefig(dirname+label+"_spikes_"+pop+".png",dpi=300)

    #figure, axes= plt.subplots(3)

    #for i in range(78,82):
    #    axes[0].plot(t_array, state_bufs["ORs_ra"][:,i])
    #    axes[1].plot(t_array, state_bufs["ORNs_V"][:,i*n["ORNs"]])
    #    axes[2].plot(t_array, state_bufs["ORNs_a"][:,i*n["ORNs"]])
    #plt.savefig(dirname+label+"_rawTraces"+".png",dpi=300)

    #plt.figure()
    #plt.imshow(np.transpose(state_bufs["ORs_ra"]), extent=[0,50000,0,160], aspect='auto')
    #plt.title("OR")
    #plt.colorbar()
    #plt.savefig(dirname+label+"_ORcmap"+".png",dpi=300)

    #plt.figure()
    #for i in range(78,82):
    #    plt.plot(t_array, state_bufs["PNs_V"][:,i*n["PNs"]])
    #    plt.title("PN")

    
    for pop in plot_sdf:
        sdfs= make_sdf(spike_t[pop], spike_ID[pop], np.arange(0,N[pop]), -3*sigma_sdf, t_total+3*sigma_sdf, dt_sdf, sigma_sdf)
        plt.figure()
        plt.imshow(sdfs, extent=[-3*sigma_sdf,t_total+3*sigma_sdf,0,N[pop]], aspect='auto')
        plt.title(pop)
        plt.colorbar()
        plt.savefig(dirname+label+"_"+pop+"_sdfmap.png",dpi=300)

        if n[pop] > 1:
            sdfs= make_sdf(spike_t[pop], spike_ID[pop]//n[pop], np.arange(0,n_glo), -3*sigma_sdf, t_total+3*sigma_sdf, dt_sdf, sigma_sdf)
            print(sdfs.shape)
            plt.figure()
            plt.imshow(sdfs/n[pop], extent=[-3*sigma_sdf,t_total+3*sigma_sdf,0,n_glo], aspect='auto')
            plt.title("average SDF of "+pop+" in each glomerulus")
            plt.colorbar()
            plt.savefig(dirname+label+"_"+pop+"_glo_sdfmap.png",dpi=300)
    
        figure, axes= plt.subplots(len(plot_sdf[pop]), sharey= True)
        t_array= np.arange(-3*sigma_sdf, t_total+3*sigma_sdf, dt_sdf)
        j= 0
        for i in plot_sdf[pop]:
            axes[j].plot(t_array, sdfs[i,:]/n[pop])
            j= j+1
        axes[0].set_title("average SDF of "+pop+" in each glomerulus")
        plt.savefig(dirname+label+"_"+pop+"_sdftraces.png",dpi=300)

    # Show plot
    plt.show()

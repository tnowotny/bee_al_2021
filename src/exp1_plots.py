import matplotlib.pyplot as plt
import numpy as np
from helper import *

def exp1_plots(state_bufs, spike_t, spike_ID, paras, display= True):
    dirname= paras["dirname"]+"/"
    dt_sdf= 1.0
    sigma_sdf= 300.0
    
    # Doing some plots
    for pop in paras["plot_raster"]:
        plt.figure()
        if spike_t[pop] is not None:
            plt.plot(spike_t[pop], spike_ID[pop], '.', markersize= 1)
        plt.title(pop)
        plt.xlim([0, paras["t_total"]])
        plt.ylim([0, paras["N"][pop]])
        plt.savefig(dirname+paras["label"]+"_spikes_"+pop+".png",dpi=300)

    state_plot= False
    if state_plot:
        figure, axes= plt.subplots(3)
        t_array= np.arange(0, paras["t_total"], paras["dt"])
        for i in range(78,82):
            axes[0].plot(t_array, state_bufs["ORs_ra"][:,i])
            axes[1].plot(t_array, state_bufs["ORNs_V"][:,i*paras["n"]["ORNs"]])
            axes[2].plot(t_array, state_bufs["ORNs_a"][:,i*paras["n"]["ORNs"]])
        plt.savefig(paras["dirname"]+paras["label"]+"_rawTraces"+".png",dpi=300)

        plt.figure()
        plt.imshow(np.transpose(state_bufs["ORs_ra"]), extent=[0,50000,0,160], aspect='auto')
        plt.title("OR")
        plt.colorbar()
        plt.savefig(dirname+label+"_ORcmap"+".png",dpi=300)

        plt.figure()
        for i in range(78,82):
            plt.plot(t_array, state_bufs["PNs_V"][:,i*paras["n"]["PNs"]])
            plt.title("PN")

    
    for pop in paras["plot_sdf"]:
        sdfs= make_sdf(spike_t[pop], spike_ID[pop], np.arange(0,paras["N"][pop]), 0, paras["t_total"], dt_sdf, sigma_sdf)
        plt.figure()
        plt.imshow(sdfs, extent=[-3*sigma_sdf,paras["t_total"]+3*sigma_sdf,0,paras["N"][pop]], aspect='auto')
        plt.title(pop)
        plt.colorbar()
        plt.savefig(paras["dirname"]+paras["label"]+"_"+pop+"_sdfmap.png",dpi=300)

        if paras["n"][pop] > 1:
            sdfs= make_sdf(spike_t[pop], spike_ID[pop]//paras["n"][pop], np.arange(0,paras["n_glo"]), 0, paras["t_total"], dt_sdf, sigma_sdf)
            plt.figure()
            plt.imshow(sdfs/paras["n"][pop], extent=[-3*sigma_sdf,paras["t_total"]+3*sigma_sdf,0,paras["n_glo"]], aspect='auto')
            plt.title("average SDF of "+pop+" in each glomerulus")
            plt.colorbar()
            plt.savefig(paras["dirname"]+paras["label"]+"_"+pop+"_glo_sdfmap.png",dpi=300)
    
        figure, axes= plt.subplots(len(paras["plot_sdf"][pop]), sharey= True)
        t_array= np.arange(-3*sigma_sdf, paras["t_total"]+3*sigma_sdf, dt_sdf)
        j= 0
        for i in paras["plot_sdf"][pop]:
            axes[j].plot(t_array, sdfs[i,:]/paras["n"][pop])
            j= j+1
        axes[0].set_title("average SDF of "+pop+" in each glomerulus")
        plt.savefig(paras["dirname"]+paras["label"]+"_"+pop+"_sdftraces.png",dpi=300)

    # Show plot
    if display:
        plt.show()

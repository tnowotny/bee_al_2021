import numpy as np
from helper import *
from exp1 import *
from exp1_plots import exp1_plots

# Prepare variables for recording results
state_bufs= dict()
state_pops= np.unique([k for k,i in rec_state])
# file= open(dirname+label+"_t.bin", "rb")
# t_array= np.load(file)
for pop, var in rec_state:
    lbl= pop+"_"+var
    file= open(dirname+label+"_"+lbl+".bin", "rb")
    state_bufs[lbl]= np.load(file, )
    file.close()

spike_t= dict()
spike_ID= dict()
for pop in rec_spikes:
    file= open(dirname+label+pop+"_spike_t.bin", "rb")
    spike_t[pop]= np.load(file)
    file.close()
    file= open(dirname+label+pop+"_spike_ID.bin", "rb")
    spike_ID[pop]= np.load(file)
    file.close()

exp1_plots(state_bufs, spike_t, spike_ID, plot_raster, plot_sdf, t_total, n_glo, n, N, dirname, label)

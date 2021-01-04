import numpy as np
from helper import *
from exp1 import *
from exp1_plots import exp1_plots

# Prepare variables for recording results
state_bufs= dict()
state_pops= np.unique([k for k,i in rec_state])
# file= open(dirname+label+"_t.bin", "rb")
# t_array= np.load(file)
dirname= dirname+"/"
for pop, var in rec_state:
    lbl= pop+"_"+var
    state_bufs[lbl]= np.load(dirname+label+"_"+lbl+".bin")

spike_t= dict()
spike_ID= dict()
for pop in rec_spikes:
    spike_t[pop]= np.load(dirname+label+pop+"_spike_t.bin")
    spike_ID[pop]= np.load(dirname+label+pop+"_spike_ID.bin")

exp1_plots(state_bufs, spike_t, spike_ID, plot_raster, plot_sdf, t_total, dt, n_glo, n, N, dirname, label)

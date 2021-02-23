import numpy as np
from helper import *
from exp1 import *
from sim import *
from exp1_plots import exp1_plots

# Prepare variables for recording results
state_bufs= dict()
state_pops= np.unique([k for k,i in paras["rec_state"]])
# file= open(dirname+label+"_t.bin", "rb")
# t_array= np.load(file)
dirname= paras["dirname"]+"/"
for pop, var in paras["rec_state"]:
    lbl= pop+"_"+var
    state_bufs[lbl]= np.load(dirname+paras["label"]+"_"+lbl+".npy")

spike_t= dict()
spike_ID= dict()
for pop in paras["rec_spikes"]:
    spike_t[pop]= np.load(dirname+paras["label"]+"_"+pop+"_spike_t.npy")
    spike_ID[pop]= np.load(dirname+paras["label"]+"_"+pop+"_spike_ID.npy")

exp1_plots(state_bufs, spike_t, spike_ID, paras)

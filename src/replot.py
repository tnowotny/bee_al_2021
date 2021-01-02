import numpy as np
from helper import *
from exp1 import *

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

with open('exp1_plots.py') as f: exec(f.read())

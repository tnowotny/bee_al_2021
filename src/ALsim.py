import numpy as np
import matplotlib.pyplot as plt

from pygenn import genn_wrapper
from pygenn.genn_model import GeNNModel, init_connectivity

from OR import (or_model, or_params, or_ini)
from synapse import *
from neuron import *
from helper import *

from exp1 import *

# Create a single-precision GeNN model
model = GeNNModel("double", "honeyAL"#, backend="SingleThreadedCPU"
                  )

# Set simulation timestep to 0.1ms
model.dT = 0.5

# Add neuron populations and current source to model
ors = model.add_neuron_population("ORs", n_glo, or_model, or_params, or_ini)
orns = model.add_neuron_population("ORNs", n_glo*n["ORNs"], adaptive_LIF, orn_params, orn_ini)
if n["PNs"] > 0:
    pns= model.add_neuron_population("PNs", n_glo*n["PNs"], adaptive_LIF, pn_params, pn_ini)
if n["LNs"] > 0:
    lns= model.add_neuron_population("LNs", n_glo*n["LNs"], adaptive_LIF, ln_params, ln_ini)

# Connect ORs to ORNs
ors_orns = model.add_synapse_population("ORs_ORNs", "SPARSE_GLOBALG", genn_wrapper.NO_DELAY,
                                        ors, orns,
                                        pass_or, {}, {}, {}, {},
                                        pass_postsyn, {}, {},
                                        init_connectivity(ors_orns_connect, {})
                                        )

# Connect ORNs to PNs
if n["PNs"] > 0:
    orns_pns = model.add_synapse_population("ORNs_PNs", "SPARSE_GLOBALG", genn_wrapper.NO_DELAY,
                                            orns, pns,
                                            "StaticPulse", {}, orns_pns_ini, {}, {},
                                            "ExpCond", orns_pns_post_params, {},
                                            init_connectivity(orns_al_connect, {"n_orn": n["ORNs"], "n_trg": n["PNs"]})
                                            )

# Connect ORNs to LNs
if n["LNs"] > 0:
    orns_lns = model.add_synapse_population("ORNs_LNs", "SPARSE_GLOBALG", genn_wrapper.NO_DELAY,
                                            orns, lns,
                                            "StaticPulse", {}, orns_lns_ini, {}, {},
                                            "ExpCond", orns_lns_post_params, {},
                                            init_connectivity(orns_al_connect, {"n_orn": n["ORNs"], "n_trg": n["LNs"]})
                                            )
    

# Connect PNs to LNs
if n["LNs"] > 0 and n["PNs"] > 0:
    lns_pns =  model.add_synapse_population("PNs_LNs", "SPARSE_GLOBALG", genn_wrapper.NO_DELAY,
                                            pns, lns,
                                            "StaticPulse", {}, pns_lns_ini, {}, {},
                                            "ExpCond", pns_lns_post_params, {},
                                            init_connectivity(pns_lns_connect, {"n_pn": n["PNs"], "n_ln": n["LNs"]})
                                            )
# Connect LNs to PNs
if n["LNs"] > 0 and n["PNs"] > 0:
    lns_pns =  model.add_synapse_population("LNs_PNs", "DENSE_INDIVIDUALG", genn_wrapper.NO_DELAY,
                                            lns, pns,
                                            "StaticPulse", {}, lns_pns_ini, {}, {},
                                            "ExpCond", lns_pns_post_params, {}
                                            )
    
                                            
# Connect LNs to LNs
if n["LNs"] > 0:
    lns_lns =  model.add_synapse_population("LNs_LNs", "DENSE_INDIVIDUALG", genn_wrapper.NO_DELAY,
                                            lns, lns,
                                            "StaticPulse", {}, lns_lns_ini, {}, {},
                                            "ExpCond", lns_lns_post_params, {}
                                            )

    print("building model ...");
# Build and load model
model.build()
model.load()

if n["LNs"] > 0 and n["PNs"] > 0:
    # do not inhibit own PN
    g= lns_pns.vars["g"].view
    for i in range(0,n_glo):
        off1= i*n["LNs"]    # first row of this glomerulus   
        for j in range(off1, off1+n["LNs"]):
            off2= i*n["PNs"]    # first column of this glomerulus    
            for k in range(off2, off2+n["PNs"]):
                id= j*(n_glo*n["PNs"])+k
                g[id]= 0.0

if n["LNs"] > 0:
    # do not inhibit LN in own glomerulus
    g= lns_lns.vars["g"].view
    for i in range(0,n_glo):
        off1= i*n["LNs"]    # first row of this glomerulus   
        for j in range(off1, off1+n["LNs"]):
            off2= i*n["LNs"]    # first column of this glomerulus    
            for k in range(off2, off2+n["LNs"]):
                id= j*(n_glo*n["LNs"])+k
                g[id]= 0.0
    
# Prepare variables for recording results
state_views= dict()
state_bufs= dict()
state_pops= np.unique([k for k,i in rec_state])
for pop, var in rec_state:
    lbl= pop+"_"+var
    state_views[lbl]= model.neuron_populations[pop].vars[var].view
    state_bufs[lbl]= None

spike_t= dict()
spike_ID= dict()
for pop in rec_spikes:
    spike_t[pop]= None
    spike_ID[pop]= None
    
# Simulate
prot_pos= 0
while model.t < t_total:
    if prot_pos < len(protocol) and model.t >= protocol[prot_pos]["t"]:
        tp= protocol[prot_pos]
        set_odor_simple(ors, tp["ochn"], odors[tp["odor"],:], tp["concentration"], hill_exp)
        model.push_state_to_device("ORs")
        prot_pos+= 1
    model.step_time()
    if int(model.t/model.dT)%1000 == 0:
        print(model.t)
    
    for pop in state_pops:
        model.pull_state_from_device(pop)
    
    for p in state_bufs:
        state_bufs[p]= np.copy(state_views[p]) if state_bufs[p] is None else np.vstack((state_bufs[p], state_views[p]))

    for pop in rec_spikes:
        the_pop= model.neuron_populations[pop]
        the_pop.pull_current_spikes_from_device()
        if (the_pop.spike_count[0] > 0):
            ln= the_pop.spike_count[0]
            spike_t[pop]= np.copy(model.t*np.ones(ln)) if spike_t[pop] is None else np.hstack((spike_t[pop], model.t*np.ones(ln)))
            spike_ID[pop]= np.copy(the_pop.spikes[0:ln]) if spike_ID[pop] is None else np.hstack((spike_ID[pop], the_pop.spikes[0:ln]))


# Saving results
if state_bufs:
    # only save the time array if anything is being saved
    t_array= np.arange(0.0,t_total,model.dT)
    file= open(dirname+label+"_t.bin", "wb")
    np.save(file, t_array)
    for p in state_bufs:
        file= open(dirname+label+"_"+p+".bin", "wb")
        np.save(file, state_bufs[p])
        file.close()
        
for pop in rec_spikes:
    file= open(dirname+label+pop+"_spike_t.bin", "wb")
    np.save(file, spike_t[pop])
    file.close()
    file= open(dirname+label+pop+"_spike_ID.bin", "wb")
    np.save(file, spike_ID[pop])
    file.close()

with open('exp1_plots.py') as f: exec(f.read())

import numpy as np
import matplotlib.pyplot as plt

from pygenn import genn_wrapper
from pygenn.genn_model import GeNNModel, init_connectivity

from OR import (or_model, or_params, or_ini)
from synapse import *
from neuron import *
from helper import *
import time
import os

# prepare writing results sensibly
timestr = time.strftime("%Y-%m-%d")
dirname= timestr+"-runs"
path = os.path.isdir(dirname)
if not path:
    os.makedirs(dirname)
dirname=dirname+"/"

# Create a single-precision GeNN model
model = GeNNModel("double", "honeyAL"#, backend="SingleThreadedCPU"
                  )

# Set simulation timestep to 0.1ms
model.dT = 0.5


# number of glomeruli
#n_glo= 2
# number of ORNs per glomerulus
#n_orn= 600
# number of PNs per glomerulus
#n_pn= 5
# number of LNs per glomerulus
#n_ln= 1

from exp1 import *

# Add neuron populations and current source to model
ors = model.add_neuron_population("ORs", n_glo, or_model, or_params, or_ini)
orns = model.add_neuron_population("ORNs", n_glo*n_orn, adaptive_LIF, orn_params, orn_ini)
if n_pn > 0:
    pns= model.add_neuron_population("PNs", n_glo*n_pn, adaptive_LIF, pn_params, pn_ini)
if n_ln > 0:
    lns= model.add_neuron_population("LNs", n_glo*n_ln, adaptive_LIF, ln_params, ln_ini)

# Connect ORs to ORNs
ors_orns = model.add_synapse_population("ORs_ORNs", "SPARSE_GLOBALG", genn_wrapper.NO_DELAY,
                                        ors, orns,
                                        pass_or, {}, {}, {}, {},
                                        pass_postsyn, {}, {},
                                        init_connectivity(ors_orns_connect, {})
                                        )

# Connect ORNs to PNs
if n_pn > 0:
    orns_pns = model.add_synapse_population("ORNs_PNs", "SPARSE_GLOBALG", genn_wrapper.NO_DELAY,
                                            orns, pns,
                                            "StaticPulse", {}, orns_pns_ini, {}, {},
                                            "ExpCond", orns_pns_post_params, {},
                                            init_connectivity(orns_al_connect, {"n_orn": n_orn, "n_trg": n_pn})
                                            )

# Connect ORNs to LNs
if n_ln > 0:
    orns_lns = model.add_synapse_population("ORNs_LNs", "SPARSE_GLOBALG", genn_wrapper.NO_DELAY,
                                            orns, lns,
                                            "StaticPulse", {}, orns_lns_ini, {}, {},
                                            "ExpCond", orns_lns_post_params, {},
                                            init_connectivity(orns_al_connect, {"n_orn": n_orn, "n_trg": n_ln})
                                            )
    

# Connect PNs to LNs
if n_ln > 0 and n_pn > 0:
    lns_pns =  model.add_synapse_population("PNs_LNs", "SPARSE_GLOBALG", genn_wrapper.NO_DELAY,
                                            pns, lns,
                                            "StaticPulse", {}, pns_lns_ini, {}, {},
                                            "ExpCond", pns_lns_post_params, {},
                                            init_connectivity(pns_lns_connect, {"n_pn": n_pn, "n_ln": n_ln})
                                            )
# Connect LNs to PNs
if n_ln > 0 and n_pn > 0:
    lns_pns =  model.add_synapse_population("LNs_PNs", "DENSE_INDIVIDUALG", genn_wrapper.NO_DELAY,
                                            lns, pns,
                                            "StaticPulse", {}, lns_pns_ini, {}, {},
                                            "ExpCond", lns_pns_post_params, {}
                                            )
                                            

# Build and load model
model.build()
model.load()

# Create a numpy view to efficiently access the membrane voltage from Python
r0_view = ors.vars["r0"].view
ra_view = ors.vars["ra"].view
v_view= orns.vars["V"].view
a_view= orns.vars["a"].view

# Simulate
r0 = None
ra = None
v= None
a= None
ornSpkt= None
ornSpkID= None
t_total= 5000.0
cstr= "1e-3_n05"
while model.t < t_total:
    if (np.abs(model.t - 1000.0) < 1e-5):
        set_odor_simple(ors, "0", odors[0,:], 1e-3, 0.5)
        model.push_state_to_device("ORs")
        print("odor set")
    if (np.abs(model.t - 4000.0) < 1e-5):
        set_odor_simple(ors, "0", odors[0,:], 0.0, 0.5)
        model.push_state_to_device("ORs")
        print("odor removed")
    model.step_time()
    if int(model.t/model.dT)%1000 == 0:
        print(model.t)
    model.pull_state_from_device("ORs")
    model.pull_state_from_device("ORNs")
    r0 = np.copy(r0_view) if r0 is None else np.vstack((r0, r0_view))
    ra = np.copy(ra_view) if ra is None else np.vstack((ra, ra_view))
    v= np.copy(v_view) if v is None else np.vstack((v, v_view))
    a= np.copy(a_view) if a is None else np.vstack((a, a_view))
    orns.pull_current_spikes_from_device()
    if (orns.spike_count[0] > 0):
        n= orns.spike_count[0]
        ornSpkt= np.copy(model.t*np.ones(n)) if ornSpkt is None else np.hstack((ornSpkt, model.t*np.ones(n)))
        ornSpkID= np.copy(orns.spikes[0:n]) if ornSpkID is None else np.hstack((ornSpkID, orns.spikes[0:n]))

plt.figure
plt.plot(ornSpkt, ornSpkID, '.')
plt.savefig(dirname+"spikes_"+cstr+".png",dpi=300)

figure, axes= plt.subplots(3)

t_array= np.arange(0.0, t_total, model.dT)
for i in range(78,82):
    axes[0].plot(t_array, ra[:,i])
    axes[0].plot(t_array, r0[:,i])
    axes[1].plot(t_array, v[:,i*n_orn])
    axes[2].plot(t_array, a[:,i*n_orn])
plt.savefig(dirname+"rawTraces_"+cstr+".png",dpi=300)

plt.figure()
plt.imshow(np.transpose(ra), extent=[0,50000,0,160], aspect='auto')
plt.colorbar()
plt.savefig(dirname+"ORcmap_"+cstr+".png",dpi=300)

file = open(dirname+"ornSpkt_"+cstr+".bin", "wb")
np.save(file, ornSpkt)
file.close
file = open(dirname+"ornSpkID_"+cstr+".bin", "wb")
np.save(file, ornSpkID)
file.close

sigma= 100.0
NORN= n_glo*n_orn
sdfs= make_sdf(ornSpkt, ornSpkID, np.arange(0,NORN), -3*sigma, t_total+3*sigma, 1.0, sigma)
plt.figure()
plt.imshow(sdfs, extent=[-3*sigma,t_total+3*sigma,0,NORN], aspect='auto')
plt.colorbar()
plt.savefig("ORNsdfmap_"+cstr+".png",dpi=300)

figure, axes= plt.subplots(6, sharey= True)
t_array= np.arange(-3*sigma, t_total+3*sigma, 1.0)
j= 0
for i in range(74*n_orn,86*n_orn,2*n_orn):
    axes[j].plot(t_array, sdfs[i,:])
    j= j+1
plt.savefig("ornSDFtraces_"+cstr+".png",dpi=300)

# Show plot
# plt.show()

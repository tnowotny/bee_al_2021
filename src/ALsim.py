import numpy as np
import matplotlib.pyplot as plt

from pygenn import genn_wrapper
from pygenn.genn_model import GeNNModel, init_connectivity

from OR import (or_model, or_params, or_ini)
from synapse import *
from neuron import *

# Create a single-precision GeNN model
model = GeNNModel("double", "honeyAL", backend="SingleThreadedCPU")

# Set simulation timestep to 0.1ms
model.dT = 0.1


# number of glomeruli
#n_glo= 2
# number of ORNs per glomerulus
#n_orn= 600
# number of PNs per glomerulus
#n_pn= 5
# number of LNs per glomerulus
#n_ln= 1

from exp1 import *
print(n_orn)

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
                                            init_connectivity(orns_pns_connect, {"n_orn": n_orn, "n_pn": n_pn})
                                            )
# Connect ORNs to LNs
if n_ln > 0:
    orns_lns = model.add_synapse_population("ORNs_LNs", "SPARSE_GLOBALG", genn_wrapper.NO_DELAY,
                                            orns, lns,
                                            "StaticPulse", {}, orns_lns_ini, {}, {},
                                            "ExpCond", orns_lns_post_params, {}
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
t_total= 5000.0
while model.t < t_total:
    if (np.abs(model.t - 1000.0) < 1e-5):
        set_odor_simple(ors, "0", odors[0,:], 1e-6, 1)
        model.push_state_to_device("ORs")
        print("odor set")
    model.step_time()
    print(model.t)
    model.pull_state_from_device("ORs")
    model.pull_state_from_device("ORNs")
    r0 = np.copy(r0_view) if r0 is None else np.vstack((r0, r0_view))
    ra = np.copy(ra_view) if ra is None else np.vstack((ra, ra_view))
    v= np.copy(v_view) if v is None else np.vstack((v, v_view))
    a= np.copy(a_view) if a is None else np.vstack((a, a_view))

figure, axes= plt.subplots(3)

t_array= np.arange(0.0, t_total, 0.1)
for i in range(78,82):
    axes[0].plot(t_array, ra[:,i])
    axes[0].plot(t_array, r0[:,i])
    axes[1].plot(t_array, v[:,i*n_orn],'.')
    axes[2].plot(t_array, a[:,i*n_orn],'.')


plt.figure()
plt.imshow(ra, extent=[0,50000,0,160], aspect='auto')
    # Create plot
#figure, axes = plt.subplots(4, sharex=True)

# Plot voltages
#for i, t in enumerate(["RS", "FS", "CH", "IB"]):
#    axes[i].set_title(t)
#    axes[i].set_ylabel("V [mV]")
#    axes[i].plot(np.arange(0.0, 200.0, 0.1), v[:,i])
#axes[-1].set_xlabel("Time [ms]")

# Show plot
plt.show()

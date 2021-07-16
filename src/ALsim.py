import numpy as np
import matplotlib.pyplot as plt
import os
import json

from pygenn import genn_wrapper
from pygenn.genn_model import GeNNModel, init_connectivity, init_var

from OR import or_model
from synapse import pass_or, pass_postsyn, ors_orns_connect, orns_al_connect, pns_lns_connect, lns_pns_conn_init, lns_lns_conn_init
from neuron import adaptive_LIF
from helper import set_odor_simple
import sim

def ALsim(odors, hill_exp, paras, protocol, lns_gsyn= None):
    path = os.path.isdir(paras["dirname"])
    if not path:
        print("making dir "+paras["dirname"])
        os.makedirs(paras["dirname"])
    dirname=paras["dirname"]+"/"

    with open(dirname+paras["label"]+".para","w") as f:
        f.write(json.dumps(paras))
        f.write("\n")
        f.close()
        
    # Create a single-precision GeNN model
    model = GeNNModel("double", "honeyAL"#, backend="SingleThreadedCPU"
    )

    # Set simulation timestep to 0.1ms
    model.dT = sim.dt
    
    # Add neuron populations to model
    ors = model.add_neuron_population("ORs", paras["n_glo"], or_model, paras["or_params"], paras["or_ini"])
    orns = model.add_neuron_population("ORNs", paras["n_glo"]*paras["n"]["ORNs"], adaptive_LIF, paras["orn_params"], paras["orn_ini"])
    if paras["use_spk_rec"]:
        if "ORNs" in paras["rec_spikes"]:
            orns.spike_recording_enabled= True

    if paras["n"]["PNs"] > 0:
        pns= model.add_neuron_population("PNs", paras["n_glo"]*paras["n"]["PNs"], adaptive_LIF, paras["pn_params"], paras["pn_ini"])
        if paras["use_spk_rec"]:
            if "PNs" in paras["rec_spikes"]:
                pns.spike_recording_enabled= True
            
    if paras["n"]["LNs"] > 0:
        lns= model.add_neuron_population("LNs", paras["n_glo"]*paras["n"]["LNs"], adaptive_LIF, paras["ln_params"], paras["ln_ini"])
        if paras["use_spk_rec"]:
            if "LNs" in paras["rec_spikes"]:
                lns.spike_recording_enabled= True

    # Connect ORs to ORNs
    ors_orns = model.add_synapse_population("ORs_ORNs", "SPARSE_GLOBALG", genn_wrapper.NO_DELAY,
                                            ors, orns,
                                            pass_or, {}, {}, {}, {},
                                            pass_postsyn, {}, {},
                                            init_connectivity(ors_orns_connect, {})
                                            )

    # Connect ORNs to PNs
    if paras["n"]["PNs"] > 0:
        orns_pns = model.add_synapse_population("ORNs_PNs", "SPARSE_GLOBALG", genn_wrapper.NO_DELAY,
                                                orns, pns,
                                                "StaticPulse", {}, paras["orns_pns_ini"], {}, {},
                                                "ExpCond", paras["orns_pns_post_params"], {},
                                                init_connectivity(orns_al_connect, {"n_orn": paras["n"]["ORNs"], "n_trg": paras["n"]["PNs"], "n_pre": paras["n_orn_pn"]})
                                                )

    # Connect ORNs to LNs
    if paras["n"]["LNs"] > 0:
        orns_lns = model.add_synapse_population("ORNs_LNs", "SPARSE_GLOBALG", genn_wrapper.NO_DELAY,
                                                orns, lns,
                                                "StaticPulse", {}, paras["orns_lns_ini"], {}, {},
                                                "ExpCond", paras["orns_lns_post_params"], {},
                                                init_connectivity(orns_al_connect, {"n_orn": paras["n"]["ORNs"], "n_trg": paras["n"]["LNs"], "n_pre": paras["n_orn_ln"]})
                                                )
    # Connect PNs to LNs
    if paras["n"]["LNs"] > 0 and paras["n"]["PNs"] > 0:
        pns_lns =  model.add_synapse_population("PNs_LNs", "SPARSE_GLOBALG", genn_wrapper.NO_DELAY,
                                                pns, lns,
                                                "StaticPulse", {}, paras["pns_lns_ini"], {}, {},
                                                "ExpCond", paras["pns_lns_post_params"], {},
                                                init_connectivity(pns_lns_connect, {"n_pn": paras["n"]["PNs"], "n_ln": paras["n"]["LNs"]})
                                                )
    # Connect LNs to PNs
    if paras["n"]["LNs"] > 0 and paras["n"]["PNs"] > 0:
        if lns_gsyn is not None:
            the_lns_gsyn= np.repeat(lns_gsyn, repeats=paras["n"]["LNs"], axis=0)
            the_lns_gsyn= np.repeat(the_lns_gsyn, repeats=paras["n"]["PNs"], axis=1)
            the_lns_gsyn*= paras["lns_pns_g"]
            the_lns_gsyn= np.reshape(the_lns_gsyn, paras["n"]["LNs"]*paras["n"]["PNs"]*paras["n_glo"]*paras["n_glo"])
            lns_pns =  model.add_synapse_population("LNs_PNs", "DENSE_INDIVIDUALG", genn_wrapper.NO_DELAY,
                                                lns, pns,
                                                "StaticPulse", {}, {"g": the_lns_gsyn}, {}, {},
                                                "ExpCond", paras["lns_pns_post_params"], {}
                                                )
            print("Set explicit LN -> PN inhibition matrix")
        else:
            lns_pns =  model.add_synapse_population("LNs_PNs", "DENSE_INDIVIDUALG", genn_wrapper.NO_DELAY,
                                                lns, pns,
                                                "StaticPulse", {}, {"g": init_var(lns_pns_conn_init, {"n_pn": paras["n"]["PNs"],"n_ln": paras["n"]["LNs"],"g": paras["lns_pns_g"]})}, {}, {},
                                                "ExpCond", paras["lns_pns_post_params"], {}
                                                )
            print("Set homogeneous LN -> PN inhibition matrix using initvar snippet")
    # Connect LNs to LNs
    if paras["n"]["LNs"] > 0:
        if lns_gsyn is not None:
            the_lns_gsyn= np.repeat(lns_gsyn, repeats=paras["n"]["LNs"], axis=0)
            the_lns_gsyn= np.repeat(the_lns_gsyn, repeats=paras["n"]["LNs"], axis=1)
            the_lns_gsyn*= paras["lns_lns_g"]
            the_lns_gsyn= np.reshape(the_lns_gsyn, paras["n"]["LNs"]*paras["n"]["LNs"]*paras["n_glo"]*paras["n_glo"])
            lns_lns =  model.add_synapse_population("LNs_LNs", "DENSE_INDIVIDUALG", genn_wrapper.NO_DELAY,
                                                lns, lns,
                                                "StaticPulse", {}, {"g": the_lns_gsyn}, {}, {},
                                                "ExpCond", paras["lns_lns_post_params"], {}
                                                )
            print("Set explicit LN -> LN inhibition matrix")
        else:
            lns_lns =  model.add_synapse_population("LNs_LNs", "DENSE_INDIVIDUALG", genn_wrapper.NO_DELAY,
                                                lns, lns,
                                                "StaticPulse", {}, {"g": init_var(lns_lns_conn_init,{"n_ln": paras["n"]["LNs"],"g": paras["lns_lns_g"]})}, {}, {},
                                                "ExpCond", paras["lns_lns_post_params"], {}
                                                )
            print("Set homogeneous LN -> LN inhibition matrix using initvar snippet")
    print("building model ...");
    # Build and load model
    model.build()
    model.load(num_recording_timesteps= paras["spk_rec_steps"])
    
    # Prepare variables for recording results
    state_views= dict()
    state_bufs= dict()
    state_pops= np.unique([k for k,i in paras["rec_state"]])
    for pop, var in paras["rec_state"]:
        lbl= pop+"_"+var
        state_views[lbl]= model.neuron_populations[pop].vars[var].view
        state_bufs[lbl]= []

    spike_t= dict()
    spike_ID= dict()
    ORN_cnts= []
    for pop in paras["rec_spikes"]:
        if pop != "ORNs":
            spike_t[pop]= []
            spike_ID[pop]= []
    
    # Simulate
    prot_pos= 0
    int_t= 0
    while model.t < paras["t_total"]:
        while prot_pos < len(protocol) and model.t >= protocol[prot_pos]["t"]:
            tp= protocol[prot_pos]
            set_odor_simple(ors, tp["ochn"], odors[tp["odor"],:,:], tp["concentration"], hill_exp)
            model.push_state_to_device("ORs")
            prot_pos+= 1
        model.step_time()
        if int_t%1000 == 0:
            print(model.t)

        int_t+= 1
        # for pop in state_pops:
        #     model.pull_state_from_device(pop)
        for pop, var in paras["rec_state"]:
            model.neuron_populations[pop].pull_var_from_device(var)
    
        for p in state_bufs:
            state_bufs[p].append(np.copy(state_views[p])) 

        if paras["use_spk_rec"]:
            if int_t%paras["spk_rec_steps"] == 0:
                model.pull_recording_buffers_from_device()
                for pop in paras["rec_spikes"]:
                    the_pop= model.neuron_populations[pop]
                    if pop == "ORNs":
                        # only record total spike number
                        ORN_cnts.append(len(the_pop.spike_recording_data[0]))
                    else:
                        spike_t[pop].append(the_pop.spike_recording_data[0])
                        spike_ID[pop].append(the_pop.spike_recording_data[1])
                print("fetched spikes from buffer ... complete")
        else:
            for pop in paras["rec_spikes"]:
                the_pop= model.neuron_populations[pop]
                the_pop.pull_current_spikes_from_device()
                if pop == "ORNs":
                    # only record total spike number
                    ORN_cnts.append(the_pop.spike_count[0])
                else:
                    if (the_pop.spike_count[0] > 0):
                        ln= the_pop.spike_count[0]
                        spike_t[pop].append(np.copy(model.t*np.ones(ln))) 
                        spike_ID[pop].append(np.copy(the_pop.spikes[0:ln]))

    # Saving results
    if paras["write_to_disk"]:
        if state_bufs:
            # only save the time array if anything is being saved
            t_array= np.arange(0.0,paras["t_total"],model.dT)
            np.save(dirname+paras["label"]+"_t", t_array)
            for p in state_bufs:
                state_bufs[p]= np.vstack(state_bufs[p])
                np.save(dirname+paras["label"]+"_"+p, state_bufs[p])

        for pop in paras["rec_spikes"]:
            if pop == "ORNs":
                ORN_cnts= np.hstack(ORN_cnts)
                np.save(dirname+paras["label"]+"_"+pop+"_spike_counts",ORN_cnts)
            else:
                spike_t[pop]= np.hstack(spike_t[pop])
                np.save(dirname+paras["label"]+"_"+pop+"_spike_t", spike_t[pop])
                spike_ID[pop]= np.hstack(spike_ID[pop])
                np.save(dirname+paras["label"]+"_"+pop+"_spike_ID", spike_ID[pop])

    return state_bufs, spike_t, spike_ID, ORN_cnts


import numpy as np
import matplotlib.pyplot as plt
import time
from helper import *
from exp1_plots import exp1_plots
from ALsim import ALsim


from ALsimParameters import std_paras

paras= std_paras()
paras["plotting"]= True
paras["use_spk_rec"]= True

# write results into a dir with current date in the name
timestr = time.strftime("%Y-%m-%d")
paras["dirname"]= timestr+"-runs"

paras["t_total"]= 30000.0

paras["rec_state"]= [
    ("ORs", "ra"),
#    ("ORNs", "V"),
#    ("ORNs", "a"),
#    ("PNs", "V")
]

paras["rec_spikes"]= [
    "ORNs",
    "PNs",
    "LNs"
    ]

paras["plot_raster"]= [
    "ORNs",
    "PNs",
    "LNs"
    ]

paras["plot_sdf"]= {
    "ORNs": list(range(74,86,2)),
    "PNs": list(range(74,87,2)),
    "LNs": list(range(74,87,2))
    }

paras["label"]= "simple_test"
# let's assume that sensible values are 0.5 to 1.5

hill_exp= np.load("2021-02-22-runs-jamest/test_sig_hill.npy")

od= gauss_odor(paras["n_glo"], 80, 30)
odors= od
od= gauss_odor(paras["n_glo"], 80, 40)
odors= np.vstack((odors, od))
   
paras["protocol"]= [
    {
        "t": 1000.0,
        "odor": 0,
        "ochn": "0",
        "concentration": 1e-6
        },
    {
        "t": 4000.0,
        "odor": 0,
        "ochn": "0",
        "concentration": 0.0
        },
    {
        "t": 6000.0,
        "odor": 0,
        "ochn": "0",
        "concentration": 1e-5
        },
    {
        "t": 9000.0,
        "odor": 0,
        "ochn": "0",
        "concentration": 0.0
        },
    {
        "t": 11000.0,
        "odor": 0,
        "ochn": "0",
        "concentration": 1e-4
        },
    {
        "t": 14000.0,
        "odor": 0,
        "ochn": "0",
        "concentration": 0.0
        },
    {
        "t": 16000.0,
        "odor": 0,
        "ochn": "0",
        "concentration": 1e-3
        },
    {
        "t": 19000.0,
        "odor": 0,
        "ochn": "0",
        "concentration": 0.0
        },
    {
        "t": 21000.0,
        "odor": 0,
        "ochn": "0",
        "concentration": 1e-2
        },
    {
        "t": 24000.0,
        "odor": 0,
        "ochn": "0",
        "concentration": 0.0
        },
    {
        "t": 26000.0,
        "odor": 0,
        "ochn": "0",
        "concentration": 1e-1
        },
    {
        "t": 29000.0,
        "odor": 0,
        "ochn": "0",
        "concentration": 0.0
        }
     ]

paras["lns_pns_g"]= 0.0   
paras["lns_lns_g"]= 0.0

if __name__ == "__main__":
    state_bufs, spike_t, spike_ID= ALsim(odors, hill_exp, paras)

    if paras["plotting"]:
        exp1_plots(state_bufs, spike_t, spike_ID, paras)

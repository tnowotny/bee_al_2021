import numpy as np
import matplotlib.pyplot as plt
import time
from helper import *
from exp1_plots import exp1_plots
from ALsim import *
import sim


# write results into a dir with current date in the name
timestr = time.strftime("%Y-%m-%d")
dirname= timestr+"-runs"

t_total= 30000.0

rec_state= [
    ("ORs", "ra"),
#    ("ORNs", "V"),
#    ("ORNs", "a"),
#    ("PNs", "V")
]

rec_spikes= [
    "ORNs",
    "PNs",
    "LNs"
    ]

plot_raster= [
    "ORNs",
    "PNs",
    "LNs"
    ]

plot_sdf= {
    "ORNs": range(74,86,2),
    "PNs": range(74,87,2),
    "LNs": range(74,87,2)
    }

label= "1e-6_n07"
hill_exp= 2

od= gauss_odor(n_glo, 80, 10)
odors= od
od= gauss_odor(n_glo, 90, 10)
odors= np.vstack((odors, od))
   
protocol= [
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

if __name__ == "__main__":
    state_bufs, spike_t, spike_ID= ALsim(n_glo, n, N, t_total, sim.dt, rec_state, rec_spikes, odors, hill_exp, protocol, dirname, label)

    exp1_plots(state_bufs, spike_t, spike_ID, plot_raster, plot_sdf, t_total, sim.dt, n_glo, n, N, dirname, label)

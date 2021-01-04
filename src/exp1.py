import numpy as np
import matplotlib.pyplot as plt
import time
from helper import *
from exp1_plots import exp1_plots
from ALsim import ALsim

# write results into a dir with current date in the name
timestr = time.strftime("%Y-%m-%d")
dirname= timestr+"-runs"

n_glo= 160
n= {
    "ORNs": 60,
    "PNs": 5,
    "LNs": 1
    }

N= {
    "ORNs": n_glo*n["ORNs"],
    "PNs": n_glo*n["PNs"],
    "LNs": n_glo*n["LNs"]
}


t_total= 20000.0
dt= 0.5

rec_state= [
#    ("ORs", "ra"),
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
hill_exp= 0.7

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
        }
    ]

if __name__ == "__main__":
    state_bufs, spike_t, spike_ID= ALsim(n_glo, n, N, t_total, dt, rec_state, rec_spikes, odors, hill_exp, protocol, dirname, label)

    exp1_plots(state_bufs, spike_t, spike_ID, plot_raster, plot_sdf, t_total, dt, n_glo, n, N, dirname, label)

import numpy as np
import matplotlib.pyplot as plt
import time
from helper import *
from exp1_plots import exp1_plots
from ALsim import *
import sim

"""
experiment to investigate the effect of decreasing response with higher concentration for a single odor but for different breadth of odour activation from very narrow to very broad
"""

# write results into a dir with current date in the name
timestr = time.strftime("%Y-%m-%d")
dirname= timestr+"-runs"

plotting= False
plotdisplay= False



rec_state= [
#    ("ORs", "ra"),
#    ("ORNs", "V"),
#    ("ORNs", "a"),
#    ("PNs", "V")
]

rec_spikes= [
#    "ORNs",
    "PNs",
    "LNs"
    ]

plot_raster= [
#    "ORNs",
    "PNs",
    "LNs"
    ]

plot_sdf= {
#    "ORNs": range(74,86,2),
    "PNs": range(74,87,2),
    "LNs": range(74,87,2)
    }

label= "test_sig"

# Assume a uniform distribution of Hill coefficients inspired by Rospars'
# work on receptors tiling the space of possible sigmoid responses
hill_exp= np.random.uniform(0.6, 1.0, n_glo)
print(hill_exp)

# Let's do a progression of broadening odours
odors= None
odor_sigma= 0.2
oNo= 30  # how many odors to try
for i in range(oNo):
    od= gauss_odor(n_glo, 80, odor_sigma)
    odor_sigma*= 1.1
    if odors is None:
        odors= od
    else:
        odors= np.vstack((odors, od))

# Now, let's make a protocol where they are presented for 3 secs with
# 3 second breaks
protocol= []
t_off= 3000.0
for i in range(oNo):
    for c in range(4):
        sub_prot= {
            "t": t_off,
            "odor": i,
            "ochn": "0",
            "concentration": 1e-6*np.power(10,c),
        }
        protocol.append(sub_prot)
        sub_prot= {
            "t": t_off+3000.0,
            "odor": i,
            "ochn": "0",
            "concentration": 0.0,
        }
        protocol.append(sub_prot)
        t_off+= 6000.0;

print(protocol)

t_total= t_off

if __name__ == "__main__":
    state_bufs, spike_t, spike_ID= ALsim(n_glo, n, N, t_total, sim.dt, rec_state, rec_spikes, odors, hill_exp, protocol, dirname, label)

    if plotting:
        exp1_plots(state_bufs, spike_t, spike_ID, plot_raster, plot_sdf, t_total, sim.dt, n_glo, n, N, dirname, label, display=plotdisplay)

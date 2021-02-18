import numpy as np
import matplotlib.pyplot as plt
import time
from helper import *
from exp1_plots import exp1_plots
from ALsim import *
import sim
import sys

"""
experiment to investigate the effect of decreasing response with higher concentration for a single odor but for different breadth of odour activation from very narrow to very broad
"""

if len(sys.argv) < 2:
    print("usage: python exp2.py <run#>")
    exit()

ino= int(sys.argv[1])
print("lns_pns_g: %f" % lns_pns_g)
lns_pns_g*= 0.1*np.power(1.2,ino)
print("lns_pns_g: %f" % lns_pns_g)

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

hill_new= True

if hill_new:
    hill_exp= np.random.uniform(0.5, 1.5, n_glo)
    np.save(dirname+"/"+label+"_"+ino+"_hill",hill_exp)
else:
    hill_exp= np.load(dirname+"/"+label+"_"+ino+"_hill.npy")
print(hill_exp)

# Let's do a progression of broadening odours
odors= None
odor_sigma= 0.2
oNo= 30  # how many odors to try
for i in range(oNo):
    od= gauss_odor(n_glo, 80, odor_sigma)
    odor_sigma*= 1.2
    if odors is None:
        odors= od
    else:
        odors= np.vstack((odors, od))

# Now, let's make a protocol where they are presented for 3 secs with
# 3 second breaks
protocol= []
t_off= 3000.0
base= np.power(10,0.2)
for i in range(oNo):
    for c in range(25):
        sub_prot= {
            "t": t_off,
            "odor": i,
            "ochn": "0",
            "concentration": 1e-6*np.power(base,c),
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
    state_bufs, spike_t, spike_ID= ALsim(n_glo, n, N, t_total, sim.dt, rec_state, rec_spikes, odors, hill_exp, protocol, dirname, label+"_"+ino)

    if plotting:
        exp1_plots(state_bufs, spike_t, spike_ID, plot_raster, plot_sdf, t_total, sim.dt, n_glo, n, N, dirname, label+"_"+ino, display=plotdisplay)

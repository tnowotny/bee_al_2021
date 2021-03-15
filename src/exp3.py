import numpy as np
import matplotlib.pyplot as plt
import time
from helper import *
from exp1_plots import exp1_plots
from ALsim import ALsim
import sim
import sys
import os
from ALsimParameters import std_paras

"""
experiment to investigate the effect of decreasing response with higher concentration for a single odor but for different breadth of odour activation from very narrow to very broad
"""

if len(sys.argv) < 2:
    print("usage: python exp2.py <run#>")
    exit()

ino= int(sys.argv[1])
paras= std_paras()

if ino == -1:
    paras["lns_pns_g"]= 0
    paras["lns_lns_g"]= 0
else:
    paras["lns_pns_g"]*= 0.1*np.power(np.sqrt(10),ino)
    paras["lns_lns_g"]*= 0.1*np.power(np.sqrt(10),ino)

# write results into a dir with current date in the name
timestr = time.strftime("%Y-%m-%d")
paras["dirname"]= timestr+"-runs"

path = os.path.isdir(paras["dirname"])
if not path:
    print("making dir "+paras["dirname"])
    os.makedirs(paras["dirname"])

paras["plotting"]= False
paras["plotdisplay"]= False
paras["use_spk_rec"]= True

paras["rec_state"]= [
#    ("ORs", "ra"),
#    ("ORNs", "V"),
#    ("ORNs", "a"),
#    ("PNs", "V")
]

paras["rec_spikes"]= [
#    "ORNs",
    "PNs",
    "LNs"
    ]

paras["plot_raster"]= [
#    "ORNs",
    "PNs",
    "LNs"
    ]

paras["plot_sdf"]= {
#    "ORNs": range(74,86,2),
    "PNs": list(range(74,87,2)),
    "LNs": list(range(74,87,2))
    }

label= "test_sig"
paras["label"]= label+"_"+str(ino)

# Assume a uniform distribution of Hill coefficients inspired by Rospars'
# work on receptors tiling the space of possible sigmoid responses

hill_new= True

if hill_new:
    hill_exp= np.random.uniform(0.5, 1.5, paras["n_glo"])
    np.save(paras["dirname"]+"/"+label+"_hill",hill_exp)
else:
    hill_exp= np.load(paras["dirname"]+"/"+label+"_hill.npy")

# Let's do a progression of broadening odours
odor_new= True

if odor_new:
    odors= None
    odor_sigma= np.array([ 1.0, 10.0 ])
    oNo= len(odor_sigma)  # how many odors to try
    increment= 160//oNo
    centre= 0
    for i in range(oNo):
        od= gauss_odor(paras["n_glo"], centre, odor_sigma[i])
        if odors is None:
            odors= od
        else:
            odors= np.vstack((odors, od))
        centre= centre+increment
    np.save(paras["dirname"]+"/"+label+"_odors",odors)
else:
    odors= np.load(paras["dirname"]+"/"+label+"_odors.npy")
    oNo= odors.shape[0]

# Now, let's make a protocol where they are presented for 3 secs with
# 3 second breaks
paras["protocol"]= []
t_off= 3000.0
base= np.power(10,0.25)
c= [ 0, 0]
for c[0] in range(24):
    for c[1] in range(24):
        for i in range(oNo):
            sub_prot= {
                "t": t_off,
                "odor": i,
                "ochn": str(i),
                "concentration": 1e-7*np.power(base,c[i]),
            }
            paras["protocol"].append(sub_prot)
        for i in range(oNo):
            sub_prot= {
                "t": t_off+3000.0,
                "odor": i,
                "ochn": str(i),
                "concentration": 0.0,
            }
            paras["protocol"].append(sub_prot)
        t_off+= 6000.0;

paras["t_total"]= t_off

if __name__ == "__main__":
    state_bufs, spike_t, spike_ID= ALsim(odors, hill_exp, paras)

    if paras["plotting"]:
        exp1_plots(state_bufs, spike_t, spike_ID, paras, display=plotdisplay)

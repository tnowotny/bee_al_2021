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
import random

"""
experiment to investigate the effect of decreasing response with higher concentration. In this version we generate N_odour odours randomly with the following properties: 
1. Each odour has a a Gaussian profile of glomerulus activation with sigma drawn from Gauss(mu_sig,sig_sig). 
2. the Gaussian odour profile is over a random permutation of the glomeruli.
"""

paras= std_paras()
paras["version"]= "correlations and clipped Gaussians; branch clipped_gauss, commit c3c41aa9a33f3effa2cde7cee06538b07793bc46 (HEAD -> clipped_gauss, origin/clipped_gauss)"
paras["N_odour"]= 100
paras["mu_sig"]= 8
paras["sig_sig"]= 2
paras["odor_clip"]= 0.05

if len(sys.argv) < 2:
    print("usage: python exp4.py <run#>")
    exit()

ino= float(sys.argv[1])

if ino == -100:
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

label= "test_new"
paras["label"]= label+"_"+str(ino)

# Assume a uniform distribution of Hill coefficients inspired by Rospars'
# work on receptors tiling the space of possible sigmoid responses

hill_new= False

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
    for i in range(paras["N_odour"]):
        sigma= random.gauss(paras["mu_sig"],paras["sig_sig"])
        od= clipped_gauss_odor(paras["n_glo"], 0, sigma, paras["odor_clip"])
        random.shuffle(od)
        print(od)
        if odors is None:
            odors= np.copy(od)
        else:
            odors= np.vstack((odors, np.copy(od)))
    np.save(paras["dirname"]+"/"+label+"_odors",odors)
else:
    odors= np.load(paras["dirname"]+"/"+label+"_odors.npy")
    oNo= odors.shape[0]

correl= np.corrcoef(odors,rowvar=False)
correl= correl/2.0+1.0
for i in range(paras["n_glo"]):
    correl[i,i]= 0.0

# Now, let's make a protocol where each odor is presented for 3 secs with
# 3 second breaks and at each of 24 concentration values
paras["protocol"]= []
t_off= 3000.0
base= np.power(10,0.25)

for i in range(paras["N_odour"]):
    for c in range(24):
        sub_prot= {
            "t": t_off,
            "odor": i,
            "ochn": str(0),
            "concentration": 1e-7*np.power(base,c),
        }
        paras["protocol"].append(sub_prot)
        sub_prot= {
            "t": t_off+3000.0,
            "odor": i,
            "ochn": str(0),
            "concentration": 0.0,
        }
        paras["protocol"].append(sub_prot)
        t_off+= 6000.0;

paras["t_total"]= t_off
print("We are running for a total simulated time of {}ms".format(t_off))

if __name__ == "__main__":
    state_bufs, spike_t, spike_ID= ALsim(odors, hill_exp, paras, lns_gsyn= correl)

    if paras["plotting"]:
        exp1_plots(state_bufs, spike_t, spike_ID, paras, display=plotdisplay)

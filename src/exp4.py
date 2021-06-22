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
paras["N_odour"]= 100
paras["mu_sig"]= 5
paras["sig_sig"]= 0.5
paras["min_sig"]= 3
paras["odor_clip"]= 0.05
paras["act_rate_min"]= 0.01
paras["act_rate_max"]= 0.03

if len(sys.argv) < 3:
    print("usage: python exp4.py <run#> <connect_I: corr0/corr1/cov0/cov1")
    exit()

ino= float(sys.argv[1])
connect_I= sys.argv[2]

if ino == -100:
    paras["lns_pns_g"]= 0
    paras["lns_lns_g"]= 0
else:
    paras["lns_pns_g"]*= np.power(np.sqrt(10),ino)
    paras["lns_lns_g"]*= np.power(np.sqrt(10),ino)

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
#    "LNs"
    ]

paras["plot_raster"]= [
#    "ORNs",
    "PNs",
#    "LNs"
    ]

paras["plot_sdf"]= {
#    "ORNs": range(74,86,2),
    "PNs": list(range(74,87,2)),
#    "LNs": list(range(74,87,2))
    }

label= "test_new"
paras["label"]= label+"_"+connect_I+"_"+str(ino)

# Assume a uniform distribution of Hill coefficients inspired by Rospars'
# work on receptors tiling the space of possible sigmoid responses

hill_new= True

if hill_new:
    hill_exp= np.random.uniform(0.7, 1.2, paras["n_glo"])
    np.save(paras["dirname"]+"/"+label+"_hill",hill_exp)
else:
    hill_exp= np.load(paras["dirname"]+"/"+label+"_hill.npy")

# Let's do a progression of broadening odours
odor_new= True

if odor_new:
    odors= []
    for i in range(paras["N_odour"]-3):
        sigma= 0
        while sigma < paras["min_sig"]:
            sigma= random.gauss(paras["mu_sig"],paras["sig_sig"])
        od= gauss_odor(paras["n_glo"], 0, sigma, paras["odor_clip"], paras["act_rate_min"], paras["act_rate_max"])
        random.shuffle(od)
        odors.append(np.copy(od))
    odors= np.array(odors)
    np.save(paras["dirname"]+"/"+label+"_odors",odors)
else:
    odors= np.load(paras["dirname"]+"/"+label+"_odors.npy")

if connect_I == "corr0":
    correl= np.corrcoef(odors[:,:,0].reshape(paras["N_odour"]-3,paras["n_glo"]),rowvar=False)
    correl= (correl+1.0)/20.0 # extra factor 10 in comparison to covariance ...
    for i in range(paras["n_glo"]):
        correl[i,i]= 0.0
    print("AL inhibition with correlation, no self-inhibition")
else:
    if connect_I == "corr1":
        correl= np.corrcoef(odors[:,:,0].reshape(paras["N_odour"]-3,paras["n_glo"]),rowvar=False)
        correl= (correl+1.0)/20.0  # extra factor 10 in comparison to covariance ...
        print("AL inhibition with correlation and self-inhibition")
    else:
        if connect_I == "cov0":
            correl= np.cov(odors[:,:,0].reshape(paras["N_odour"]-3,paras["n_glo"]),rowvar=False)
            correl= np.maximum(0.0, correl)
            for i in range(paras["n_glo"]):
                correl[i,i]= 0.0
            print("AL inhibition with covariance, no self-inhibition")
        else:
            correl= np.cov(odors[:,:,0].reshape(paras["N_odour"]-3,paras["n_glo"]),rowvar=False)
            correl= np.maximum(0.0, correl)
            print("AL inhibition with covariance and self-inhibition")

# let's make 3 extra odours: 5, 10, 15 wide. Each shall contain the most inhibited glomeruli
csum= np.sum(correl,axis= 1)
idx= np.argsort(csum)
for sigma in [ 5, 10, 15 ]:
    od= gauss_odor(paras["n_glo"], 0, sigma, paras["odor_clip"], 0.01, 0.01)
    # the extra "diagnostic" odors all activate very weakly
    sod= np.sort(od[:,0])
    od[idx,0]= sod
    odors= np.vstack((np.reshape(np.copy(od),(1,paras["n_glo"],2)),odors))

print(odors.shape)
print(odors[:,0,1])

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

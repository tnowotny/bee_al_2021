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
if len(sys.argv) < 3:
    print("usage: python exp4.py <run#> <connect_I: hom/corr0/corr1/cov0/cov1")
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
    hill_exp= np.random.uniform(0.7, 0.8, paras["n_glo"])
    np.save(paras["dirname"]+"/"+label+"_hill",hill_exp)
else:
    hill_exp= np.load(paras["dirname"]+"/"+label+"_hill.npy")

# Generate odors or load previously generated odors from file
odor_new= True

if odor_new:
    odors= []
    for i in range(paras["N_odour"]-1):
        sigma= 0
        while sigma < paras["min_sig"]:
            sigma= np.random.normal(paras["mu_sig"],paras["sig_sig"])
        A= -100.0
        while A < paras["min_A"] or A > paras["max_A"]:
            A= np.random.normal(paras["mean_A"], paras["sig_A"])
        od= gauss_odor(paras["n_glo"], 0, sigma, A, paras["odor_clip"], paras["mean_act"], paras["sig_act"],paras["min_act"],paras["max_act"])
        random.shuffle(od)
        odors.append(np.copy(od))
    # One "diagnostic odor" that is particularly early binding, broad, and low activating
    sigma= 15
    A= paras["max_A"]
    act= 0.01
    od= gauss_odor(paras["n_glo"], paras["n_glo"]//2, sigma, A, paras["odor_clip"], act, 0.0, 0.01)
    odors.append(np.copy(od))
    odors= np.array(odors)
    np.save(paras["dirname"]+"/"+label+"_odors",odors)
else:
    odors= np.load(paras["dirname"]+"/"+label+"_odors.npy")
    oNo= odors.shape[0]


# define the inhibitory connectivity pattern in the antennal lobe either homogeneous (equal strength)
# for HOM_LN_GSYN= True or according to correlations (corr0 without self-inhibition, corr1 with self-inhibition)
# or according to covariance (cov0 without self-inhibition, cov1 with self-inhibition)
HOMO_LN_GSYN= False
if connect_I == "corr0":
    correl= np.corrcoef(odors[:,:,0].reshape(paras["N_odour"],paras["n_glo"]),rowvar=False)
    correl= (correl+1.0)/20.0 # extra factor 10 in comparison to covariance ...
    for i in range(paras["n_glo"]):
        correl[i,i]= 0.0
    print("AL inhibition with correlation, no self-inhibition")
else:
    if connect_I == "corr1":
        correl= np.corrcoef(odors[:,:,0].reshape(paras["N_odour"],paras["n_glo"]),rowvar=False)
        correl= (correl+1.0)/20.0  # extra factor 10 in comparison to covariance ...
        print("AL inhibition with correlation and self-inhibition")
    else:
        if connect_I == "cov0":
            correl= np.cov(odors[:,:,0].reshape(paras["N_odour"],paras["n_glo"]),rowvar=False)
            correl= np.maximum(0.0, correl)
            for i in range(paras["n_glo"]):
                correl[i,i]= 0.0
            print("AL inhibition with covariance, no self-inhibition")
        else:
            if connect_I == "cov1":
                correl= np.cov(odors[:,:,0].reshape(paras["N_odour"],paras["n_glo"]),rowvar=False)
                correl= np.maximum(0.0, correl)
                print("AL inhibition with covariance and self-inhibition")
            else:
                correl= np.ones((paras["n_glo"],paras["n_glo"]))
                HOMO_LN_GSYN= True
                print("Homogeneous AL inhibition")
                
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
    if HOMO_LN_GSYN:
        state_bufs, spike_t, spike_ID= ALsim(odors, hill_exp, paras)
    else:
        state_bufs, spike_t, spike_ID= ALsim(odors, hill_exp, paras, lns_gsyn= correl)

    if paras["plotting"]:
        exp1_plots(state_bufs, spike_t, spike_ID, paras, display=plotdisplay)

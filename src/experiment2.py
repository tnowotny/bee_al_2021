import numpy as np
import matplotlib.pyplot as plt
import time
from helper import *
from ALsim import ALsim
import sim
import sys
import os
from ALsimParameters import std_paras
import random

"""
Experiment to investigate the binary mixtured of two odors. Odors are defined as in experiment1.py.
"""

if len(sys.argv) < 5:
    print("usage: python experiment2.py <ino> <connect_I: corr0/corr1/cov0/cov1> <odor 1> <odor 2>" )
    exit()

ino= float(sys.argv[1])
connect_I= sys.argv[2]
o1= int(sys.argv[3])
o2= int(sys.argv[4])

paras= std_paras()

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

paras["use_spk_rec"]= True

paras["rec_state"]= [
    ("ORs", "ra"),
#    ("ORNs", "V"),
#    ("ORNs", "a"),
#    ("PNs", "V")
]

paras["rec_spikes"]= [
    "ORNs",
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


label= "run"
paras["label"]= label+"_"+str(ino)+"_odors_"+str(o1)+"_"+str(o2)

# Load odors and Hill coefficients from file. This experiment assumes that these have been
# already generated with experiment1.py
hill_exp= np.load(paras["dirname"]+"/"+label+"_hill.npy")
odors= np.load(paras["dirname"]+"/"+label+"_odors.npy")
paras["N_odour"]= odors.shape[0]

if connect_I == "corr0":
    correl= np.corrcoef(odors[:,:,0].reshape(paras["N_odour"],paras["n_glo"]),rowvar=False)
    correl= (correl+1.0)/20.0 # extra factor 10 in comparison to covariance ...
    for i in range(paras["n_glo"]):
        correl[i,i]= 0.0
    print("AL inhibition with correlation, no self-inhibition")
else:
    if connect_I == "corr1":
        correl= np.corrcoef(odors[:,:,0].reshape(paras["N_odour"],paras["n_glo"]),rowvar=False)
        correl= (correl+1.0)/20.0 # extra factor 10 in comparison to covariance ...
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
                 correl= None
                 print("Homogeneous AL inhibition")
                 

# Now, let's make a protocol where each odor is presented for 3 secs with
# 3 second breaks and at each of 24 concentration values
paras["protocol"]= []
t_off= 3000.0
base= np.power(10,0.25)

for c1 in range(24):
    for c2 in range(24):
        sub_prot= {
            "t": t_off,
            "odor": o1,
            "ochn": "0",
            "concentration": 1e-7*np.power(base,c1),
        }
        paras["protocol"].append(sub_prot)
        sub_prot= {
            "t": t_off,
            "odor": o2,
            "ochn": "1",
            "concentration": 1e-7*np.power(base,c2),
        }
        paras["protocol"].append(sub_prot)        
        sub_prot= {
            "t": t_off+3000.0,
            "odor": o1,
            "ochn": "0",
            "concentration": 0.0,
        }
        paras["protocol"].append(sub_prot)
        sub_prot= {
            "t": t_off+3000.0,
            "odor": o2,
            "ochn": "1",
            "concentration": 0.0,
        }
        paras["protocol"].append(sub_prot)
        t_off+= 6000.0;

paras["t_total"]= t_off
print("We are running for a total simulated time of {}ms".format(t_off))

state_bufs, spike_t, spike_ID= ALsim(odors, hill_exp, paras, lns_gsyn= correl)

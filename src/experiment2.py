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
paras["write_to_disk"]= True
paras["trial_time"]= 12000.0

if ino == -100:
    paras["lns_pns_g"]= 0
    paras["lns_lns_g"]= 0
else:
    paras["lns_pns_g"]*= np.power(10,ino)
    paras["lns_lns_g"]*= np.power(10,ino)

# write results into a dir with current date in the name
timestr = time.strftime("%Y-%m-%d")
paras["dirname"]= timestr+"-runs"

path = os.path.isdir(paras["dirname"])
if not path:
    print("making dir "+paras["dirname"])
    os.makedirs(paras["dirname"])

paras["use_spk_rec"]= True

paras["rec_state"]= [
]

paras["rec_spikes"]= [
    "PNs",
    ]

label= "run"
paras["label"]= label+"_"+connect_I+"_"+str(ino)+"_odors_"+str(o1)+"_"+str(o2)

# Load odors and Hill coefficients from file. This experiment assumes that these have been
# already generated with experiment1.py
hill_exp= np.load(paras["dirname"]+"/"+label+"_hill.npy")
odors= np.load(paras["dirname"]+"/"+label+"_odors.npy")
paras["N_odour"]= odors.shape[0]

# define the inhibitory connectivity pattern in the antennal lobe
correl= choose_inh_connectivity(paras,connect_I)

# Now, let's make a protocol where each odor is presented for 3 secs with
# 3 second breaks and at each of 25 concentration values
protocol= []
t_off= 3000.0
base= np.power(10,0.25)

for c1 in range(25):
    for c2 in range(25):
        sub_prot= {
            "t": t_off,
            "odor": o1,
            "ochn": "0",
            "concentration": 1e-7*np.power(base,c1),
        }
        protocol.append(sub_prot)
        sub_prot= {
            "t": t_off,
            "odor": o2,
            "ochn": "1",
            "concentration": 1e-7*np.power(base,c2),
        }
        protocol.append(sub_prot)        
        sub_prot= {
            "t": t_off+3000.0,
            "odor": o1,
            "ochn": "0",
            "concentration": 0.0,
        }
        protocol.append(sub_prot)
        sub_prot= {
            "t": t_off+3000.0,
            "odor": o2,
            "ochn": "1",
            "concentration": 0.0,
        }
        protocol.append(sub_prot)
        t_off+= paras["trial_time"];

paras["t_total"]= t_off
print("We are running for a total simulated time of {}ms".format(t_off))

state_bufs, spike_t, spike_ID, ORN_cnts= ALsim(odors, hill_exp, paras, protocol, lns_gsyn= correl)
